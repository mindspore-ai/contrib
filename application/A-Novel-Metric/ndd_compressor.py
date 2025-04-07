import numpy as np
from mindspore import Tensor, ops
from mindnlp.transformers import BertTokenizer, BertForMaskedLM
from nltk.tokenize.treebank import TreebankWordDetokenizer
import spacy
from spacy.tokens import Doc
from copy import deepcopy
from utils import overlap, OSS
from tqdm import tqdm

class Compressor:
    def __init__(self, bert="bert-base-cased", device="Ascend", spacy_name='en_core_web_sm'):
        # 加载 Tokenizer 和模型
        self.tokenizer = BertTokenizer.from_pretrained(bert)
        self.model = BertForMaskedLM.from_pretrained(bert)
        self.device = device  # MindSpore 默认使用 Ascend 或 CPU
        self.model.set_train(False)  # 设置为推理模式
        self.nlp, self.nlp_dp = spacy.load(spacy_name), spacy.load(spacy_name)
        self.nlp_dp.tokenizer = lambda tokens: Doc(self.nlp_dp.vocab, tokens)
        self.detokenizer = TreebankWordDetokenizer()

    def NeighboringDistribution(self, tokens, neighbors):
        ids_batch, masks = [], []
        n_neighbor = len(neighbors)
        for idx in neighbors:
            tokens_masked = deepcopy(tokens)
            tokens_masked[idx] = self.tokenizer.mask_token
            items = self.tokenizer(' '.join(tokens_masked))
            ids_batch.append(items['input_ids'])
            masks.append(items['attention_mask'])

        max_len = max([len(ids) for ids in ids_batch])
        ids_batch_padded = [[self.tokenizer.pad_token_id for _ in range(max_len)] for ids in ids_batch]
        masks_padded = [[0 for _ in range(max_len)] for mask in masks]

        for idx in range(n_neighbor):
            ids_batch_padded[idx][:len(ids_batch[idx])] = ids_batch[idx]
            masks_padded[idx][:len(masks[idx])] = masks[idx]

        ids_batch_padded = Tensor(ids_batch_padded, dtype=mindspore.int32)
        masks_padded = Tensor(masks_padded, dtype=mindspore.int32)

        mask_pos = ops.argmax((ids_batch_padded == self.tokenizer.mask_token_id).astype(mindspore.float32), axis=1)
        logits = self.model(ids_batch_padded, masks_padded)['logits']
        logits = ops.softmax(logits, axis=-1)
        return ops.stack([logit[mask_pos[idx]] for idx, logit in enumerate(logits)], axis=0)

    def NeighboringDistributionDivergence(self, tokens, span, start, end, decay_rate, biased, bias_rate):
        tokens_ = tokens[:start] + span + tokens[end:]
        start_, end_ = start, start + len(span)
        neighbors = [idx for idx in range(len(tokens)) if idx < start or idx >= end]
        neighbors_ = [idx for idx in range(len(tokens_)) if idx < start_ or idx >= end_]
        sc = self.NeighboringDistribution(tokens, neighbors)
        sc_ = self.NeighboringDistribution(tokens_, neighbors_)
        w = np.array(list(np.arange(start)[::-1]) + list(np.arange(len(tokens) - end)))
        w_ = w + (len(w) - w[::-1])
        w = Tensor(np.power(decay_rate, w), dtype=mindspore.float32) + Tensor(np.power(decay_rate, w_), dtype=mindspore.float32)
        if biased:
            b = np.power(bias_rate, np.arange(len(w)))
            w = w * Tensor(b, dtype=mindspore.float32)
        ndd = ((sc_ * (sc_ / sc).log()).sum(1) * w).sum(0)
        return ndd.asnumpy().item()

    def SpanSearch(self, sent, head, max_span, threshold, decay_rate, biased, bias_rate):
        spans = []
        length = len(sent)
        candidates = [(start, end) for start in range(length) for end in range(start + 1, min(length, start + max_span))]
        bar = tqdm(candidates)
        bar.set_description("Compressing...")
        for candidate in bar:
            start, end = candidate
            ndd = self.NeighboringDistributionDivergence(sent, [], start, end, decay_rate, biased, bias_rate)
            if ndd < threshold:
                spans.append({'ids': np.arange(start, end), 'ndd': ndd})
        return spans

    def Compress(self, sent, max_itr=5, max_span=5, max_len=50, threshold=1.0, decay_rate=0.9, biased=False, bias_rate=0.98):
        logs = {}
        sent = [token.text for token in self.nlp(sent)]
        logs["Base"] = self.detokenizer.detokenize(sent)

        if len(sent) <= max_len:
            for itr in range(max_itr):
                head = [token.head.i for token in self.nlp_dp(sent)]
                spans = self.SpanSearch(sent, head, max_span, threshold, decay_rate, biased, bias_rate)
                spans = OSS(spans)
                if len(spans) == 0:
                    break
                span_ids = [idx for span in spans for idx in span['ids']]
                sent = [sent[idx] for idx in range(len(sent)) if idx not in span_ids]
                logs[f"Iter{itr}"] = self.detokenizer.detokenize(sent)
        return logs