import mindspore as ms
import numpy as np
import json
import argparse


class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<pad>": 0, "<eos>": 1, "[writer_intent]": 2, "[effect_on_reader]": 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text, return_tensors=None):
        tokens = [self.vocab.get(tok, 0) for tok in text.split()]
        return ms.Tensor([tokens], dtype=ms.int32)

    def decode(self, tokens):
        return ' '.join([self.inv_vocab.get(tok, "<unk>") for tok in tokens])

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)


class SimpleModel:
    def __init__(self):
        pass

    def generate(self, input_ids, num_beams=1, top_k=None, max_length=10):
        batch_size, seq_length = input_ids.shape
        output = ms.Tensor(np.random.randint(2, 4, size=(batch_size, max_length)), dtype=ms.int32)
        return output


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="t5")
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--input_file', type=str, default='./example.txt')
parser.add_argument('--output_file', type=str, default='./output.jsonl')
parser.add_argument('--decoding', type=str, default='beam_search')
args = parser.parse_args()
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

tokenizer = SimpleTokenizer()
model = SimpleModel()

dims = ["[writer_intent]", "[effect_on_reader]", "[reader_action]", "[pred_label]", "[gold_label]", "[spread]"]
domains = ["[climate]", "[covid]", "[cancer]", "[Other]"]
tokenizer.add_tokens(dims + domains)

output_file = open(args.output_file, "w")
for headline in open(args.input_file).readlines():
    headline = headline.strip()
    line_ = {"headline": headline}

    for dim in dims:
        input_ = tokenizer.encode(headline + " " + dim, return_tensors="ms")
        output_ = model.generate(input_ids=input_, num_beams=3 if args.decoding == "beam_search" else 1, max_length=10)
        output_text = tokenizer.decode(output_[0].asnumpy())
        line_[dim] = output_text

    output_file.write(str(json.dumps(line_)) + "\n")

output_file.close()