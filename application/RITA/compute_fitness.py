import mindspore
from mindnlp.transformers import GPT2Tokenizer, GPT2LMHeadModel
from mindspore import nn, Tensor, context
import numpy as np
import tqdm

def calc_fitness(model, prots, tokenizer, device_target='CPU', model_context_len=1023):
    loss_list = []
    loss_fn = nn.CrossEntropyLoss()
    model.set_train(False)  # 设置模型为评估模式

    for prot in tqdm.tqdm(prots):
        loss_val = 0

        sequence_chunks = []
        if len(prot) < model_context_len:
            sequence_chunks = [prot]
        else:
            len_target_seq = len(prot)
            num_windows = 1 + int(len_target_seq / model_context_len)
            start = 0
            for window_index in range(1, num_windows + 1):
                sequence_chunks.append(prot[start:start + model_context_len])
                start += model_context_len

        for chunk in sequence_chunks:
            for p in [chunk, chunk[::-1]]:
                ids = tokenizer.encode(p)
                ids = Tensor([ids], mindspore.int32)
                input_ids = ids[:, :-1]
                targets = ids[:, 1:]

                outputs = model(input_ids)
                logits = outputs.logits  # 获取 logits

                logits = logits.reshape(-1, logits.shape[-1])
                targets = targets.reshape(-1)

                loss = loss_fn(logits, targets)
                loss_val += -loss.asnumpy().item()

        loss_list.append(loss_val)
    return np.array(loss_list)

def get_mutated_sequence(focus_seq, mutant, start_idx=1, AA_vocab="ACDEFGHIKLMNPQRSTVWY"):
    """
    通过给定的突变信息（仅限替换），对输入序列（focus_seq）进行突变。
    突变信息通常基于1索引：start_idx用于切换到0索引。
    """
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with mutant: " + str(mutation))
            continue
        relative_position = position - start_idx
        assert (from_AA == focus_seq[relative_position]), "Invalid from_AA or mutant position: " + str(mutation) + \
            " from_AA: " + str(from_AA) + " relative pos: " + str(relative_position) + " focus_seq: " + str(focus_seq)
        assert (to_AA in AA_vocab), "Mutant to_AA is invalid: " + str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)

# 测试代码
if __name__ == "__main__":
    # 设置MindSpore的运行模式和设备
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    # 加载模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # 示例蛋白质序列
    focus_seq = 'ACDEFGHIKLMNPQRSTVWY'
    mutant = 'A1C:D3E'  # 修正了突变信息
    mutated_seq = get_mutated_sequence(focus_seq, mutant)
    print("Mutated sequence:", mutated_seq)

    prots = [mutated_seq, focus_seq]

    # 计算适应度得分
    fitness_scores = calc_fitness(model, prots, tokenizer)
    print("Fitness scores:", fitness_scores)