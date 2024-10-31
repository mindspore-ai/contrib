import numpy as np

import mindspore as ms
import mindspore.ops as ops
import math
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

def normalization(embeds):
    norms = ops.norm(embeds, 2, 1, keepdim=True)
    return embeds / norms

def calculate_labse_score(x_s, x_a, model_labse):
    embeddings_s = ms.Tensor(model_labse.encode(x_s))
    embeddings_a = ms.Tensor(model_labse.encode(x_a))

    embeddings_s = normalization(embeddings_s)
    embeddings_a = normalization(embeddings_a)

    labse_score = ops.matmul(embeddings_s, embeddings_a.T)
    return labse_score


def manual_softmax(logits):
    logits = np.array(logits)
    # 计算每个元素的指数
    exp_logits = np.exp(logits)

    # 计算指数和
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)

    # 计算 softmax
    probs = exp_logits / sum_exp_logits

    return probs
def get_msp_and_prediction(model_pre, tokenizer_pre, x_s, x_a):
    inputs = tokenizer_pre(x_s, x_a, return_tensors='pt', truncation=True, padding=True)
    outputs = model_pre(**inputs)
    logits = outputs.logits


    probs = manual_softmax(logits_data)

    logits_data_list=logits_data.tolist()

    pred_label, _ = max(enumerate(logits_data_list[0]), key=lambda x: x[1])

    score_msp = probs.max()

    return pred_label, score_msp

def SAMPLING(Target_data, model_labse, model_pre, tokenizer_pre, batch_size, threshold_MSP=0.5, pbar=True):
    """
    对目标数据进行抽样处理。

    本函数旨在对给定的目标数据进行过滤和评分，以选取符合特定条件的数据项。它通过计算LABSE分数和MSP分数来评估每对数据项，
    并基于这些分数和给定的阈值来选择数据项。

    参数:
    - Target_data: 待处理的目标数据列表，每个数据项为一个包含两个元素的元组。
    - model_labse: 用于计算LABSE分数的模型。
    - model_pre: 用于获取MSP分数和预测标签的模型。
    - tokenizer_pre: 用于预处理文本的tokenizer。
    - batch_size: 批处理大小。
    - threshold_MSP: MSP分数的阈值，默认为0.5。
    - pbar: 是否显示进度条，默认为True。

    返回:
    - final_batch: 经过滤和评分后，选出的符合特定条件的数据项列表。
    """
    # 初始化干净数据列表，用于存储非空的数据项
    clean_data = []

    # 遍历目标数据，移除空的数据项
    for sample in Target_data:
        x_s, x_a = sample
        if len(x_s.strip()) > 0 and len(x_a.strip()) > 0:
            clean_data.append(sample)

    # 计算批处理数量
    num_batches = math.ceil(len(clean_data) / batch_size)

    # 初始化选中的批次和临时LABSE分数列表
    selected_batch = []
    temp_scores_L = []

    # 根据pbar参数决定是否显示进度条
    if pbar:
        # 使用tqdm库显示进度条
        for i in tqdm(range(num_batches)):
            # 获取当前批次的数据
            batch = clean_data[i * batch_size : i * batch_size + batch_size]

            # 遍历批次中的每个数据项
            for x_s, x_a in batch:
                # 计算并保存LABSE分数
                score_L = calculate_labse_score([x_s], [x_a], model_labse)
                score_L_item=score_L.asnumpy()
                temp_scores_L.append(score_L_item[0][0])

                # 获取MSP分数和预测标签
                y, score_MSP = get_msp_and_prediction(model_pre, tokenizer_pre, x_s, x_a)

                # 如果MSP分数高于阈值，则将数据项添加到选中批次
                if score_MSP > threshold_MSP:
                    selected_batch.append((x_s, x_a, y, score_L))

    else:
        # 不显示进度条的情况下处理数据
        for i in range(num_batches):
            batch = clean_data[i * batch_size : i * batch_size + batch_size]

            for x_s, x_a in batch:
                score_L = calculate_labse_score([x_s], [x_a], model_labse)
                temp_scores_L.append(score_L.item())

                y, score_MSP = get_msp_and_prediction(model_pre, tokenizer_pre, x_s, x_a)

                if score_MSP > threshold_MSP:
                    selected_batch.append((x_s, x_a, y, score_L))

    # 计算LABSE分数的平均值
    avg_L = np.mean(temp_scores_L)

    # 基于LABSE分数的平均值进一步过滤选中的批次
    final_batch = [sample for sample in selected_batch if sample[3].asnumpy() > avg_L]

    # 返回最终的批次
    return final_batch


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="Sampling with LaBSE and a classification model.")
    # parser.add_argument('--target_data', type=str, required=True, default='./data.pkl', help="Path to the target data file.")
    # parser.add_argument('--model_labse', type=str, required=True, default='./LaBSE', help="Pretrained LaBSE model identifier.")
    # parser.add_argument('--model_pre', type=str, required=True, default='./KPC', help="Pretrained model identifier for Ψpre.")
    # parser.add_argument('--batch_size', type=int,  default=1, help="Batch size for processing.")
    # parser.add_argument('--threshold_msp', type=float, default=0.5, help="Threshold for MSP score.")
    # parser.add_argument('--pbar', type=bool, default=True, help="Show progress bar (True/False).")
    #
    # args = parser.parse_args()

    # 手动设置参数
    args = {
        'target_data': './data.pkl',
        'model_labse': './LaBSE',
        'model_pre': './KPC',
        'batch_size': 1,
        'threshold_msp': 0.5,
        'pbar': True
    }

    # 将字典转换为命名空间对象
    from types import SimpleNamespace

    args = SimpleNamespace(**args)

    model_labse = SentenceTransformer('LaBSE')  # Using sentence-transformers for LaBSE

    model_pre = AutoModelForSequenceClassification.from_pretrained('KPC')
    tokenizer_pre = AutoTokenizer.from_pretrained('KPC')

    # Load target data
    # Here we assume target data is a list of tuples (x_s, x_a)
    Target_data = [("review sentence", "aspect")]

    final_batch = SAMPLING(Target_data, model_labse, model_pre, tokenizer_pre, args.batch_size, args.threshold_msp, args.pbar)

    #print(f"Final Batch Size: {len(final_batch)}")
    #print(final_batch)
    
    with open("final_batch.pkl", "wb") as f:
        pickle.dump([(x_s, x_a, y) for (x_s, x_a, y, _) in final_batch], f)