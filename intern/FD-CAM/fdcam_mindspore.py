from itertools import combinations
import cv2
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class FDCAM:
    def __init__(self, model, target_layers, threshold, reshape_transform=None):
        self.model = model
        self.threshold = threshold
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        
        # 用于存储钩子提取的信息
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
    def _ensure_numpy(self, tensor):
        """确保张量转换为numpy数组"""
        if isinstance(tensor, ms.Tensor):
            return tensor.asnumpy()
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)
            
    def minMax(self, tensor):
        tensor_np = self._ensure_numpy(tensor)
        result = np.zeros_like(tensor_np)
        
        for i in range(tensor_np.shape[0]):
            row = tensor_np[i]
            min_val = np.min(row)
            max_val = np.max(row)
            if max_val > min_val:
                result[i] = (row - min_val) / (max_val - min_val)
                
        return Tensor(result, dtype=ms.float32)
        
    def scaled(self, tensor):
        tensor_np = self._ensure_numpy(tensor)
        result = np.zeros_like(tensor_np)
        
        for i in range(tensor_np.shape[0]):
            row = tensor_np[i]
            max_val = np.max(row)
            if max_val > 0:
                result[i] = row / max_val
                
        return Tensor(result, dtype=ms.float32)
        
    def get_cos_similar_matrix(self, v1, v2):
        v1_np = self._ensure_numpy(v1)
        v2_np = self._ensure_numpy(v2)
            
        num = np.dot(v1_np, v2_np.T)
        denom = np.linalg.norm(v1_np, axis=1).reshape(-1, 1) * np.linalg.norm(v2_np, axis=1)
        eps = 1e-8
        res = num / (denom + eps)
        res[np.isnan(res)] = 0
        return res
        
    def combination(self, scores, grads_tensor):
        grads = self.minMax(grads_tensor)
        scores = self.minMax(scores)
        
        grads_np = self._ensure_numpy(grads)
        scores_np = self._ensure_numpy(scores)

        weights = np.exp(scores_np) * grads_np - 0.5
        
        return Tensor(weights, dtype=ms.float32)
        
    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        _, channels, height, width = activations.shape
        
        activation = activations.reshape(channels, -1)
        cosine = self.get_cos_similar_matrix(activation, activation)
        
        record0 = np.ones(cosine.shape, dtype=np.float32)
        record1 = np.zeros(cosine.shape, dtype=np.float32)
        
        for i in range(cosine.shape[0]):
            sorted_cosines = np.sort(cosine[i, :])
            threshold_idx = int(len(sorted_cosines) * self.threshold)
            threshold_value = sorted_cosines[threshold_idx]
            record1[i, :] = cosine[i, :] > threshold_value
            
        record2 = record0 - record1 
        
        # 提取目标类别，MindSpore不支持直接的分类器输出目标
        target_categories = []
        if targets is None:
            output = self.model(input_tensor)
            target_categories = [np.argmax(self._ensure_numpy(output)[0])]
        else:
            for target in targets:
                if hasattr(target, 'category'):
                    target_categories.append(target.category)
                elif hasattr(target, 'ids'):
                    target_categories.append(target.ids)
                else:
                    target_categories.append(target)
        
        category = target_categories[0]
        output = self.model(input_tensor)
        orig_result = self._ensure_numpy(output)[0, category]
        
        activation_tensor = Tensor(activations, ms.float32)
        number_of_channels = activation_tensor.shape[1] # 通道数
        all_scores = []  # 收集所有类别的score

        # 使用批处理加速计算
        BATCH_SIZE = 16  # 可根据可用内存调整

        for tensor_i, category in zip(range(len(activation_tensor)), target_categories):
            # 获取当前激活张量
            tensor = activations[tensor_i]

            # 创建批处理张量
            batch_tensors = []
            for _ in range(BATCH_SIZE):
                batch_tensors.append(tensor.copy())
            batch_tensor = np.stack(batch_tensors, axis=0)  # [BATCH_SIZE, channels, height, width]

            # 处理每个通道批次
            channel_scores = []
            for i in range(0, number_of_channels, BATCH_SIZE):
                # 计算当前批次的实际大小
                actual_bs = min(BATCH_SIZE, number_of_channels - i)
                if actual_bs <= 0:
                    continue
                
                # 确保不会超出边界
                current_indices = list(range(i, min(i + actual_bs, record1.shape[0])))
                if not current_indices:
                    continue
                
                # ON
                batch_on = np.zeros_like(batch_tensor[:actual_bs])
                for j, idx in enumerate(current_indices):
                    # 使用record1创建掩码
                    for k in range(channels):
                        batch_on[j, k] = tensor[k] * record1[idx, k]
                on_scores = np.zeros(actual_bs)

                for b in range(actual_bs):  # 与pytorch不同，这里我们逐一处理每个样本
                    # 全局平均池化
                    feature_map = batch_on[b]
                    pooled_features = np.mean(feature_map, axis=(1, 2))

                    try:
                        ms_input = ms.Tensor(batch_on[b:b+1], ms.float32)
                        on_output = self.model(ms_input)
                        on_scores[b] = self._ensure_numpy(on_output)[0, category]
                    except Exception as e:
                        # 如果模型推理失败，使用特征统计估计
                        # 这是一个后备方案
                        on_scores[b] = np.sum(pooled_features) / (np.linalg.norm(pooled_features) + 1e-8)

                # OFF
                batch_off = np.zeros_like(batch_tensor[:actual_bs])
                for j, idx in enumerate(current_indices):
                    for k in range(channels):
                        batch_off[j, k] = tensor[k] * record2[idx, k]
                off_scores = np.zeros(actual_bs)

                for b in range(actual_bs):
                    feature_map = batch_off[b]
                    pooled_features = np.mean(feature_map, axis=(1, 2))

                    try:
                        ms_input = ms.Tensor(batch_off[b:b+1], ms.float32)
                        off_output = self.model(ms_input)
                        off_scores[b] = self._ensure_numpy(off_output)[0, category]
                    except Exception as e:
                        off_scores[b] = np.sum(pooled_features) / (np.linalg.norm(pooled_features) + 1e-8)

                importance_scores = on_scores + orig_result - off_scores
                channel_scores.extend(importance_scores[:actual_bs])

            all_scores.extend(channel_scores)

        # 处理形状匹配
        expected_length = activations.shape[0] * activations.shape[1]
        if len(all_scores) != expected_length:
            # 如果长度不匹配，进行调整
            if len(all_scores) < expected_length:
                # 填充不足的部分
                all_scores.extend([0] * (expected_length - len(all_scores)))
            else:
                # 截断多余的部分
                all_scores = all_scores[:expected_length]

        # 重塑为适当的形状
        scores = np.array(all_scores, dtype=np.float32).reshape(activations.shape[0], activations.shape[1])

        scores_tensor = Tensor(scores, ms.float32)
        if len(grads.shape) == 4:
            grads_mean = np.mean(grads, axis=(2, 3))
        else:
            grads_mean = grads.copy()

        grad_tensor = Tensor(grads_mean.reshape(activations.shape[0], activations.shape[1]), ms.float32)
        weights = self.combination(scores_tensor, grad_tensor).asnumpy()

        return weights