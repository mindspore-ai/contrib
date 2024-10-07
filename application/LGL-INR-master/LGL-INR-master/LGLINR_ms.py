
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
class LGL_INR(nn.Cell):
    
    def __init__(self, num_classes, beta=1, detach_weight=False):
        super(LGL_INR, self).__init__()
        self.num_classes = num_classes
        # self.use_ascend = use_Ascend
        self.sigmoid = ops.Sigmoid()
        self.beta = beta
        self.detach_weight=detach_weight # whether to allow the weight in backpropagation 
    
    def construct(self, inputs, targets):
        """
        Args: 
            inputs: predicted logit, of shape (batch_size, num_classes)
            targets: true label of shape (batch_size)
        """
        eps = 1e-7
        probs = self.sigmoid(inputs)

        # 创建与 probs 大小相同的全零张量
        zeros = ops.operations.ZerosLike()(probs)

        # 将 targets 转换为 one-hot 编码
        targets = ops.one_hot(targets, probs.shape[1], ms.Tensor(1, ms.float32), ms.Tensor(0, ms.float32))

        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        size_class = targets.sum(axis=0) # size per class in minibatch
        
        loss_pos = 0 # loss of positive class
        mean_probs = list()
        mean_loss_neg = list()
        for k in range(self.num_classes): # class-wise processing
            idx_k = targets[:,k].type(ms.bool_) # indicator if sample belongs to class k
            targets_k = targets[idx_k]
            if targets_k.shape[0] == 0: # some class may have no samples in minibatch
                continue
            probs_k = probs[idx_k]
            mean_probs_k = probs_k.mean(axis=0)
            mean_probs_k = (mean_probs_k.asnumpy())
            # mean_probs_k[k] = -1 # mask out the probability of true label
            if self.detach_weight==True:
                mean_probs_k = ops.stop_gradient(mean_probs_k)
            mean_probs_k=ms.Tensor(mean_probs_k,dtype=ms.float32)
            mean_probs.append(mean_probs_k)
            
            logp_pos_k = ops.log(probs_k+eps)
            logp_neg_k = ops.log(1.0-probs_k+eps)
            loss_pos = loss_pos + (targets_k * logp_pos_k).sum()/size_class[k] # sumup 
            mean_loss_neg_k =  ((1.0-targets_k) * logp_neg_k).mean(axis=0) # no sumup, need reweighting
            mean_loss_neg.append(mean_loss_neg_k)
        # mean_probs=ms.Tensor(mean_probs,dtype=ms.float32)
        mean_probs = ops.stack(mean_probs, axis=0)
        mean_loss_neg = ops.stack(mean_loss_neg, axis=0)
        
        loss_neg = 0 # loss of reweighted negative class
        for k in range(self.num_classes):
            idx_neg = (mean_probs[:,k]!=-1).type(ms.bool_)
            prob_neg_k = mean_probs[idx_neg,k]
            weight_neg = ops.softmax(self.beta * prob_neg_k, axis=0)
            loss_neg_k = (weight_neg * mean_loss_neg[idx_neg,k]).sum()
            loss_neg = loss_neg + loss_neg_k
        
        loss = -1 * (loss_pos + loss_neg)
        return loss

import unittest
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context


class TestLGL_INR(unittest.TestCase):
    def setUp(self):
        # 设置测试环境和实例化LGL_INR对象
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        self.num_classes = 5
        self.beta = 1.0
        self.detach_weight = False
        self.model = LGL_INR(self.num_classes, self.beta, self.detach_weight)

    def test_LGL_INR_forward(self):
        # 构建输入和目标数据
        inputs = ms.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=ms.float32)
        targets = ms.Tensor([0], dtype=ms.int32)

        # 前向传播
        loss = self.model(inputs, targets)

        # 验证loss是否正确计算
        self.assertIsNotNone(loss, "Loss should not be None")
        self.assertEqual(loss.dtype, ms.float32, "Loss should be of type float32")

    def test_LGL_INR_beta_detach(self):
        # 测试不同的beta值和detach_weight设置
        inputs = ms.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=ms.float32)
        targets = ms.Tensor([0], dtype=ms.int32)
        beta = 2.0
        detach_weight = True

        # 重新实例化模型
        model = LGL_INR(self.num_classes, beta, detach_weight)

        # 前向传播
        loss = model(inputs, targets)

        # 验证loss是否正确计算
        self.assertIsNotNone(loss, "Loss should not be None")
        self.assertEqual(loss.dtype, ms.float32, "Loss should be of type float32")

if __name__ == '__main__':
    unittest.main()
