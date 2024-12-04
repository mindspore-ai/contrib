# Zoneout-RNN
A Zoneout implemetion based on mindspore

What is Zoneout? Here is the paper：

[Regularizing RNNs by Randomly Preserving Hidden Activations](https://arxiv.org/abs/1606.01305)

Zoneout 的基本思想是通过随机地保留 RNN 单元的状态，而不是像 Dropout 那样随机地丢弃神经元的输出。具体来说，Zoneout 在每个时间步随机选择一些状态单元，并将它们保持不变（即保留），而不是更新它们。这种保留状态的方式有助于模型在训练过程中保持更多的信息，从而提高模型的泛化能力和稳定性。
