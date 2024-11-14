import mindspore
from bi_tempered_loss import *
#set device
mindspore.set_context(device_target='Ascend', device_id=0)

#same as pytorch
activations = mindspore.Tensor([[-0.5, 0.1, 2.0]], mindspore.float32)
labels = mindspore.Tensor([[0.2, 0.5, 0.3]], mindspore.float32)

# The standard logistic loss is obtained when t1 = t2 = 1.0
loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=1.0, t2=1.0)
print("Loss, t1=1.0, t2=1.0: ", loss)

# test model
loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=0.7, t2=1.3)
print("Loss, t1=0.7, t2=1.3: ", loss)
