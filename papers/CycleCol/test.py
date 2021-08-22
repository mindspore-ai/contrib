
#推理部分
from mindspore import load_checkpoint, load_param_into_net
ckpt_file_name = "./model/coloring_2-8_775.ckpt"
param_set = load_checkpoint(ckpt_file_name)
net=cyclecolornet()
load_param_into_net(net, param_set)
Loss3=SLoss()
optim=nn.RMSProp(params=NET.trainable_params(), learning_rate=0.001)
model=Model(net,loss_fn=Loss3,optimizer=optim)

testDataset=imgDataset(val)
testdataset=ds.GeneratorDataset(testDataset,column_names=['img','label'],num_parallel_workers=4)
testdataset=testdataset.batch(1)

testdata_iter = testdataset.create_dict_iterator()
testdata = next(testdata_iter)
#print(Tensor(testdata['img']).shape)
predicted = model.predict(Tensor(testdata['img']))
predicted=predicted[0]
