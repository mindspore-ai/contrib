from __future__ import print_function
import sys

sys.path.append("Your Project dir")

if len(sys.argv) != 6:
    print('Usage:')
    print('python train.py datacfg darknetcfg weightfile')
    exit()

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

import mindspore
import mindspore.ops as ops
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import py_transforms as transforms

from core import dataset
from core.utils import *
from core.cfg import parse_cfg, cfg
from tool.darknet.darknet_decoupling import Darknet

# Training settings
datacfg       = sys.argv[1]
darknetcfg    = parse_cfg(sys.argv[2])
learnetcfg    = parse_cfg(sys.argv[3])
weightfile    = sys.argv[4]

data_options  = read_data_cfg(datacfg)
net_options   = darknetcfg[0]
meta_options  = learnetcfg[0]

# Configure options
cfg.config_data(data_options)
cfg.config_meta(meta_options)
cfg.config_net(net_options)

# Parameters 
metadict      = data_options['meta']
trainlist     = data_options['train']

testlist      = data_options['valid']
backupdir     = data_options['backup']
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
use_cuda      = True
#seed          = int(time.time())
seed = 1
eps           = 1e-5
dot_interval  = 70  # batches
# save_interval = 10  # epoches

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

## --------------------------------------------------------------------------
## MAIN
#backupdir = cfg.backup
backupdir    = sys.argv[5]

print('logging to ' + backupdir)
if not os.path.exists(backupdir):
    os.mkdir(backupdir)

mindspore.set_seed(seed)

model       = Darknet(darknetcfg, learnetcfg)
region_loss = model.loss

#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(weightfile))

model.load_weights(weightfile)
model.print_network()


###################################################
### Meta-model parameters
region_loss.seen  = model.seen
processed_batches = 0 if cfg.tuning else model.seen/batch_size
trainlist         = dataset.build_dataset(data_options)
nsamples          = len(trainlist)
init_width        = model.width
init_height       = model.height
init_epoch        = 0 if cfg.tuning else model.seen/nsamples
max_epochs        = max_batches*batch_size/nsamples+1
max_epochs        = int(math.ceil(cfg.max_epoch*1./cfg.repeat)) if cfg.tuning else max_epochs 
print(cfg.repeat, nsamples, max_batches, batch_size)
print(num_workers)

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

# Adjust learning rate
factor = len(test_metaset.classes)
if cfg.neg_ratio == 'full':
    factor = 15.
elif cfg.neg_ratio == 1:
    factor = 3.0
elif cfg.neg_ratio == 0:
    factor = 1.5
elif cfg.neg_ratio == 5:
    factor = 8.0

print('factor:', factor)
learning_rate /= factor


optimizer = nn.SGD(model.trainable_params(),learning_rate=learning_rate, 0.9)


def Inverted_gradient(feature, ratio, mask):
    """
        feature: rpn features
        ratio: how many to be inverted
    """
    mask_all = []
    mask_count = []
    for rpn_feat in feature:
        num_batch = rpn_feat.shape[0]
        num_channel = rpn_feat.shape[1]
        num_height = rpn_feat.shape[2]
        num_width = rpn_feat.shape[3]
        
        rpn_feat = ops.abs(rpn_feat)
        rpn_feat = rpn_feat * mask
        for batch in range(num_batch):
            mask_count.append(mindspore.numpy.sum(mask[batch]))
        feat_channel_mean = mindspore.numpy.mean(rpn_feat.view(num_batch, num_channel, -1), dim=2)
        feat_channel_mean = feat_channel_mean.view(num_batch, num_channel, 1, 1)
        cam = mindspore.numpy.sum(rpn_feat * feat_channel_mean, 1) #[B 1 H W]
        cam_tmp = cam.view(num_batch, num_height*num_width)
        cam_tmp_sort, cam_tmp_indice = ops.Sort(cam_tmp, axis  = 1, descending = True)
        for batch in range(num_batch):
            th_idx = int(ratio * mask_count[batch])
            
            threshold = cam_tmp_sort[batch][th_idx - 1]
            threshold_map = threshold * ops.ones(1, num_height, num_width)
            mask_all_cuda = mindspore.numpy.where(cam[batch] > threshold_map, ops.zeros(cam[batch].shape),
                                ops.ones(cam[batch].shape))
            mask_all_cuda = mask_all_cuda.view(1,1,num_height,num_width)
            if batch == 0:
                result = mask_all_cuda
            else:
                result = ops.Concat(result,mask_all_cuda,0)
        mask_all.append(result)

    return mask_all


class TrainOnesStepCell(nn.Cell):
    def __init__(self, network, optimizer, loss_function, sens=1.0, mask_ratio=0.15, repeat_time=1, reduce_flag=False, mean=True, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.grad = ops.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.sens = Tensor([sens,], mstype.float32)
        self.repeat_time = repeat_time
        self.mask_ratio = mask_ratio
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
            
    def fn(self, image_mask, grad):
        mask = Inverted_gradient(grad, self.mask_ratio, image_mask)
        return mask[0]

    def construct(self, img, metax, mask, target, target_cls_ids):
	# For loop means the same input "x" will train many times and will be modifyed according to gradient many times
        loss = 0
        for i in range(self.repeat_time):
            weights = self.weights
            net_output, dynamic_weight = self.network(img, metax, mask)
            loss = self.loss_function(net_output, target, dynamic_weights, target_cls_ids)
            grads = self.grad(self.network, weights)(img, metax, mask, self.sens)
            
            if self.reduce_flag:
                grads = self.grad_reducer(grads)

            mask = fn(mask, grads)

            ops.depend(loss, self.optimizer(grads))
        
        return loss


for epoch in range(int(init_epoch), int(max_epochs)):
    if cfg.tuning:
        if "1shot" in metadict:
            shot_num = 1
        elif "2shot" in metadict:
            shot_num = 2
        elif "3shot" in metadict:
            shot_num = 3
        elif "5shot" in metadict:
            shot_num = 5
        elif "10shot" in metadict:
            shot_num = 10
        else:
            print("error!")
        repeat_time = 13 - shot_num
        mask_ratio = 0.15
    else:
        repeat_time = 1
        mask_ratio = 0
        
    train(epoch,repeat_time,mask_ratio)


def train(epoch, repeat_time, mask_ratio):
    global processed_batches
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model


    generator = dataset.listDataset(trainlist, shape=(init_width, init_height),
                            shuffle=False,
                            transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]),
                            train=True,
                            seen=cur_model.seen,
                            batch_size=batch_size,
                            num_workers=num_workers, **kwargs)
    
    dataset = GeneratorDataset(generator, column_names=['img']).batch(batch_size)
    
    meta_generator = dataset.MetaDataset(metafiles=metadict, train=True, with_ids=True)
    metaloader = GeneratorDataset(generator, column_names=['img']).batch(metaset.batch_size)
    
    metaloader = iter(metaloader)

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d/%d, processed %d samples, lr %f' % (epoch, max_epochs, epoch * len(train_loader.dataset), lr))

    model.train()
    novel_id = cfg['novel_ids']
    
    for batch_idx, (data, target) in enumerate(train_loader):
        metax, mask, target_cls_ids = metaloader.next()
        
        novel_cls_flag = ops.zeros(len(target_cls_ids))
        for index,j in enumerate(target_cls_ids):
            #print(index)
            if j in novel_id:
                #print("flag",index)
                novel_cls_flag[int(index)] = 0
            else:
                novel_cls_flag[int(index)] = 1
                
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        data, target = Parameter(data), Parameter(target)
        metax, mask, target_cls_ids = Parameter(metax), Parameter(mask), Parameter(target_cls_ids)
        for i in range(repeat_time):
            optimizer.zero_grad()
            metax_disturbance = metax
            if i == 0:
                mask_disturbance = mask
            else:
                for index, flag_each in enumerate(novel_cls_flag):
                    if flag_each == 0:
                        mask_disturbance[index] = mask[index]
                    elif flag_each == 1:
                        mask_disturbance[index] = mask[index] * metax_mask[0][index]
                    else:
                        print("error")
            output, dynamic_weights = model(data, metax_disturbance, mask_disturbance)
            region_loss.seen = region_loss.seen + data.data.size(0)
            if i == 0:
                loss = region_loss(output, target, dynamic_weights, target_cls_ids)
                dynamic_weights_store = dynamic_weights
                target_cls_ids_store = target_cls_ids
                dynamic_weight_buffer = dynamic_weights
            else:
                with torch.no_grad():
                    for index, flag_each in enumerate(novel_cls_flag):
                        if flag_each == 1:
                            dynamic_weights_store = [ops.Concat((dynamic_weights_store[0],ops.ExpandDims(dynamic_weights[0][index],0)),dim = 0)]
                        else:
                            continue
                    for num in range(int((novel_cls_flag).sum() // len(novel_id))):
                        Tensor_novel_id = Tensor(novel_id, dtype=mindspore.int16)
                        target_cls_ids_store = [ops.Concat((target_cls_ids_store,Tensor_novel_id),0)
                loss = region_loss(output, target, dynamic_weights_store, target_cls_ids_store)
                
            loss.backward()
            metax_mask = Inverted_gradient([metax.grad], mask_ratio, mask)
            
            optimizer.step()
    logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))

    if (epoch+1) % cfg.save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))


evaluate = False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
    for epoch in range(int(init_epoch), int(max_epochs)):
        if cfg.tuning:
            if "1shot" in metadict:
                shot_num = 1
            elif "2shot" in metadict:
                shot_num = 2
            elif "3shot" in metadict:
                shot_num = 3
            elif "5shot" in metadict:
                shot_num = 5
            elif "10shot" in metadict:
                shot_num = 10
            else:
                print("error!")
            repeat_time = 13 - shot_num
            mask_ratio = 0.15
        else:
            repeat_time = 1
            mask_ratio = 0
        
        train(epoch,repeat_time,mask_ratio)