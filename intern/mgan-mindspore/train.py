from PIL import Image
import sys
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops, grad
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.experimental import optim
from models import Generator, Discriminator

mindspore.set_context(pynative_synchronize=True)

batch_size = 1
lambda_cycle = 1
lambda_identity = 2
lr = 0.0001
seed = 0
mindspore.set_seed(seed)
print_every = 200
n_epochs = 60
input_shape = (216, 176)
odir = '/root/mgan-mindspore'
mindspore.set_context(device_target="GPU")
def crop_image(img):
    return img[ 1:-1, 1:-1,:,]
# Init dataset
transformer1 = transforms.Compose([
    vision.Decode(),
    vision.RandomHorizontalFlip(),
    ])
transformer2 =transforms.Compose([
        vision.ToPIL(),
        vision.ToTensor(),
        vision.Normalize([0.5,0.5,0.5] , [0.5,0.5,0.5],is_hwc=False)  
    ])
class MyMapDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_list = list(dataset.create_dict_iterator(num_epochs=1, output_numpy=True))
    def __getitem__(self, index):
        data = self.data_list[index]
        return data['image'], data['label']
    def __len__(self):
        return len(self.data_list)
    def map(self, operations, input_columns):
        self.dataset = self.dataset.map(operations=operations, input_columns=input_columns)
        self.data_list = list(self.dataset.create_dict_iterator(num_epochs=1, output_numpy=True))
        return self  
    def create_dict_iterator(self,num_epochs, output_numpy):
        return self.dataset.create_dict_iterator(num_epochs, output_numpy) 
    def get_dataset(self):
        return self.dataset   
dataset = ds.ImageFolderDataset('data/celeba/')
mymap_dataset = MyMapDataset(dataset)
mymap_dataset = mymap_dataset.map(operations=transformer1, input_columns="image")
mymap_dataset = mymap_dataset.map(operations=crop_image, input_columns="image")
mymap_dataset = mymap_dataset.map(operations=transformer2, input_columns="image")



labels_neg = []
labels_pos = []
# num=0
for i, data in enumerate(mymap_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
    label = data['label'].item()  
    if label == 0:
        labels_neg.append(i)  
    elif label == 1:
        labels_pos.append(i)  
print("labels_neg.append completed")
sampler_neg = ds.SubsetRandomSampler(labels_neg)
sampler_pos = ds.SubsetRandomSampler(labels_pos)

pos_loader = ds.GeneratorDataset(mymap_dataset,column_names=['image', 'label'],sampler=sampler_pos)
pos_loader = pos_loader.batch(batch_size)
neg_loader = ds.GeneratorDataset(mymap_dataset,column_names=['image', 'label'],sampler=sampler_neg)
neg_loader = neg_loader.batch(batch_size)
# Init models
netDP = Discriminator()
netDN = Discriminator()
netP2N = Generator()
netN2P = Generator()

criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
criterion_gan = nn.MSELoss()

# Init tensors
real_pos = ops.zeros((batch_size,3, 
                       input_shape[0], input_shape[1])
                      )
real_neg = ops.zeros((batch_size,3,  
                       input_shape[0], input_shape[1])
                      )
real_lbl = ops.zeros((batch_size, 1))
real_lbl[:, 0] = 1
fake_lbl = ops.zeros((batch_size, 1))
fake_lbl[:, 0] = -1
def add_prefix_to_params(model, prefix):
    for name, param in model.parameters_and_names():
        param.name = f"{prefix}.{name}"
    return model
add_prefix_to_params(netP2N,'netP2N')
add_prefix_to_params(netP2N,'netN2P')
add_prefix_to_params(netDN,'netDN')
add_prefix_to_params(netDP,'netDP')
opt_G = optim.Adam(list(netP2N.get_parameters())+list(netN2P.get_parameters()), lr=lr, betas=(0.5,0.999))
opt_D = optim.Adam(list(netDN.get_parameters())+list(netDP.get_parameters()), lr=lr, betas=(0.5,0.999))

scheduler_G = optim.lr_scheduler.StepLR(opt_G, step_size=10, gamma=0.317)
scheduler_D = optim.lr_scheduler.StepLR(opt_D, step_size=10, gamma=0.317)

netDN.set_train()
netDP.set_train()
netP2N.set_train()
netN2P.set_train()

print('Training...')
for epoch in range(n_epochs):
    batch = 0

for epoch in range(n_epochs):
    scheduler_G.step()
    scheduler_D.step()
    batch = 0

    for (pos, _), (neg, _) in zip(pos_loader, neg_loader):

        real_pos=pos.copy()
        real_neg=neg.copy()

        # Train P2N Generator
        real_pos_v = Tensor(real_pos)
        fake_neg, mask_neg = netP2N(real_pos_v)
        rec_pos, _ = netN2P(fake_neg)
        fake_neg_lbl = netDN(fake_neg)
        loss_P2N_cyc = criterion_cycle(rec_pos, real_pos_v)
        loss_P2N_gan = criterion_gan(fake_neg_lbl, Tensor(real_lbl))
        loss_N2P_idnt = criterion_identity(fake_neg, real_pos_v)

        # Train N2P Generator
        real_neg_v = Tensor(real_neg)
        fake_pos, mask_pos = netN2P(real_neg_v)
       
        rec_neg, _ = netP2N(fake_pos)

        fake_pos_lbl = netDP(fake_pos)
        loss_N2P_cyc = criterion_cycle(rec_neg, real_neg_v)
        loss_N2P_gan = criterion_gan(fake_pos_lbl, Tensor(real_lbl))
        loss_P2N_idnt = criterion_identity(fake_pos, real_neg_v)

        loss_G = ((loss_P2N_gan + loss_N2P_gan)*0.5 +
                  (loss_P2N_cyc + loss_N2P_cyc)*lambda_cycle +
                  (loss_P2N_idnt + loss_N2P_idnt)*lambda_identity)
        def forward_G(loss_G):
            
            return loss_G

        grad_G = mindspore.value_and_grad(forward_G, grad_position=None, weights=list(opt_G.get_parameters()))
        loss_G, grads_G = grad_G(loss_G)
        
        opt_G(grads_G)
        # Train Discriminators
        fake_neg_score = netDN(fake_neg)  
        fake_neg_score = ops.stop_gradient(fake_neg_score)  
        loss_D = criterion_gan(fake_neg_score, Tensor(fake_lbl))

        fake_pos_score = netDP(fake_pos)  
        fake_pos_score = ops.stop_gradient(fake_pos_score)  
        loss_D += criterion_gan(fake_pos_score, Tensor(fake_lbl))

        real_neg_score = netDN(real_neg_v)  
        loss_D += criterion_gan(real_neg_score, Tensor(real_lbl))

        real_pos_score = netDP(real_pos_v) 
        loss_D += criterion_gan(real_pos_score, Tensor(real_lbl))

        loss_D = loss_D*0.25
        def forward_D(loss_D):
            return loss_D
        grad_D=mindspore.value_and_grad(forward_D,grad_position=None,weights=list(opt_D.get_parameters()))
        loss_D, grads_D = grad_D(loss_D)
        opt_D(grads_D)

        if batch % print_every == 0 and batch > 1:
            print('Epoch #%d' % (epoch+1))
            print('Batch #%d' % batch)

            print('Loss D: %0.3f' % loss_D + '\t' +
                  'Loss G: %0.3f' % loss_G)
            print('Loss P2N G real: %0.3f' % loss_P2N_gan + '\t' +
                  'Loss N2P G fake: %0.3f' % loss_N2P_gan)

            print('-'*50)
            sys.stdout.flush()
            def save_image(tensor, filename):
                tensor = tensor.asnumpy()
                pil_image = tensor.transpose(1, 2, 0)  
                pil_image = Image.fromarray((pil_image * 255).astype('uint8'))  
                pil_image.save(filename)
            save_image(ops.cat([
                real_neg[0]*0.5+0.5,
                mask_pos[0],
                fake_pos[0]*0.5+0.5], 2),
                'progress_pos.png')
            save_image(ops.cat([
                real_pos[0]*0.5+0.5,
                mask_neg[0],
                fake_neg[0]*0.5+0.5], 2),
                'progress_neg.png')
            mindspore.save_checkpoint(netN2P, odir+'/netN2P.ckpt')
            mindspore.save_checkpoint(netN2P, odir+'/netP2N.ckpt')
            mindspore.save_checkpoint(netN2P, odir+'/netDN.ckpt')
            mindspore.save_checkpoint(netN2P, odir+'/netDP.ckpt')
        batch += 1