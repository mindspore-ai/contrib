import mindspore as ms
from mindspore import value_and_grad,ops,nn,Tensor,Parameter
#from torch.autograd import Variable
#from torchvision import models
import mindcv as cv
import cv2
import sys
import numpy as np

#Hyper parameters. 
#TBD: Use argparse
tv_beta = 3
lr = 0.1
max_iterations = 500
l1_coeff = 0.01
tv_coeff = 0.2

def tv_norm(inputx, tv_beta):
    img = inputx[0, 0, :]
    row_grad = ops.mean(ops.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = ops.mean(ops.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    preprocessed_img_tensor = Tensor(preprocessed_img)

    preprocessed_img_tensor.unsqueeze(0)
    v = Parameter(preprocessed_img_tensor, requires_grad = False)
    return v

def save(mask, img, blurred):
    mask = np.array(mask.tolist())[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = 1.0*heatmap + np.float32(img)/255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)	

    cv2.imwrite("perturbated.png", np.uint8(255*perturbated))
    cv2.imwrite("heatmap.png", np.uint8(255*heatmap))
    cv2.imwrite("mask.png", np.uint8(255*mask))
    cv2.imwrite("cam.png", np.uint8(255*cam))

def numpy_to_mindspore(img, requires_grad = True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = Tensor(output)
    #if use_cuda:
         #output = output.cuda()

    output=output.unsqueeze(0)
    v = Parameter(output, requires_grad = True)
    return v

def load_model():
    #model = models.vgg19(pretrained=True)
    model=cv.create_model('vgg19',pretrained=True)
    #model.eval()
    #if use_cuda:
         #model.cuda()
    #for p in model.features.parameters():
        #p.requires_grad = False
    #for p in model.classifier.parameters():
        #p.requires_grad = False

    return model

def lossfn(mask,tv_beta,category):
    x=l1_coeff*ops.mean(ops.abs(1 - mask)) + \
				tv_coeff*tv_norm(mask, tv_beta) + outputs[0, category]
    return x,category
def grad_fn(optimizer,mask,tv_beta,category):
#     fx=value_and_grad(lossfn,None,weights=optimizer.trainable_params(), has_aux=True)
    fx=value_and_grad(lossfn,None,weights=[mask], has_aux=True)
    (value,_),grad=fx(mask,tv_beta,category)
    #optimizer(grad)
    return value,grad

if __name__ == '__main__':
    

    model = load_model()
    sys.argv[1]="perturbated_dog.png"
    original_img = cv2.imread(sys.argv[1], 1)
    original_img = cv2.resize(original_img, (224, 224))
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(original_img, 11))/255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    mask_init = np.ones((28, 28), dtype = np.float32)

    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img2)
    mask = numpy_to_mindspore(mask_init)
    #print(type(mask))

    #if use_cuda:
        #upsample = nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
    #else:
    #upsample = nn.UpsamplingBilinear2d(size=(224, 224))
    
    optimizer = nn.Adam([mask], learning_rate=lr)
    #print(img.dim)
    #print(model(img))
    #img.unsqueeze(0)
    x=[1]
    y=list(img.shape)
    for i in range(3):
        x.append(y[i])
    shape=tuple(x)
    #print(shape)
    #img.reshape(shape)
    #print(img.dim)
    imgx=img.reshape(shape)
    #print(imgx.shape)
    target = nn.Softmax()(model(imgx))
    category = np.argmax(np.array(target.tolist()))
    print("Category with highest probability", category)
    print("Optimizing.. ")

    for i in range(max_iterations):
        upsampled_mask = ops.interpolate(mask,size=(224,224),mode='bilinear')
        #print(type(upsampled_mask))
        upsampled_mask=Tensor(upsampled_mask)
        #print(type(upsampled_mask))
        #print(upsampled_mask.size)
        #upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image, 
        # so the mask is duplicated to have 3 channel,
        y=Tensor(np.ones((1, 3, upsampled_mask.shape[2], \
                upsampled_mask.shape[3])),dtype=ms.float32)
        upsampled_mask = \
            upsampled_mask.expand_as(y)

        # Use the mask to perturbated the input image.
        perturbated_input = ops.mul(img,upsampled_mask) + \
        ops.mul(blurred_img,1-upsampled_mask)

        noise = np.zeros((224, 224, 3), dtype = np.float32)
        cv2.randn(noise, 0, 0.2)
        noise = numpy_to_mindspore(noise)
        perturbated_input = perturbated_input + noise

        outputs = nn.Softmax()(model(perturbated_input))
#         loss = l1_coeff*ops.mean(ops.abs(1 - mask)) + \
#                 tv_coeff*tv_norm(mask, tv_beta) + outputs[0, category]
        value,grad=grad_fn(optimizer,mask,tv_beta,category)
#         optimizer.zero_grad()#用原来求导的方法
#         loss.backward()
#         optimizer.step()
#         print(grad)
        optimizer(grad)
        # Optional: clamping seems to give better results
        ops.clamp(mask,0, 1)
#         print(f"Epoch:{i}/{max_iterations}------{value}")

    #upsampled_mask = upsample(mask)
    upsampled_mask = ops.interpolate(mask,size=(224,224),mode='bilinear')
    save(upsampled_mask, original_img, blurred_img_numpy)