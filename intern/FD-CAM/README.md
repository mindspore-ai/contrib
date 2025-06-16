# FD-CAM - MindSpore

This code is a mindspore implementation of FD-CAM 

## Requirements

### **Main dependencies:**

```bash
pip install numpy opencv-python Pillow matplotlib tqdm mindspore torch torchvision pytorch-grad-cam scikit-image pandas scipy
```

## Usage

### dataset

We use ILSVRC2015 val set and VOC2007 val set as dataset.

### model

We get pretrained model VGG16 as the model to be explained from PyTorch model zoo.

```
model = torchvision.models.vgg16(pretrained=True).eval()
```

And we also finetune VGG16 in Pascal VOC dataset.

> What's more, because model.vgg16 in MindSpore is different, There is an extra ModelAdapter designed in mindspore implementation.

## Project Structure
- main_ms.py
    - fdcam_mindspore.py

## Reference
[paper link] https://arxiv.org/abs/2206.08792

[github link] https://github.com/crishhh1998/fd-cam