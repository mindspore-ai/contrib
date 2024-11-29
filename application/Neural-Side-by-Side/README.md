# NeuralSBS and SBS180K dataset
PyTorch Implementation of Neural Side-By-Side: Predicting Human Preferences for No-Reference Super-Resolution Evaluation
## SBS180K dataset
You can download the dataset (37 GB) from [this Dropbox url](https://www.dropbox.com/s/45tz3m5al9axyc5/NeuralSBS_dataset.zip).

## Pretrained model.
To build a model, use the following code.
```
from model import get_score_model
score_model = get_score_model('inception_v3', pretrained=True)
```
Note that `pretrained=True` to ensure correct normalization coming with the torchvision implementation of Inception.

Checkpoint used for evaluation is available at [this Dropbox url](https://www.dropbox.com/s/gwalk982rombtov/neuralsbs.pth).

It can be loaded as 
```
score_model.load_state_dict(torch.load('neuralsbs.pth')['model_state_dict'])
score_model.eval()
```
## Evaluation
We used [Albumentations](https://github.com/albumentations-team/albumentations) to simultaneously augment both images.
Images have to be converted to the BGR format first, and scaled to the [0, 1] range. Then the score can be computed as follows.
```
from transform import get_test_transform
transform = get_test_transform(normalize=True, resize=299)
# load im1, im2 in the format described above, e.g., with cv2.imread and divide by 255
processed = transform(image=im1, image2=im2)
im1, im2 = processed['image'], processed['image2']
im = torch.stack((im1, im2)).unsqueeze(0)
# input to the model is of shape B x 2 x C x H x W 
with torch.no_grad():
    score = torch.sigmoid(score_model(im)).item()
```
If you used our model or dataset in your research, please consider citing our paper.
```
@InProceedings{Khrulkov_2021_CVPR,
    author    = {Khrulkov, Valentin and Babenko, Artem},
    title     = {Neural Side-by-Side: Predicting Human Preferences for No-Reference Super-Resolution Evaluation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {4988-4997}
}
```




