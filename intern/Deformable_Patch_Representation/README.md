# Deformable-Patch-Representation (MindSpore)

This code is a mindspore implementation of Deformable-Patch-Representation which is avaliable at https://github.com/Omenzychen/Deformable_Patch_Representation ,introduced in the paper https://paperswithcode.com/paper/shape-matters-deformable-patch-attack by Zhaoyu Chen, Bo Li, Shuang Wu, Jianghe Xu, Shouhong Ding, Wenqiang Zhang. 

## Requirements

- Python 3.9
- MindSpore 2.3
- Numpy
- cv2

### Running on Huawei Cloud ModelArts

This implementation can be directly run on Huawei Cloud’s ModelArts platform using the *Guizhou 1* node. For this, select the environment image:

```
mindspore_2.3.0-cann_8.0.rc1-py_3.9-euler_2.10.7-aarch64-snt9b
```

## Usage

To test the loss function with similar values to the test in the original repo, run the following file:

```
python deformable_patch_representation.py
```

