# A Curb Dataset with LiDAR Data for Autonomous Driving

This is a dataset with curb annotations by using 3D LiDAR data and we build this dataset based on the SemanticKITTI dataset.

## File format

The file format follows with that of the SemanticKITTI dataset, but in a 2D coordinate frame. \
For example, the data in './dataset/sequences/00/curb/000000.txt' represents the curb points in a 2D bird's eye view corresponding to the LiDAR point cloud of frame 000000, sequence 00. The curb data of this frame has a dimension of N*3: N means the number of curb point clouds and 3 means (x, y, curb_instance_id). \
We labeled the LiDAR data in sequences 00-10, totaling 23201 frames.

## Download

The dataset is available at [mindspore](https://download.mindspore.cn/dataset/ACurbDataSet.rar).

## How to use

The dataset is easy to be downloaded and parsed with a provided Python script to visualize the curb data.

```python
python vis_data.py -c ${curb_dataset_folder_path}/dataset/sequences/00
```

Moreover, if you want to get the raw point clouds data, you could go to the website of SemanticKITTI and visualize the point clouds with the curb data.

```python
python vis_data.py -c ${curb_dataset_folder_path}/dataset/sequences/00 -d ${SemanticKITTI_folder_path}/dataset/sequences/00
```

SemanticKITTI : http://www.semantic-kitti.org

## License

Our dataset is based on the SemanticKITTI dataset, therefore we distribute the dataset under **Creative Commons Attribution-NonCommercial-ShareAlike** license. You are free to share and adapt the data, but have to give appropriate credit and may not use the work for commercial purposes.

## Citations

>

```python

@inproceedings{behley2019iccv,
author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
title = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
booktitle = {Proc. of the IEEE/CVF International Conf.~on Computer Vision (ICCV)},
year = {2019}
}
```

>

```python

@inproceedings{geiger2012cvpr,
author = {A. Geiger and P. Lenz and R. Urtasun},
title = {{Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite}},
booktitle = {Proc.~of the IEEE Conf.~on Computer Vision and Pattern Recognition (CVPR)},
pages = {3354--3361},
year = {2012}
}
```