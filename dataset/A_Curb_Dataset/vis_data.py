""" ============================================================================
 Copyright 2021 Huawei Technologies Co., Ltd

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 The Curb Dataset's license is different, and you can find it in ReadMe.md.

============================================================================"""

import argparse
import os
import numpy as np
import cv2

color_map = {  # bgr
    0: [0, 0, 0],
    1: [255, 255, 255],
    2: [245, 150, 150],
    3: [245, 230, 100],
    4: [250, 80, 150],
    5: [150, 60, 30],
    6: [255, 150, 0],
    7: [180, 30, 80],
    8: [255, 0, 150],
    9: [30, 150, 255],
    10: [200, 150, 255],
    11: [90, 150, 150],
    12: [255, 0, 255],
    13: [255, 150, 255],
    14: [75, 0, 75],
    15: [75, 0, 175],
    16: [0, 200, 255],
    17: [50, 120, 255],
    18: [0, 150, 255],
    19: [170, 255, 150],
    20: [0, 175, 0],
    21: [0, 60, 135],
    22: [80, 240, 150],
    23: [150, 240, 255],
    24: [0, 0, 255],
    25: [255, 255, 50],
    26: [245, 150, 100],
    27: [255, 0, 0],
    28: [200, 40, 255],
    29: [30, 30, 255],
    30: [90, 30, 150],
    31: [250, 80, 100],
    32: [30, 30, 30],
    33: [90, 120, 180],
}


def get_args():
    """
    @ params: None
    @ return: all configure params
    @ description : parser all params
    """
    parser = argparse.ArgumentParser("./vis_data.py", description='vis curb annotations with LiDAR point cloud')
    parser.add_argument('-c', '--curb_path', type=str, required=True, help='Data-set path of curb annotations')
    parser.add_argument('-k', '--kernel_size', type=int, required=False, default=7, help='kernel size in curb dilating')
    parser.add_argument('-d', '--semantic_kitti_path', type=str, required=False, help='vis point cloud')
    parser.add_argument('-p', '--pass_frames', type=int, required=False, default=0, help='start frame in sequence')
    return parser.parse_args()


def filter_point(pc_in):
    """
    @ params: input pointcloud (N*3)
    @ return: filtered pointcloud (N*3)
    @ description : filter point by xyz
    """
    idx = (pc_in[:, 0] >= roi_range[0]) & (pc_in[:, 0] <= roi_range[3]) & \
          (pc_in[:, 1] >= roi_range[1]) & (pc_in[:, 1] <= roi_range[4]) & \
          (pc_in[:, 2] >= roi_range[2]) & (pc_in[:, 2] <= roi_range[5])
    pc_out = pc_in[idx, :]
    return pc_out


def vis_image(grid_curb_in, image_size_in, kernel_in=7):
    """
    @ params: grid_curb_in: the input grid array
    @ params: image_size_in: the input image size
    @ params: kernel_in: the dilating kernel size, default=7
    @ return: the binary image & the instance image
    @ description : vis_image
    """
    grid_curb_in[:, 0] = np.clip(grid_curb_in[:, 0], 0, image_size_in[0] - 1)
    grid_curb_in[:, 1] = np.clip(grid_curb_in[:, 1], 0, image_size_in[1] - 1)
    curb_ids = np.unique(grid_curb_in[:, 2])
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_in, kernel_in))

    img_binary_out = np.zeros((image_size_in[0], image_size_in[1], 1), dtype=np.uint8)
    img_instance_out = np.zeros((image_size_in[0], image_size_in[1], 3), dtype=np.uint8)

    for curb_id in curb_ids:
        idx = grid_curb_in[:, 2] == curb_id
        curb_instance = grid_curb_in[idx, 0:2]
        img_dilate = np.zeros((image_size_in[0], image_size_in[1], 1), dtype=np.uint8)
        img_dilate[curb_instance[:, 0], curb_instance[:, 1]] = 255
        img_dilate_dst = cv2.dilate(img_dilate, ker)
        img_dilate_index = np.array(np.where(img_dilate_dst), dtype=int)
        img_binary_out[img_dilate_index[0], img_dilate_index[1]] = 255
        color = color_map_array[(curb_id + 1) % color_map_array.shape[0]]
        img_instance_out[img_dilate_index[0], img_dilate_index[1]] = color
    return img_binary_out, img_instance_out


if __name__ == '__main__':
    args = get_args()
    roi_range = [0.0, -19.2, -3, 51.2, 19.2, 0]
    grid_res = 0.1
    image_size = [int((roi_range[3]-roi_range[0]) / grid_res), int((roi_range[4]-roi_range[1]) / grid_res)]
    color_map_array = np.array(color_map.values())

    print('---------------------------------------------------------------------------------------------------------')
    print('curb_dataset : %s' % args.curb_path)
    print('SemanticKITTI_dataset : %s' % args.semantic_kitti_path)
    print('kernel_size : %d' % args.kernel_size)
    print('ROI range : %s' % roi_range)
    print('grid resolution : %.2f' % grid_res)
    print('image size : %s' % image_size)
    print('Tab space to go to the next frame !')
    print('---------------------------------------------------------------------------------------------------------')

    curb_data_path = args.curb_path
    pc_path = args.semantic_kitti_path
    kernel = args.kernel_size
    curb_files = [f for f in sorted(os.listdir(os.path.join(curb_data_path, "curb"))) if f.endswith(".txt")]
    for i, f in enumerate(curb_files):
        if i < args.pass_frames:
            continue
        curb_file = os.path.join(curb_data_path, "curb", f)
        curb = np.loadtxt(curb_file)
        grid_curb = curb.copy()
        grid_curb[:, 0] = np.floor((curb[:, 0] - roi_range[0]) / grid_res)
        grid_curb[:, 1] = np.floor((grid_curb[:, 1] - roi_range[1]) / grid_res)
        grid_curb = grid_curb.astype(int)
        binary_img, instance_img = vis_image(grid_curb_in=grid_curb, image_size_in=image_size, kernel_in=kernel)
        # vis with raw point cloud if you like
        if pc_path is not None:
            pc_file = os.path.join(pc_path, "velodyne", os.path.splitext(f)[0] + ".bin")
            scan = np.fromfile(pc_file, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            scan_in_roi = filter_point(scan)
            scan_grid = np.zeros((scan_in_roi.shape[0], 2))
            scan_grid[:, 0] = np.floor((scan_in_roi[:, 0] - roi_range[0]) / grid_res)  # ROW
            scan_grid[:, 1] = np.floor((scan_in_roi[:, 1] - roi_range[1]) / grid_res)  # COL
            scan_grid[:, 0] = np.clip(scan_grid[:, 0], 0, image_size[0] - 1)
            scan_grid[:, 1] = np.clip(scan_grid[:, 1], 0, image_size[1] - 1)
            scan_grid = scan_grid.astype(int)
            scan_img = instance_img.copy()
            scan_img[scan_grid[:, 0], scan_grid[:, 1]] = [120, 120, 120]
            cv2.imshow('curb&scan', scan_img)

        cv2.imshow('binary_curb', binary_img)
        cv2.imshow('instance_curb', instance_img)
        cv2.waitKey(0)
