import copy
import cv2
# import torch
from mindspore import Tensor
import numpy as np
from PIL import Image
from util.config import config as cfg
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_sin, split_edge_seqence_by_step, sample, fourier_transform, \
    clockwise, find_start_point


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        remove_points = []
        self.points = np.array(points)
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def equal_width_bbox_cover(self, step=16.0):

        inner_points1, inner_points2 = split_edge_seqence_by_step(self.points, self.e1, self.e2, step=step)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center

        return inner_points1, inner_points2, center_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training

    @staticmethod
    def fill_polygon(mask, pts, value):
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))

    def make_text_region(self, img, polygon, tr_mask, train_mask, x_map, y_map, k, scale=1/2):
        [h, w] = img.shape[:2]
        h = int(h * scale)
        w = int(w * scale)

        deal_mask = np.zeros((h, w), np.uint8)
        points = (polygon.points * scale).astype(np.int32)
        cv2.fillPoly(tr_mask, [points], color=(1,))
        cv2.fillPoly(deal_mask, [points], color=(1,))

        if polygon.text == '#':
            cv2.fillPoly(train_mask, [points], color=(0,))

        pts = sample(polygon.points * scale)
        pts = find_start_point(pts)
        c = fourier_transform(pts, k)
        c = clockwise(c, k)

        vector_x = np.real(c)
        vector_y = np.imag(c)

        for i in range(-k, k+1):
            if i != 0:
                x_map[:, :, i + k] = deal_mask * vector_x[i + k] + (1 - deal_mask) * x_map[:, :, i + k]
                y_map[:, :, i + k] = deal_mask * vector_y[i + k] + (1 - deal_mask) * y_map[:, :, i + k]
            else:
                for y, x in np.argwhere(deal_mask > 0.5):
                    x_map[y, x, k] = vector_x[k] - x
                    y_map[y, x, k] = vector_y[k] - y

    def make_text_center_line(self, sideline1, sideline2,
                              center_line, tcl_msk1, expand=0.3, shrink=1):
        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)
        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top_line = sideline2
            bot_line = sideline1
        else:
            top_line = sideline1
            bot_line = sideline2

        if len(center_line) < 5:
            shrink = 0

        for i in range(shrink, len(center_line) - 1 - shrink):
            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = top_line[i]
            top2 = top_line[i + 1]
            bottom1 = bot_line[i]
            bottom2 = bot_line[i + 1]

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            ploy1 = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_msk1, ploy1, value=1)

    def get_training_data(self, image, polygons, k, image_id, image_path):

        H, W, _ = image.shape
        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        h, w, _ = image.shape

        tr_mask_3 = np.zeros((int(h/8), int(w/8), 1), np.uint8)
        train_mask_3 = np.ones((int(h/8), int(w/8), 1), np.uint8)
        tcl_mask_3 = np.zeros((int(h / 8), int(w / 8), 1), np.uint8)
        x_map_3 = np.zeros((int(h/8), int(w/8), 2 * k + 1), np.float32)
        y_map_3 = np.zeros((int(h/8), int(w/8), 2 * k + 1), np.float32)

        tr_mask_4 = np.zeros((int(h/16), int(w/16), 1), np.uint8)
        train_mask_4 = np.ones((int(h/16), int(w/16), 1), np.uint8)
        tcl_mask_4 = np.zeros((int(h/16), int(w/16), 1), np.uint8)
        x_map_4 = np.zeros((int(h/16), int(w/16), 2 * k + 1), np.float32)
        y_map_4 = np.zeros((int(h/16), int(w/16), 2 * k + 1), np.float32)

        tr_mask_5 = np.zeros((int(h/32), int(w/32), 1), np.uint8)
        train_mask_5 = np.ones((int(h/32), int(w/32), 1), np.uint8)
        tcl_mask_5 = np.zeros((int(h/32), int(w/32), 1), np.uint8)
        x_map_5 = np.zeros((int(h/32), int(w/32), 2 * k + 1), np.float32)
        y_map_5 = np.zeros((int(h/32), int(w/32), 2 * k + 1), np.float32)

        if polygons is not None:
            for polygon in polygons:
                x_max = polygon.points[:, 0].max()
                x_min = polygon.points[:, 0].min()
                y_max = polygon.points[:, 1].max()
                y_min = polygon.points[:, 1].min()
                dx = x_max - x_min
                dy = y_max - y_min
                criterion = max(dx, dy) / (h + 1e-5)

                polygon.find_bottom_and_sideline()
                sideline1, sideline2, center_points = polygon.equal_width_bbox_cover(step=4.0)

                if criterion < 0.4:
                    self.make_text_region(image, polygon, tr_mask_3, train_mask_3, x_map_3, y_map_3, k, scale=1 / 8)
                    self.make_text_center_line(sideline1/8, sideline2/8, center_points/8, tcl_mask_3)

                if criterion > 0.3 and criterion < 0.7:
                    self.make_text_region(image, polygon, tr_mask_4, train_mask_4, x_map_4, y_map_4, k, scale=1 / 16)
                    self.make_text_center_line(sideline1/16, sideline2/16, center_points/16, tcl_mask_4)

                if criterion > 0.6:
                    self.make_text_region(image, polygon, tr_mask_5, train_mask_5, x_map_5, y_map_5, k, scale=1 / 32)
                    self.make_text_center_line(sideline1/32, sideline2/32, center_points/32, tcl_mask_5)

        # clip value (0, 1)
        tr_mask_3 = np.clip(tr_mask_3, 0, 1)
        train_mask_3 = np.clip(train_mask_3, 0, 1)
        tcl_mask_3 = np.clip(tcl_mask_3, 0, 1)
        tr_mask_4 = np.clip(tr_mask_4, 0, 1)
        train_mask_4 = np.clip(train_mask_4, 0, 1)
        tcl_mask_4 = np.clip(tcl_mask_4, 0, 1)
        tr_mask_5 = np.clip(tr_mask_5, 0, 1)
        train_mask_5 = np.clip(train_mask_5, 0, 1)
        tcl_mask_5 = np.clip(tcl_mask_5, 0, 1)

        label_3 = np.concatenate([tr_mask_3, train_mask_3, x_map_3, y_map_3, tcl_mask_3], axis=2)
        label_4 = np.concatenate([tr_mask_4, train_mask_4, x_map_4, y_map_4, tcl_mask_4], axis=2)
        label_5 = np.concatenate([tr_mask_5, train_mask_5, x_map_5, y_map_5, tcl_mask_5], axis=2)

        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        # image = Tensor.from_numpy(image).astype("float32")
        # label_3 = Tensor.from_numpy(label_3).astype("float32")
        # label_4 = Tensor.from_numpy(label_4).astype("float32")
        # label_5 = Tensor.from_numpy(label_5).astype("float32")

        if not self.is_training:
            points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
            length = np.zeros(cfg.max_annotation, dtype=int)
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    pts = polygon.points
                    points[i, :pts.shape[0]] = polygon.points
                    length[i] = pts.shape[0]

            meta = {
                'image_id': image_id,
                'image_path': image_path,
                'annotation': points,
                'n_annotation': length,
                'Height': H,
                'Width': W
            }
            # meta = [image_id, image_path, points, length, H, W]
            # meta = np.array(meta)

            return image, label_3, label_4, label_5, meta

        return image, label_3, label_4, label_5

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()
