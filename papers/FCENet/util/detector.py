import numpy as np
import cv2
from util.config import config as cfg
from util.misc import fill_hole, inverse_fourier_transform, generate_e_matrix, poly_nms
from mindspore.nn import Softmax
from mindspore.numpy import transpose


class TextDetector(object):

    def __init__(self, model):
        self.model = model
        self.tr_thresh = cfg.tr_thresh
        self.nms_thresh = cfg.nms_thresh
        self.k = cfg.k
        self.n = 50
        self.e_matrix = generate_e_matrix(n=self.n, k=self.k)
        self.softmax = Softmax(axis=0)

    def detect(self, image):
        # get model output
        cls_predict, reg_predict = self.model.construct_test(image)
        image = image[0].asnumpy()
        for i in range(3):
            tr_pred = self.softmax(cls_predict[i][0, 0:2]).asnumpy()
            tcl_pred = self.softmax(cls_predict[i][0, 2:]).asnumpy()

            cls_predict[i] = tr_pred[1] * tcl_pred[1]
            reg_predict[i] = transpose(reg_predict[i][0], (1, 2, 0)).asnumpy()

        # find text contours
        contours = self.detect_contours(cls_predict, reg_predict, self.k)  # (n_tcl, 3)

        output = {
            'image': image,
            'tr': cls_predict,
            'c': reg_predict
        }
        return contours, output

    def detect_contours(self, cls_predict, reg_predict, k):
        contours = list()
        scores = list()

        for i in range(3):
            scale = 8 * np.power(2, i)

            tr_pred = cls_predict[i]
            x_pred = reg_predict[i][:, :, 0:2 * k + 1]
            y_pred = reg_predict[i][:, :, 2 * k + 1:]

            # thresholding
            tr_pred_mask = tr_pred > self.tr_thresh

            # find disjoint regions
            tr_mask = fill_hole(tr_pred_mask)
            tr_contours, _ = cv2.findContours(tr_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            mask = np.zeros_like(tr_mask)
            for cont in tr_contours:
                deal_map = mask.copy()
                cv2.drawContours(deal_map, [cont], -1, 1, -1)

                text_map = tr_pred * deal_map

                polygons, score = self.contour_transfor_inv(x_pred, y_pred, text_map, scale)

                # nms
                polygons, score = poly_nms(polygons, score, self.nms_thresh)
                contours = contours + polygons
                scores = scores + score

        contours, score = poly_nms(contours, scores, self.nms_thresh)

        return contours

    def contour_transfor_inv(self, x_pred, y_pred, score_map, scale):
        mask = score_map > 0

        xy_text = np.argwhere(mask)
        dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

        x = x_pred[mask]
        y = y_pred[mask]

        c = (x + y * 1j) * scale
        c[:, self.k] = c[:, self.k] + dxy * scale

        polygons = inverse_fourier_transform(c, e_matrix=self.e_matrix)
        score = score_map[mask]
        return polygons, score
