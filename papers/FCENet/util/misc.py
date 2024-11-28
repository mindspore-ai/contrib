import numpy as np
import errno
import os
import cv2
import math
import Polygon as plg
from util.config import config as cfg


def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


def rescale_result(image, bbox_contours, H, W):
    ori_H, ori_W = image.shape[:2]
    image = cv2.resize(image, (W, H))
    contours = list()
    # for (cont, lines) in bbox_contours:
    for cont in bbox_contours:
        cont[:, 0] = (cont[:, 0] * W / ori_W).astype(int)
        cont[:, 1] = (cont[:, 1] * H / ori_H).astype(int)
        contours.append(cont)
    return image, contours


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return (~canvas | input_mask.astype(np.uint8))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))


def vector_sin(v):
    assert len(v) == 2
    # sin = y / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-5
    return v[1] / l


def vector_cos(v):
    assert len(v) == 2
    # cos = x / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-5
    return v[0] / l

def add1(pts, mid_list, dist_list):
    for i in range(len(pts)):
        for j in range(len(pts)):
            s1, e1, mid1 = mid_list[i]
            s2, e2, mid2 = mid_list[j]
            dist = norm2(mid1 - mid2)
            dist_list.append((s1, e1, s2, e2, dist))
    return dist_list

def find_bottom(pts):
    if len(pts) > 4:
        # length = len(pts)
        # bottoms = [(length - 1, 0), (int(length/2 - 1), int(length/2))]

        e = np.concatenate([pts, pts[:3]])
        candidate = []
        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i + 2] - e[i + 1]
            if cos(v_prev, v_next) < -0.875:
                candidate.append((i % len(pts), (i + 1) % len(pts), norm2(e[i] - e[i + 1])))


        if (len(candidate) != 2) * (candidate[0][0] == candidate[1][1]) * (candidate[0][1] == candidate[1][0]):
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            dist_list = []
            if len(candidate) > 2:

                bottom_idx = np.argsort([angle for s1, s2, angle in candidate])[0:2]
                bottoms = [candidate[bottom_idx[0]][:2], candidate[bottom_idx[1]][0:2]]
                long_edge1, long_edge2 = find_long_edges(pts, bottoms)
                edge_length1 = [norm2(pts[e1] - pts[e2]) for e1, e2 in long_edge1]
                edge_length2 = [norm2(pts[e1] - pts[e2]) for e1, e2 in long_edge2]
                l1 = sum(edge_length1)
                l2 = sum(edge_length2)
                len1 = len(edge_length1)
                len2 = len(edge_length2)

                if (l1 > 2*l2) * (l2 > 2*l1) * (len1 == 0) * (len2 == 0):
                    for i in range(len(pts)):
                        mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                        mid_list.append((i, (i + 1) % len(pts), mid_point))

                    dist_list = add1(pts, mid_list, dist_list)

                    bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-1]
                    bottoms = [dist_list[bottom_idx][:2], dist_list[bottom_idx][2:4]]
            else:
                mid_list = []
                for i in range(len(pts)):
                    mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                    mid_list.append((i, (i + 1) % len(pts), mid_point))

                dist_list = []
                dist_list = add1(pts, mid_list, dist_list)
                bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
                bottoms = [dist_list[bottom_idx[0]][:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]
    else:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if 2*d1 < d2 else [(1, 2), (3, 0)]
        # bottoms = [(0, 1), (2, 3)] if 2 * d1 < d2 and d1 > 32 else [(1, 2), (3, 0)]
    assert len(bottoms) == 2, 'fewer than 2 bottoms'

    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]

    if (b1_start == b2_end) * (b1_end == b2_start):
        b2_start = (b1_start + len(pts) // 2) % len(pts)
        b2_end = (b2_start + 1) % len(pts)

    bottoms[0] = [b1_start, b1_end]
    bottoms[1] = [b2_start, b2_end]

    return bottoms


def split_long_edges(points, bottoms):
    """
    Find two long edge sequence of and polygon
    """
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)

    i = b1_end + 1
    long_edge_1 = []
    while i % n_pts != b2_end:
        long_edge_1.append((i - 1, i))
        i = (i + 1) % n_pts

    i = b2_end + 1
    long_edge_2 = []
    while i % n_pts != b1_end:
        long_edge_2.append((i - 1, i))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def find_long_edges(points, bottoms):
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)
    i = (b1_end + 1) % n_pts
    long_edge_1 = []

    while i % n_pts != b2_end:
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_1.append((start, end))
        i = (i + 1) % n_pts

    i = (b2_end + 1) % n_pts
    long_edge_2 = []
    while i % n_pts != b1_end:
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_2.append((start, end))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def split_edge_seqence(points, long_edge, n_parts):

    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while cur_end > point_cumsum[cur_node + 1]:
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


def split_edge_seqence_by_step(points, long_edge1, long_edge2, step=16.0):

    edge_length1 = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge1]
    edge_length2 = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge2]
    # 取长边 计算bbox个数
    total_length = (sum(edge_length1)+sum(edge_length2))/2
    n_parts = math.ceil(float(total_length) / step)

    inner1 = split_edge_seqence(points, long_edge1, n_parts=n_parts)
    inner2 = split_edge_seqence(points, long_edge2, n_parts=n_parts)

    return inner1, inner2


def sample(polygon, n=400):
    length = []

    for i in range(len(polygon)):
        p1 = polygon[i]
        if i == len(polygon)-1:
            p2 = polygon[0]
        else:
            p2 = polygon[i + 1]
        length.append(((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5)

    total_lenght = sum(length)

    num_of_each_line = (length / total_lenght) * n
    num_of_each_line = np.array(num_of_each_line, dtype='int32')
    new_polygon = []

    for i in range(len(polygon)):
        num = num_of_each_line[i]
        p1 = polygon[i]
        if i == len(polygon)-1:
            p2 = polygon[0]
        else:
            p2 = polygon[i + 1]

        if num == 0:
            continue

        dxdy = (p2 - p1) / num
        for j in range(num):
            point = p1 + dxdy * j
            new_polygon.append(point)

    return np.array(new_polygon)


def fourier_transform(polygon, k):
    points = polygon[:, 0] + polygon[:, 1] * 1j
    n = len(points)
    t = np.multiply([i / n for i in range(n)], -2 * np.pi * 1j)

    e = complex(np.e)
    c = np.zeros((2 * k + 1,), dtype='complex')

    for i in range(-k, k + 1):
        c[i + k] = np.sum(points * np.power(e, i * t)) / n

    return c


def inverse_fourier_transform(c, e_matrix):
    assert type(c) == np.ndarray
    assert c.shape[1] == e_matrix.shape[0]

    n = e_matrix.shape[1]
    polygons = np.zeros((c.shape[0], n, 2))

    points = np.matmul(c, e_matrix)
    p_x = np.real(points)
    p_y = np.imag(points)
    polygons[:, :, 0] = p_x
    polygons[:, :, 1] = p_y
    return polygons.astype('int32')


def generate_e_matrix(n, k):
    e = complex(np.e)
    e_matrix = np.zeros([2 * k + 1, n], dtype='complex')

    temp = np.zeros([n], dtype='complex')
    for i in range(n):
        temp[i] = 2 * np.pi * 1j / n * i

    for i in range(2 * k + 1):
        e_matrix[i, :] = temp * (i - k)

    return np.power(e, e_matrix)


def clockwise(c, k):
    if np.abs(c[k + 1]) > np.abs(c[k - 1]):
        return c
    elif np.abs(c[k + 1]) < np.abs(c[k - 1]):
        return c[::-1]
    else:
        if np.abs(c[k + 2])>np.abs(c[k - 2]):
            return c
        else:
            return c[::-1]

def find_start_point(polygon):
    temp_polygon = polygon - polygon.mean(axis=0)
    x = np.abs(temp_polygon[:, 0])
    y = temp_polygon[:, 1]
    index_x = np.argsort(x)
    index_y = np.argmin(y[index_x[:8]])
    index = index_x[index_y]
    new_polygon = np.concatenate([polygon[index:], polygon[:index]])
    return new_polygon

def poly_nms(polygons, score, threshold):
    keep_poly = []
    keep_score = []
    index = np.argsort(score)[::-1]

    while len(index) > 0:
        keep_poly.append(polygons[index[0]])
        keep_score.append(score[index[0]])

        A = plg.Polygon(polygons[index[0]])

        index = np.delete(index, 0)

        iou_list = np.zeros((len(index),))
        for i in range(len(index)):
            B = plg.Polygon(polygons[index[i]])

            iou_list[i] = poly_iou(A, B)
        remove_index = np.where(iou_list > threshold)
        index = np.delete(index, remove_index)

    return keep_poly, keep_score


def poly_intersection(poly_det, poly_gt):
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    poly_inter = poly_det & poly_gt
    if len(poly_inter) == 0:
        return 0, poly_inter
    return poly_inter.area(), poly_inter


def poly_union(poly_det, poly_gt):
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    area_det = poly_det.area()
    area_gt = poly_gt.area()
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    return area_det + area_gt - area_inters


def poly_iou(poly_det, poly_gt):
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)
    area_inters, _ = poly_intersection(poly_det, poly_gt)

    return area_inters / poly_union(poly_det, poly_gt)
