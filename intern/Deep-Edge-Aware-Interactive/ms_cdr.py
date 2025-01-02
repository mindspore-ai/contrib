import numpy as np

import cv2
import mindspore
from mindspore import ops
from mindspore import Tensor
from skimage import feature
from skimage.segmentation import slic



def cdr(inputs, gt, kernel_size=5, target_region_mask=None):
    """
    inputs: Target Image(Lab) with C * H * W
    gt: GT Image(Lab) with C * H * W
    kernel_size: Area of mesure the CDR
    target_region_mask: Region of interest (Binary Scribble in the paper)  
    return: CDR ratio for each a & b channel
    """
    if type(inputs) == np.ndarray:
        print('inputs is np.ndarray')
        inputs = mindspore.Tensor(inputs)
        gt = mindspore.Tensor(gt)
    assert inputs.dim() == 3
    assert gt.dim() == 3
    inputs = ops.transpose(inputs, (1, 2, 0))
    # print("inputs", inputs)
    gt = ops.transpose(gt, (1, 2, 0))

    pad_size = (kernel_size-1, kernel_size-1, kernel_size-1, kernel_size-1)
    gt_a = ops.unsqueeze(gt[:, :, 1], -1).tile((1, 1, 3))
    gt_b = ops.unsqueeze(gt[:, :, 2], -1).tile((1, 1, 3))
    inputs_a = ops.unsqueeze(inputs[:, :, 1], -1).tile((1, 1, 3))
    inputs_b = ops.unsqueeze(inputs[:, :, 2], -1).tile((1, 1, 3))
    # print('gt_a', gt_a)
    gt_a_double = gt_a.astype("float64")
    gt_b_double = gt_b.astype("float64")
    inputs_a_double = inputs_a.astype("float64")
    inputs_b_double = inputs_b.astype("float64")
    gt_a_slic = mindspore.Tensor(slic(gt_a_double.asnumpy(), n_segments=250,
                                  compactness=10, sigma=1, start_label=1), mindspore.float32)
    gt_b_slic = mindspore.Tensor(slic(gt_b_double.asnumpy(), n_segments=250,
                                  compactness=10, sigma=1, start_label=1), mindspore.float32)
    inputs_a_slic = mindspore.Tensor(slic(inputs_a_double.asnumpy(), n_segments=250,
                                 compactness=10, sigma=1, start_label=1), mindspore.float32)
    inputs_b_slic = mindspore.Tensor(slic(inputs_b_double.asnumpy(), n_segments=250,
                                 compactness=10, sigma=1, start_label=1), mindspore.float32)

    # Add the padding
    gt_a_slic = ops.pad(gt_a_slic, pad_size, "constant", 0)
    gt_b_slic = ops.pad(gt_b_slic, pad_size, "constant", 0)
    inputs_a_slic = ops.pad(inputs_a_slic, pad_size, "constant", 0)
    inputs_b_slic = ops.pad(inputs_b_slic, pad_size, "constant", 0)

    # canny_a_data = feature.canny(gt_a[:, :, 0].asnumpy(), sigma=1.2,
    #                              high_threshold=0.7, low_threshold=0.2, use_quantiles=0.4)
    # print('canny_a_data', canny_a_data)

    canny_a = mindspore.Tensor(feature.canny(gt_a[:, :, 0].asnumpy(), sigma=1.2,
                           high_threshold=0.7, low_threshold=0.2, use_quantiles=0.4), mindspore.float32)
    canny_b = mindspore.Tensor(feature.canny(gt_b[:, :, 0].asnumpy(), sigma=1.2,
                           high_threshold=0.7, low_threshold=0.2, use_quantiles=0.4), mindspore.float32)


    if target_region_mask is None:
        canny_a = ops.pad(canny_a, pad_size, "constant", 0)
        canny_b = ops.pad(canny_b, pad_size, "constant", 0)
    else:
        canny_a = ops.pad(canny_a * target_region_mask, pad_size, "constant", 0)
        canny_b = ops.pad(canny_a * target_region_mask, pad_size, "constant", 0)
    canny_a_coor = ops.nonzero(canny_a)
    canny_b_coor = ops.nonzero(canny_b)
    cdr_a, cdr_b = 1., 1.
    if len(canny_a_coor) != 0:
        cdr_a = 0.
        for num_edge_a, coor in enumerate(range(canny_a_coor.shape[0])):
            h, w = canny_a_coor[coor][-2], canny_a_coor[coor][-1]

            gt_sc_a = gt_a_slic[h - kernel_size:h + kernel_size,
                                w - kernel_size:w + kernel_size] != gt_a_slic[h, w]

            inputs_sc_a = inputs_a_slic[h - kernel_size:h + kernel_size,
                                        w - kernel_size:w + kernel_size] == inputs_a_slic[h, w]
            gt_sc_a = gt_sc_a.astype(mindspore.float32)
            inputs_sc_a = inputs_sc_a.astype(mindspore.float32)
            # print('gt_sc_a.sum()', gt_sc_a.sum())
            # print('(gt_sc_a * inputs_sc_a).sum()', (gt_sc_a * inputs_sc_a))
            if gt_sc_a.sum() != 0:
                cdr_a += 1 - float((gt_sc_a * inputs_sc_a).sum()) / float(gt_sc_a.sum())
            else:
                cdr_a += 1
        cdr_a /= (num_edge_a + 1)

    if len(canny_b_coor) != 0:
        cdr_b = 0.
        for num_edge_b, coor in enumerate(range(canny_b_coor.shape[0])):
            h, w = canny_b_coor[coor][-2], canny_b_coor[coor][-1]

            gt_sc_b = gt_b_slic[h - kernel_size:h + kernel_size,
                                w - kernel_size:w + kernel_size] != gt_b_slic[h, w]

            inputs_sc_b = inputs_b_slic[h - kernel_size:h + kernel_size,
                                        w - kernel_size:w + kernel_size] == inputs_b_slic[h, w]

            gt_sc_b = gt_sc_b.astype(mindspore.float32)
            inputs_sc_b = inputs_sc_b.astype(mindspore.float32)
            if gt_sc_b.sum() != 0:
                cdr_b += 1 - float((gt_sc_b * inputs_sc_b).sum()) / float(gt_sc_b.sum())
            else:
                cdr_b += 1
        cdr_b /= (num_edge_b + 1)

    return cdr_a, cdr_b    

if __name__ == "__main__":
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    example_gt = cv2.cvtColor(cv2.imread('./example/gt.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
    org_inputs = cv2.cvtColor(cv2.imread('./example/org_inputs.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
    enh_inputs = cv2.cvtColor(cv2.imread('./example/enh_inputs.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)

    cdr_a, cdr_b = cdr(inputs = org_inputs, gt = example_gt)
    print('CDR score of the original result of colorization model')
    print(f'Org_CDR_a:{cdr_a:.3f} Org_CDR_b:{cdr_b:.3f}')
    cdr_a, cdr_b = cdr(inputs = enh_inputs, gt = example_gt)
    print('CDR score of enhancing result of our model')
    print(f'Enh_CDR_a:{cdr_a:.3f} Enh_CDR_b:{cdr_b:.3f}')
