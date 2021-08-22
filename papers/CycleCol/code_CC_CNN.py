# 本模块用于读取数据，生成加载器，相关参数设置如下：

left_path = './data/color'
right_path = './data/mono'
split_ratio = 0.8

import random
import cv2
import mindspore.dataset as ds
import os
import numpy as np
import glob
import mindspore as mds
import mindspore.nn as nn
import mindspore.numpy as mdsnp
mds.context.set_context(device_target='Ascend',mode=mds.context.PYNATIVE_MODE)#  GRAPH_MODE
from mindspore import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn import Loss
import mindspore.ops as ops
from mindspore import Tensor


def load_data_path(left_path, right_path, split_ratio=0.03):
    # load data path
    left_images = os.listdir(left_path)
    right_images = os.listdir(right_path)
    left_images.sort()
    right_images.sort()
    left_images_path = sorted(glob.glob(left_path + "/*.png"))
    right_images_path = sorted(glob.glob(right_path + "/*.png"))
    # print(left_images)
    # print(right_images)
    # print(left_images)

    # left_images_path=[left_path+'/'+img for img in left_images ]
    # right_images_path=[right_path+'/'+img for img in right_images]

    # print(left_images_path[:10])
    # print(right_images_path[:10])

    # split data
    data_length = len(left_images_path)
    temp = [(l, r) for l, r in zip(left_images_path, right_images_path)]
    # print(temp[:10])
    random.shuffle(temp)
    num_traindata = int(split_ratio * data_length)
    train_data_path = temp[0:num_traindata]
    # print(num_traindata)  638
    val_data_path = temp[num_traindata:]

    return train_data_path, val_data_path
def data_generator(data_path, is_train=True):
        left_img = cv2.imread(data[0])
        right_img = cv2.imread(data[1])
        left_img = cv2.resize(left_img, (200, 100), interpolation=cv2.INTER_CUBIC)
        right_img = cv2.resize(right_img, (200, 100), interpolation=cv2.INTER_CUBIC)
        left_img_rotate = _rotateImage_(left_img)
        right_img_rotate = _rotateImage_(right_img)
        img_shape = left_img_rotate.shape
        left_geo_feat = _getGeometryFeat_(img_shape)
        right_geo_feat = _getGeometryFeat_(img_shape)
        left_img_rotate = _centerImage_(left_img_rotate)
        right_img_rotate = _centerImage_(right_img_rotate)
        left_geo_feat = _centerImage_(left_geo_feat)
        right_geo_feat = _centerImage_(right_geo_feat)
        left_input = np.concatenate([left_img_rotate, left_geo_feat], axis=2)
        right_input = np.concatenate([right_img_rotate, right_geo_feat], axis=2)
        left_input = np.moveaxis(left_input, 2, 0)
        right_input = np.moveaxis(right_input, 2, 0)
        if is_train == True:
            VUY_map = np.concatenate((left_img_rotate, right_img_rotate), axis=2)
            VUY_map = np.moveaxis(VUY_map, 2, 0)
            traindata = [left_input, right_input]
            yield traindata, VUY_map
        else:
            yield traindata


class imgDataset():
    def __init__(self, tra):
        super(imgDataset, self).__init__()
        self.tra = tra
    def __getitem__(self, index):

        return self.data_generator([self.tra[index]])

    def __len__(self):
        return len(self.tra)

    def data_generator(self, data_path, is_train=True):
        for data in data_path:
            left_img = cv2.imread(data[0])
            right_img = cv2.imread(data[1])
            left_img = cv2.resize(left_img, (128, 80), interpolation=cv2.INTER_CUBIC)
            right_img = cv2.resize(right_img, (128, 80), interpolation=cv2.INTER_CUBIC)
            left_img_rotate = self._rotateImage_(left_img)
            right_img_rotate = self._rotateImage_(right_img)

            img_shape = left_img_rotate.shape
            left_geo_feat = self._getGeometryFeat_(img_shape)
            right_geo_feat = self._getGeometryFeat_(img_shape)

            left_img_rotate = self._centerImage_(left_img_rotate)
            right_img_rotate = self._centerImage_(right_img_rotate)
            left_geo_feat = self._centerImage_(left_geo_feat)
            right_geo_feat = self._centerImage_(right_geo_feat)
            left_input = np.concatenate([left_img_rotate, left_geo_feat], axis=2)
            right_input = np.concatenate([right_img_rotate, right_geo_feat], axis=2)
            left_input = np.moveaxis(left_input, 2, 0)
            right_input = np.moveaxis(right_input, 2, 0)
            if is_train == True:
                VUY_map = np.concatenate((left_img_rotate, right_img_rotate), axis=2)
                VUY_map = np.moveaxis(VUY_map, 2, 0)
                traindata = [left_input, right_input]
                return traindata, VUY_map
            else:
                return traindata

    def _centerImage_(self, img):
        img = img.astype(np.float32)
        return img

    def _rotateImage_(self, img):
        (h, w) = img.shape[:2]
        center = (w / 2 - 0.5, h / 2 - 0.5)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated

    def _getGeometryFeat_(self, img_shape):
        H = img_shape[0]
        W = img_shape[1]
        feat = np.zeros((H, W, 2))
        for j in range(H):
            for i in range(W):
                feat[j, i, 0] = np.min([j - 0, H - 1 - j]) / (H - 1) * 1.0
                feat[j, i, 1] = np.min([i - 0, W - 1 - i]) / (W - 1) * 1.0
        return feat

class resnetblock(nn.Cell):
    def __init__(self, infea, outfea, stra=1, pad='same', df='NCHW'):
        super(resnetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=infea, out_channels=outfea, kernel_size=3, stride=stra, pad_mode=pad,
                               data_format=df)
        self.bn1 = nn.BatchNorm2d(num_features=outfea, data_format=df)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=outfea, out_channels=outfea, kernel_size=3, stride=stra, pad_mode=pad,
                               data_format=df)
        self.bn2 = nn.BatchNorm2d(num_features=outfea, data_format=df)

    def construct(self, input_img):
        x = self.bn1(self.conv1(input_img))
        x = self.act(x)
        x = self.bn2(self.conv2(x))
        return self.act(input_img) + x


class UniFeature(nn.Cell):
    def __init__(self, infea, outfea, pad='same', df='NCHW'):  # 3    8
        super(UniFeature, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=infea, out_channels=outfea, kernel_size=5, stride=1, pad_mode=pad,
                               data_format=df)
        self.bn1 = nn.BatchNorm2d(num_features=outfea, data_format=df)
        self.conv2 = nn.Conv2d(in_channels=outfea, out_channels=outfea, kernel_size=3, stride=1, pad_mode=pad,
                               data_format=df)
        self.bn2 = nn.BatchNorm2d(num_features=outfea, data_format=df)
        self.conv3 = nn.Conv2d(in_channels=outfea, out_channels=outfea, kernel_size=3, stride=1, pad_mode=pad,
                               data_format=df)
        self.bn3 = nn.BatchNorm2d(num_features=outfea, data_format=df)
        self.conv4 = nn.Conv2d(in_channels=outfea, out_channels=outfea, kernel_size=3, stride=1, pad_mode=pad,
                               data_format=df)
        self.bn4 = nn.BatchNorm2d(num_features=outfea, data_format=df)
        self.act = nn.ReLU()

    def construct(self, input_img):  # (none,none,3)
        x = self.bn1(self.conv1(input_img))
        x = self.act(x)
        x = self.bn2(self.conv2(x))
        x = self.act(x)
        x = self.bn3(self.conv3(x))
        x = self.act(x)
        x = self.bn4(self.conv4(x))
        return self.act(x)


class processgeofeature(nn.Cell):
    def __init__(self, infea, outfea, pad='same', df='NCHW'):  # 2  8  2
        super(processgeofeature, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=infea, out_channels=outfea, kernel_size=5, stride=1, pad_mode=pad,
                               data_format=df)
        self.bn1 = nn.BatchNorm2d(num_features=outfea, data_format=df)
        self.conv2 = nn.Conv2d(in_channels=outfea, out_channels=outfea, kernel_size=3, stride=1, pad_mode=pad,
                               data_format=df)
        self.bn2 = nn.BatchNorm2d(num_features=outfea, data_format=df)
        self.conv3 = nn.Conv2d(in_channels=outfea, out_channels=outfea, kernel_size=3, stride=1, pad_mode=pad,
                               data_format=df)
        self.bn3 = nn.BatchNorm2d(num_features=outfea, data_format=df)
        self.conv4 = nn.Conv2d(in_channels=outfea, out_channels=infea, kernel_size=3, stride=1, pad_mode=pad,
                               data_format=df)
        self.bn4 = nn.BatchNorm2d(num_features=infea, data_format=df)
        self.act = nn.ReLU()

    def construct(self, input_geo):  # (none,none,2)
        x = self.bn1(self.conv1(input_geo))
        x = self.act(x)
        x = self.bn2(self.conv2(x))
        x = self.act(x)
        x = self.bn3(self.conv3(x))
        x = self.act(x)
        x = self.bn4(self.conv4(x))
        return self.act(x)


class addconv3d(nn.Cell):
    def __init__(self, infea, outfea, stra=(1, 1, 1), ks=(3, 3, 3), pad='same', df='NCDHW'):
        super(addconv3d, self).__init__()
        self.conv3d1 = nn.Conv3d(in_channels=infea, out_channels=outfea, kernel_size=ks, stride=stra, pad_mode=pad,
                                 data_format=df)
        self.bn3d1 = nn.BatchNorm3d(num_features=outfea, data_format=df)
        self.act = nn.ReLU()

    def construct(self, x):
        x = self.bn3d1(self.conv3d1(x))
        return self.act(x)


class convdownsampling(nn.Cell):
    def __init__(self, infea, outfea, stra=(2, 2, 2), ks=(3, 3, 3), pad='same', df='NCDHW'):
        super(convdownsampling, self).__init__()
        self.addconv3d1 = addconv3d(infea, outfea, stra)
        self.addconv3d2 = addconv3d(outfea, outfea)
        self.addconv3d3 = addconv3d(outfea, outfea)

    def construct(self, x):
        x = self.addconv3d1(x)
        x = self.addconv3d2(x)
        return self.addconv3d3(x)


class convupsampling(nn.Cell):
    def __init__(self, infea, outfea, stra=(2, 2, 2), ks=(3, 3, 3), pad='same', df='NCDHW'):
        super(convupsampling, self).__init__()
        self.deconv3d = nn.Conv3dTranspose(in_channels=infea, out_channels=outfea, kernel_size=ks, stride=stra,
                                           pad_mode=pad, data_format=df)
        self.bn3d = nn.BatchNorm3d(num_features=outfea, data_format=df)
        self.act = nn.ReLU()

    def construct(self, x):
        x = self.bn3d(self.deconv3d(x))
        return self.act(x)


class learnreg(nn.Cell):
    def __init__(self, infea, outfea, stra=1, pad='same', df='NCDHW'):  # 20 8 1
        super(learnreg, self).__init__()
        self.conv3d1 = addconv3d(infea, outfea)
        self.conv3d2 = addconv3d(outfea, outfea)  # C=8
        self.downsampling1 = convdownsampling(outfea, 2 * outfea)
        self.downsampling2 = convdownsampling(2 * outfea, 4 * outfea)
        self.upsampling1 = convupsampling(4 * outfea, 2 * outfea)
        self.upsampling2 = convupsampling(2 * outfea, outfea)
        self.conv3dlast = addconv3d(outfea, 1)
        self.add = mds.ops.Add()

    def construct(self, x):
        x = self.conv3d2(self.conv3d1(x))
        # print(x.shape)
        temp1 = x  # c
        x = self.downsampling1(x)
        # print(x.shape)
        temp2 = x  # 2c
        x = self.downsampling2(x)  # 4c
        # print(x.shape)
        x = self.add(self.upsampling1(x), temp2)  # 2c
        # print(x.shape)
        x = self.add(self.upsampling2(x), temp1)  # c
        # print(x.shape)
        x = self.conv3dlast(x)
        # print(x.shape)
        return x


class cyclecolornet(nn.Cell):
    def __init__(self, base_num_filter=8, max_d=48, num_res=8):
        super(cyclecolornet, self).__init__()
        self.pad = mds.ops.Pad(paddings=((0, 0), (0, 0), (max_d, 0), (0, 0)))  # BCHW
        self.concat_dim1 = mds.ops.Concat(1)
        self.softmax = mds.ops.Softmax(axis=1)

        self.createunifea = UniFeature(3, base_num_filter)
        self.resnet = resnetblock(base_num_filter, base_num_filter)
        self.createcomfea = processgeofeature(2, base_num_filter)
        self.learnreg = learnreg((base_num_filter + 2) * 2, base_num_filter)
        self.base_num_filter = base_num_filter
        self.max_d = max_d
        self.num_res = num_res

    def construct(self, inputs):
        
        left_input = inputs[:, 0, :, :, :]
        right_input = inputs[:, 1, :, :, :]
        result1 = self.precons([left_input, right_input])
        result1_reverse = self.__img_reverse__(result1)
        right_input_reverse = self.__img_reverse__(right_input)
        result2 = self.precons([right_input_reverse, result1_reverse])

        result2 = self.__img_reverse__(result2)
        result1_img = self.__getvuydata__(result1)
        result2_img = self.__getvuydata__(result2)
        put_img_volume = self.__concatimg__([result1_img, result2_img])
        return put_img_volume

    def precons(self, inputs):  # 5 h w ,5 h w
        left_input, right_input = inputs
        left_img = self.__getvuydata__(left_input)
        right_img = self.__getvuydata__(right_input)
        left_geo_feat = self.__getgeofeat__(left_input)
        right_geo_feat = self.__getgeofeat__(right_input)
        l_app_feature = self.createunifea(left_img)
        for res in range(self.num_res):
            l_app_feature = self.resnet(l_app_feature)  # 3->8
        r_app_feature = self.createunifea(right_img)
        for res in range(self.num_res):
            r_app_feature = self.resnet(r_app_feature)  # 3->8
        l_geo_feature = self.createcomfea(left_geo_feat)  # 2->8 -> 2
        r_geo_feature = self.createcomfea(right_geo_feat)  # 2 -> 8->2
        l_feature = self.__concatimg__([l_app_feature, l_geo_feature])  # 10=8+2
        r_feature = self.__concatimg__([r_app_feature, r_geo_feature])  # 10=8+2
        unifeature = [l_feature, r_feature]
        fv = self.__getfeaturevolume__(unifeature, self.max_d)  # N 20 D H W
        cv = self.learnreg(fv)  # n 1 d h w
        squeze = mds.ops.Squeeze(1)
        wv = squeze(cv)  # n d h w
        unidata = [wv, right_img]
        clolorization_result = self.__getweightaverage__(unidata, self.max_d)  # bchw
        output_result = self.__concatimg__([clolorization_result, left_geo_feat])
        return output_result

    def __img_reverse__(self, img):
        return mds.Tensor(np.fliplr(img.asnumpy()))

    def __getvuydata__(self, inputs):
        return inputs[:, 0:3, :, :]  # BCHW

    def __getgeofeat__(self, inputs):
        return inputs[:, 3:5, :, :]

    def __getyadta__(self, inputs):
        return inputs[:, 2:3, :, :]

    def __getvudata__(self, inputs):
        return inputs[:, 0:2, :, :]

    def __concatimg__(self, inputs):  # BCHW
        l, r = inputs
        ops = mds.ops.Concat(axis=1)
        return ops((l, r))

    def __getfeaturevolume__(self, inputs, max_d):
        left_tensor, right_tensor = inputs
        shape = right_tensor.shape
        right_tensor = self.pad(right_tensor)
        disparity_costs = []
        for d in reversed(range(max_d)):
            left_tensor_slice = left_tensor
            slice_op = mds.ops.Slice()
            right_tensor_slice = slice_op(right_tensor, (0, 0, d, 0), (shape[0], shape[1], shape[2], shape[3]))
            cost = self.concat_dim1((left_tensor_slice, right_tensor_slice))
            disparity_costs.append(cost)
        stack = mds.ops.Stack(axis=2)
        feature_volume = stack(disparity_costs)
        return feature_volume
    def __getweightaverage__(self, inputs, max_d):
        fv, right_image = inputs
        weight = self.softmax(fv)  # bdhw
        ref_V = right_image[:, 0:1, :, :]
        ref_U = right_image[:, 1:2, :, :]
        ref_Y = right_image[:, 2:3, :, :]
        right_tensor = ref_U
        shape = right_tensor.shape
        right_tensor = self.pad(right_tensor)
        disparity_costs = []
        for d in reversed(range(max_d)):
            slice_op = mds.ops.Slice()
            right_tensor_slice = slice_op(right_tensor, (0, 0, d, 0), (shape[0], shape[1], shape[2], shape[3]))
            disparity_costs.append(right_tensor_slice)
        stack = mds.ops.Stack(axis=2)
        cost_volume = stack(disparity_costs)
        squeeze = mds.ops.Squeeze(1)
        values = squeeze(cost_volume)  # b d h w
        mul = mds.ops.Mul()
        c = mul(weight, values)
        reduce_sum = mds.ops.ReduceSum(keep_dims=False)
        U_map = reduce_sum(c, 1)
        right_tensor = ref_V
        shape = right_tensor.shape
        right_tensor = self.pad(right_tensor)
        disparity_costs = []
        for d in reversed(range(max_d)):
            slice_op = mds.ops.Slice()
            right_tensor_slice = slice_op(right_tensor, (0, 0, d, 0), (shape[0], shape[1], shape[2], shape[3]))
            disparity_costs.append(right_tensor_slice)
        stack = mds.ops.Stack(axis=2)
        cost_volume = stack(disparity_costs)
        squeeze = mds.ops.Squeeze(1)
        values = squeeze(cost_volume)  # b d h w
        mul = mds.ops.Mul()
        c = mul(weight, values)
        reduce_sum = mds.ops.ReduceSum(keep_dims=False)
        V_map = reduce_sum(c, 1)
        right_tensor = ref_Y
        shape = right_tensor.shape
        right_tensor = self.pad(right_tensor)
        disparity_costs = []
        for d in reversed(range(max_d)):
            slice_op = mds.ops.Slice()
            right_tensor_slice = slice_op(right_tensor, (0, 0, d, 0), (shape[0], shape[1], shape[2], shape[3]))
            disparity_costs.append(right_tensor_slice)
        stack = mds.ops.Stack(axis=2)
        cost_volume = stack(disparity_costs)
        squeeze = mds.ops.Squeeze(1)
        values = squeeze(cost_volume)  # b d h w
        mul = mds.ops.Mul()
        c = mul(weight, values)
        reduce_sum = mds.ops.ReduceSum(keep_dims=False)
        Y_map = reduce_sum(c, 1)
        stack = mds.ops.Stack(axis=1)
        VUY_map = stack([V_map, U_map, Y_map])
        return VUY_map



###加载数据  设置batchsize
tra,val=load_data_path(left_path,right_path,split_ratio)
print(tra[:2],len(tra))
dataset=imgDataset(tra)
traindataset=ds.GeneratorDataset(dataset,column_names=['img','label'],num_parallel_workers=4)
traindataset=traindataset.batch(1)
NET=cyclecolornet()



class MISS_1(nn.Cell):
    def __init__(self):
        super(MISS_1, self).__init__()
        self.lossfn = nn.SSIM()

    def construct(self, data, label):
        ssim = self.lossfn(data, label)
        ones = mdsnp.full_like(ssim, 1)
        return ones - ssim


class SLoss(nn.Cell):
    def __init__(self, base_num_filter=8):
        super(SLoss, self).__init__()
        self.exp = ops.Exp()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)

    def construct(self, pred, true):
        loss = self.getloss(pred, true)
        return loss

    def getloss(self, pred, true):

        y_true = true[:, 0:6, 10:-10, 10:-10]
        y_pred = pred[:, 0:6, 10:-10, 10:-10]
        y_true_V = y_true[:, 0:1, :, :]
        y_true_U = y_true[:, 1:2, :, :]
        y_true_Y = y_true[:, 2:3, :, :]
        y_true_reverse_V = y_true[:, 3:4, :, :]
        y_true_reverse_U = y_true[:, 4:5, :, :]
        y_true_reverse_Y = y_true[:, 5:6, :, :]

        y_pred_V = y_pred[:, 0:1, :, :]
        y_pred_U = y_pred[:, 1:2, :, :]
        y_pred_Y = y_pred[:, 2:3, :, :]
        y_pred_reverse_V = y_pred[:, 3:4, :, :]
        y_pred_reverse_U = y_pred[:, 4:5, :, :]
        y_pred_reverse_Y = y_pred[:, 5:6, :, :]
        ssim1 = self.tf_ssim011(y_pred_Y, y_true_Y, max_val=255.0)
        ssim2 = self.tf_ssim(y_pred_reverse_V, y_true_reverse_V, max_val=255.0)
        ssim3 = self.tf_ssim(y_pred_reverse_U, y_true_reverse_U, max_val=255.0)

        ssim = (ssim1 + ssim2 + ssim3) / 3.0
        return 1 - ssim

    def tf_ssim(self, img1, img2, max_val=1, cs_map=False, mean_metric=True):
        K1 = 0.01
        K2 = 0.03
        L = max_val  # depth of image (255 in case the image has a different scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = self.conv2d(img1)
        mu2 = self.conv2d(img2)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.conv2d(img1 * img1) - mu1_sq
        sigma2_sq = self.conv2d(img2 * img2) - mu2_sq
        sigma12 = self.conv2d(img1 * img2) - mu1_mu2
        if cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                  (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                 (sigma1_sq + sigma2_sq + C2))
        reducemean = ops.ReduceMean()
        if mean_metric:
            value = reducemean(value)
        return value

    def tf_ssim011(self, img1, img2, max_val=1, mean_metric=True):
        K1 = 0.01
        K2 = 0.03
        L = max_val  # depth of image (255 in case the image has a different scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = self.conv2d(img1)
        mu2 = self.conv2d(img2)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = self.conv2d(img1 * img1) - mu1_sq
        sigma2_sq = self.conv2d(img2 * img2) - mu2_sq
        sigma12 = self.conv2d(img1 * img2) - mu1_mu2
        value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        if mean_metric:
            value = value.mean()
        return value



###training
Loss2=SLoss()
trainDataset=imgDataset(tra)
trainData=ds.GeneratorDataset(trainDataset,column_names=['img','label'],num_parallel_workers=4)
trainData=trainData.batch(1)
optim=nn.RMSProp(params=NET.trainable_params(), learning_rate=0.001)
trainnet=Model(NET,loss_fn=Loss2,optimizer=optim)
loss_cb = LossMonitor(per_print_times=1)
ckpt_config = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
ckpoint_cb = ModelCheckpoint(prefix='coloring', directory='./model', config=ckpt_config)
print('start    training')
trainnet.train(8,trainData,callbacks=[loss_cb,ckpoint_cb])








