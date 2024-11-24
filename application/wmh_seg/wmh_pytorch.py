import mindspore as ms
from mindspore import nn, ops, Tensor, context
import nibabel as nib
import numpy as np
from einops import rearrange
from tqdm import tqdm
import sys
import mindspore.dataset.vision as vision
from mindspore import context, load_checkpoint, load_param_into_net


in_path = sys.argv[1]
out_path = sys.argv[2]
gpu = sys.argv[3]
wmh_seg_home = sys.argv[4]
verbose = sys.argv[5]
pmb = sys.argv[6]
batch = np.int16(sys.argv[7])
fast = sys.argv[8]

def reduceSize(prediction):
    arg = prediction > 0.5
    out = np.zeros(prediction.shape)
    out[arg] = 1
    return out

def wmh_seg(in_path, out_path, train_transforms, device, mode):
    if mode == "True":
        img_orig = nib.load(in_path)
    
        transform = vision.Resize((256, 256, 256))
        
        img = transform(img_orig)
        input = np.squeeze(img.get_fdata())
        input = Tensor(input, ms.float32)
        affine = img.affine
        input = ops.ExpandDims()(input, 1)
        prediction_axial = np.zeros((256, 256, 256))
        prediction_cor = np.zeros((256, 256, 256))
        prediction_sag = np.zeros((256, 256, 256))
    else:
        img_orig = nib.load(in_path)
        
        img = train_transforms(img_orig)
        input = np.squeeze(img.get_fdata())
        prediction_axial = np.zeros((256, 256, 256))
        prediction_cor = np.zeros((256, 256, 256))
        prediction_sag = np.zeros((256, 256, 256))
        input = Tensor(input, ms.float32)
        affine = img.affine
        input = ops.ExpandDims()(input, 1)

    if device == "GPU":
        context = ms.context
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    prediction_input = input / input.max()
    print(f'Predicting.....')

    if verbose == "True":
        if fast == "True":
            for idx in tqdm(range(input.shape[0] // batch)):
                axial_img = rearrange(prediction_input[:, :, :, idx * batch:(idx + 1) * batch],
                                    'd0 d1 d2 d3 -> d3 d1 d0 d2').repeat(1, 3, 1, 1)
                prediction_axial[:, :, idx * batch:(idx + 1) * batch] = rearrange(
                    model(axial_img.astype(ms.float32))[0:batch].asnumpy(),
                    'd0 d1 d2 d3 -> d2 d3 (d0 d1)'
                )
            prediction = prediction_axial
        else:
            for idx in tqdm(range(input.shape[0] // batch)):
                axial_img = rearrange(prediction_input[:, :, :, idx * batch:(idx + 1) * batch],
                                    'd0 d1 d2 d3 -> d3 d1 d0 d2').repeat(1, 3, 1, 1)
                cor_img = rearrange(prediction_input[:, :, idx * batch:(idx + 1) * batch, :],
                                    'd0 d1 d2 d3 -> d2 d1 d0 d3').repeat(1, 3, 1, 1)
                sag_img = rearrange(prediction_input[idx * batch:(idx + 1) * batch, :, :, :],
                                    'd0 d1 d2 d3 -> d0 d1 d2 d3').repeat(1, 3, 1, 1)

                stacked_input = ops.Concat(axis=0)((axial_img, cor_img, sag_img))
                
                prediction_axial[:, :, idx * batch:(idx + 1) * batch] = rearrange(
                    model(stacked_input.astype(ms.float32))[0:batch].asnumpy(),
                    'd0 d1 d2 d3 -> d2 d3 (d0 d1)'
                )
                prediction_cor[:, idx * batch:(idx + 1) * batch, :] = rearrange(
                    model(stacked_input.astype(ms.float32))[batch:2 * batch].asnumpy(),
                    'd0 d1 d2 d3 -> d2 (d0 d1) d3'
                )
                prediction_sag[idx * batch:(idx + 1) * batch, :, :] = rearrange(
                    model(stacked_input.astype(ms.float32))[2 * batch:].asnumpy(),
                    'd0 d1 d2 d3 -> (d0 d1) d2 d3'
                )

            prediction = prediction_axial + prediction_cor + prediction_sag

    elif verbose != "True":
        if fast == "True":
            for idx in range(input.shape[0] // batch):
                axial_img = rearrange(prediction_input[:, :, :, idx * batch:(idx + 1) * batch],
                                    'd0 d1 d2 d3 -> d3 d1 d0 d2').repeat(1, 3, 1, 1)
                prediction_axial[:, :, idx * batch:(idx + 1) * batch] = rearrange(
                    model(axial_img.astype(ms.float32))[0:batch].asnumpy(),
                    'd0 d1 d2 d3 -> d2 d3 (d0 d1)'
                )
            prediction = prediction_axial
        else:
            for idx in range(input.shape[0] // batch):
                axial_img = rearrange(prediction_input[:, :, :, idx * batch:(idx + 1) * batch],
                                    'd0 d1 d2 d3 -> d3 d1 d0 d2').repeat(1, 3, 1, 1)
                cor_img = rearrange(prediction_input[:, :, idx * batch:(idx + 1) * batch, :],
                                    'd0 d1 d2 d3 -> d2 d1 d0 d3').repeat(1, 3, 1, 1)
                sag_img = rearrange(prediction_input[idx * batch:(idx + 1) * batch, :, :, :],
                                    'd0 d1 d2 d3 -> d0 d1 d2 d3').repeat(1, 3, 1, 1)

                stacked_input = ops.Concat(axis=0)((axial_img, cor_img, sag_img))

                prediction_axial[:, :, idx * batch:(idx + 1) * batch] = rearrange(
                    model(stacked_input.astype(ms.float32))[0:batch].asnumpy(),
                    'd0 d1 d2 d3 -> d2 d3 (d0 d1)'
                )
                prediction_cor[:, idx * batch:(idx + 1) * batch, :] = rearrange(
                    model(stacked_input.astype(ms.float32))[batch:2 * batch].asnumpy(),
                    'd0 d1 d2 d3 -> d2 (d0 d1) d3'
                )
                prediction_sag[idx * batch:(idx + 1) * batch, :, :] = rearrange(
                    model(stacked_input.astype(ms.float32))[2 * batch:].asnumpy(),
                    'd0 d1 d2 d3 -> (d0 d1) d2 d3'
                )

            prediction = prediction_axial + prediction_cor + prediction_sag

    out = reduceSize(prediction)


    img_fit = input.squeeze().asnumpy()

    resize_transform = vision.Resize((img_orig.get_fdata().shape[0],
                                    img_orig.get_fdata().shape[1],
                                    img_orig.get_fdata().shape[2]))

    out = np.expand_dims(out, 0)
    out = resize_transform(Tensor(out, ms.float32)).asnumpy()
    out = reduceSize(np.squeeze(out))

    nii_seg = nib.Nifti1Image(out, affine=img_orig.affine)
    nib.save(nii_seg, out_path)


filename = in_path.split('/')[-1]
ID = filename.split('.')[0]

if gpu == 'True':
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    print('Configuring model on GPU')
else:
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    print('Configuring model on CPU')

if pmb == "False":
    train_transforms = vision.Resize((256, 256, 256))
    
    model = nn.UNet3D(in_channels=1, out_channels=1)
    param_dict = load_checkpoint(f"{wmh_seg_home}/ChallengeMatched_Unet_mit_b5.ckpt")
    load_param_into_net(model, param_dict)
    model.set_train(False)

    wmh_seg(in_path, out_path, train_transforms, "GPU" if gpu == 'True' else "CPU", pmb)

elif pmb == "True":
    train_transforms = vision.Resize((256, 256, 256))
    
    model = nn.UNet3D(in_channels=1, out_channels=1)
    param_dict = load_checkpoint(f"{wmh_seg_home}/pmb_2d_transformer_Unet_mit_b5.ckpt")
    load_param_into_net(model, param_dict)
    model.set_train(False)

    wmh_seg(in_path, out_path, train_transforms, "GPU" if gpu == 'True' else "CPU", pmb)

