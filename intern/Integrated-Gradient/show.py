from PIL import Image
import argparse

from mindspore import context
import mindspore.dataset.transforms  as T
import mindspore.dataset.vision as V
import mindcv

from ig import Integrated_Gradient



def get_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('-img', help='img path')
    parser.add_argument('-step', default=20, type=int, help='number of steps in the path integral')
    return parser.parse_args()
    


def main():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    arg = get_arg()
    preprocess = T.Compose([V.Resize(256),
                            V.CenterCrop(224),
                            V.ToTensor(),
                            V.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225], is_hwc=False)])

    model = mindcv.create_model('mobilenet_v2_100', pretrained=True)
    ig = Integrated_Gradient(model, preprocess, arg.step)
    
    print('img:', arg.img)
    
    img = Image.open(arg.img).convert('RGB')
    # output is torch Tensor, heatmap is ndarray
    output, heatmap = ig.get_heatmap(img, baseline=None)
    print('\nPredict label:', output.max(axis = 1,return_indices=True)[1].item())

    w, h = img.size
    heatmap = Image.fromarray(heatmap)
    result = Image.new('RGB', (2 * w, h))
    result.paste(img)
    result.paste(heatmap, (w, 0))
    #result.show()
    result.save("result.png")
    


if __name__ == "__main__":
    main()
