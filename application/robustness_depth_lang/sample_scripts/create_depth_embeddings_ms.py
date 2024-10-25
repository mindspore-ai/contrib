import mindspore as ms
import mindspore.ops as ops
from tqdm import tqdm
import csv
import json
import numpy as np

def main():
    import sys
    sys.path.append('../')
    from VPD.models_ms import FrozenCLIPEmbedder

    text_encoder = FrozenCLIPEmbedder()


    config_path = "../data/zero_shot_depth_data.json"
    base_path = '/nyu_depth_v2/official_splits/test'
    
    depth_data = {}
    with open(config_path, 'r') as f:
        data = json.load(f)
        for rows in data:
            image_path = rows['test_image_path'].split(base_path)[1]
            depth_data[image_path] = rows
    
    file_name_path = 'depth_file_list.txt'
    ordered_data = []
    with open(file_name_path, 'r') as f:
        file = csv.reader(f)
        for rows in file:
            ordered_data.append(rows[0].split(' ')[0])
    
    zeroshot_weights = []
    stop_gradient=ops.stop_gradient

    for data in tqdm(ordered_data):
        depth_sentence = depth_data[data]['depth_sentences']
        texts = depth_sentence[0] +  ". " + depth_sentence[1] +  ". " + depth_sentence[2] +  ". " + depth_sentence[3]
        embeddings = text_encoder.encode([texts])
        embeddings=stop_gradient(embeddings).detach().squeeze()
        zeroshot_weights.append(embeddings)

    # 检查所有元素的形状是否一致
    shapes = {tensor.shape for tensor in zeroshot_weights}
    if len(shapes) == 1:
        zeroshot_weights_np = np.stack([tensor.numpy() for tensor in zeroshot_weights])
    else:
        raise ValueError("The shapes of elements in zeroshot_weights are inconsistent.")
    # 检查形状
    print("Shape of zeroshot_weights_np:", zeroshot_weights_np.shape)

    # 检查数据类型
    print("Data type of zeroshot_weights_np:", zeroshot_weights_np.dtype)

    # 转换为 mindspore.Tensor
    try:
        zeroshot_weights_tensor = ms.Tensor(zeroshot_weights_np, dtype=ms.float32)
        print("Conversion successful!")
    except ValueError as e:
        print(f"Conversion failed: {e}")

    zeroshot_weights_parameter=ms.Parameter(zeroshot_weights_tensor,name='zeroshot_weights')
    checkpoint_data = {'zeroshot_weights': zeroshot_weights_parameter}

    ms.save_checkpoint(checkpoint_data, '../depth_embeddings/depth_ms.ckpt')

if __name__ == '__main__':
    main()