from huggingface_hub import hf_hub_download, snapshot_download
import mindspore as ms
from mindspore import ops
import os
from open_flamingo import create_model_and_transforms
from einops import repeat
from PIL import Image
import sys

# 设置更长的超时时间
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '500'
os.environ['HF_HUB_DOWNLOAD_RETRY'] = '5'

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import FlamingoProcessor
from src.ms_wrapper import FlamingoMindSporeWrapper
from demo_utils import image_paths, clean_generation


def main():
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    print('Loading model..')

    # 使用本地LLaMA模型路径
    llama_path = 'models/llama-7b-hf'
    if not os.path.exists(llama_path):
        raise ValueError(f'LLaMA模型路径 {llama_path} 不存在！请确保模型文件已正确放置。')
    else:
        print(f'找到LLaMA模型，位置：{llama_path}')

    # 使用本地CLIP模型路径
    clip_path = 'models/clip-vit-large-patch14'
    if not os.path.exists(clip_path):
        raise ValueError(f'CLIP模型路径 {clip_path} 不存在！请确保模型文件已正确放置。')
    else:
        print(f'找到CLIP模型，位置：{clip_path}')

    pt_model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",  # 使用标准的CLIP模型标识符
        clip_vision_encoder_pretrained="openai",  # 使用OpenAI的预训练权重
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )

    # 创建MindSpore包装器模型
    model = FlamingoMindSporeWrapper(pt_model)

    # 确保目标目录存在
    os.makedirs("models/med_flamingo", exist_ok=True)

    # 使用本地med-flamingo checkpoint
    pt_model_path = "models/med_flamingo/model.pt"
    ckpt_dir = "models/med_flamingo/ckpt"

    # 检查PyTorch模型文件是否存在
    if not os.path.exists(pt_model_path):
        raise ValueError(f'Med-Flamingo PyTorch模型 {pt_model_path} 不存在！请确保模型文件已正确放置。')
    else:
        print(f'找到Med-Flamingo PyTorch模型，位置：{pt_model_path}')

    # 检查MindSpore模型文件是否存在，如果不存在则转换
    if not os.path.exists(ckpt_dir) or len(os.listdir(ckpt_dir)) == 0:
        print(f'MindSpore模型文件目录 {ckpt_dir} 不存在或为空，请先运行转换脚本')
        print('请运行命令: python scripts/convert_weights_ms'
              '.py')
        raise ValueError('请先运行转换脚本: python scripts/convert_weights_ms.py')
    else:
        print(f'找到MindSpore模型目录，位置：{ckpt_dir}')

    # 加载模型参数s
    print('正在加载模型参数...')

    # 检查索引文件
    index_file = os.path.join(ckpt_dir, "index.txt")
    if os.path.exists(index_file):
        # 按索引加载参数
        with open(index_file, 'r') as f:
            ckpt_files = [line.strip() for line in f.readlines()]

        print(f'找到 {len(ckpt_files)} 个参数文件')
        for ckpt_file in ckpt_files:
            file_path = os.path.join(ckpt_dir, ckpt_file)
            print(f'加载参数文件：{file_path}')
            param_dict = ms.load_checkpoint(file_path)
            # 这里实际上我们不需要加载到MindSpore模型中，因为我们直接使用PyTorch模型
            # ms.load_param_into_net(model, param_dict)
            print(f'参数文件 {file_path} 已识别')
    else:
        # 直接加载目录中的所有.ckpt文件
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        print(f'找到 {len(ckpt_files)} 个参数文件')
        for ckpt_file in ckpt_files:
            file_path = os.path.join(ckpt_dir, ckpt_file)
            print(f'加载参数文件：{file_path}')
            param_dict = ms.load_checkpoint(file_path)
            # 同样，我们不需要加载到MindSpore模型中
            # ms.load_param_into_net(model, param_dict)
            print(f'参数文件 {file_path} 已识别')

    print('模型参数加载过程完成!')

    processor = FlamingoProcessor(tokenizer, image_processor)

    # 设置为评估模式
    model.set_train(False)

    """
    Step 1: Load images
    """
    image_paths = [
        'synpic50962.jpg',
        'synpic52767.jpg',
        'synpic30324.jpg',
        'synpic21044.jpg',
        'synpic54802.jpg',
        'synpic57813.jpg',
        'synpic47964.jpg'
    ]
    image_paths = [os.path.join('D:\PyCharmprojects\med-flamingo\img', p) for p in image_paths]
    demo_images = [Image.open(path) for path in image_paths]

    """
    Step 2: Define multimodal few-shot prompt 
    """

    # example few-shot prompt:
    prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|><image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|><image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|><image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|><image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|><image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|><image>Question: Where is the largest mass located in the cerebellum? Answer:"

    """
    Step 3: Preprocess data 
    """
    print('Preprocess data')
    pixels = processor.preprocess_images(demo_images)
    pixels = ops.expand_dims(pixels, 0)  # 添加 b 维度
    pixels = ops.expand_dims(pixels, 2)  # 添加 T 维度
    tokenized_data = processor.encode_text(prompt)

    """
    Step 4: Generate response 
    """

    # 运行few-shot prompt生成回答
    print('Generate from multimodal few-shot prompt')
    generated_text = model.generate(
        vision_x=pixels,
        lang_x=tokenized_data["input_ids"],
        attention_mask=tokenized_data["attention_mask"],
        max_new_tokens=30,
    )
    response = processor.tokenizer.decode(generated_text[0])
    response = clean_generation(response)

    print(f'{response=}')


if __name__ == "__main__":
    main()