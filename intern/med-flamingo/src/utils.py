import mindspore as ms
import numpy as np
from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    """
    Abstract class for processors to show what methods they need to implement.
    Processors handle text encoding and image preprocessing.
    """
    @abstractmethod
    def encode_text(self, prompt):
        pass

    @abstractmethod
    def preprocess_images(self, images: list):
        pass


class FlamingoProcessor(AbstractProcessor):
    """
    Processor class for Flamingo adapted for MindSpore.
    """
    def __init__(self, tokenizer, vision_processor):
        """
        初始化处理器
        Args:
            tokenizer: 文本分词器
            vision_processor: 图像预处理器
        """
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
    
    def encode_text(self, prompt):
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")
            
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer([prompt], return_tensors="np")
            return {
                "input_ids": ms.Tensor(encoded["input_ids"], dtype=ms.int32),
                "attention_mask": ms.Tensor(encoded["attention_mask"], dtype=ms.int32)
            }
        except Exception as e:
            raise RuntimeError(f"Error during text encoding: {str(e)}")
    
    def preprocess_images(self, images: list):
        if not isinstance(images, list):
            raise TypeError("Images must be provided as a list")
            
        try:
            vision_x = []
            for im in images:
                # 处理单个图像
                processed = self.vision_processor(im)
                # 确保处理后的图像是正确的格式
                if not isinstance(processed, (np.ndarray, ms.Tensor)):
                    processed = np.array(processed)
                tensor = ms.Tensor(processed)
                vision_x.append(tensor.expand_dims(0))
            
            # 连接所有图像
            vision_x = ms.ops.concat(vision_x, axis=0)
            return vision_x
            
        except Exception as e:
            raise RuntimeError(f"Error during image preprocessing: {str(e)}")
    

