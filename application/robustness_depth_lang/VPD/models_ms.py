from mindspore import nn
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel

class FrozenCLIPEmbedder(nn.Cell):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="../model", device="cpu", max_length=77, pool=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

        self.pool = pool

    def freeze(self):
        self.transformer.eval()
        for param in self.trainable_params():
            param.requires_grad = False

    def construct(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        if self.pool:
            z = outputs.pooler_output
        else:
            z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
