import mindspore as ms
from mindspore import ops
from mindnlp.transformers import AutoProcessor, CLIPModel,AutoTokenizer
from PIL import Image

# Load the model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

#Load the image
img = Image.open('maxi_dress.jpg')

prompt = "a photo of a"
text_inputs = ["blue cowl neck maxi-dress", "red t-shirt", "white shirt"]
text_inputs = [prompt + " " + t for t in text_inputs]

tokenized_prompt = tokenizer(text_inputs,padding=True,return_tensors="ms")
inputs = processor(images=img, return_tensors="ms")

model.set_train(False)
image_features = model.get_image_features(**inputs)
text_features = model.get_text_features(**tokenized_prompt)
image_features /= ops.norm(image_features, dim=-1, keepdim=True)
text_features /= ops.norm(text_features, dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(axis=-1)

print("Labels probs:", text_probs)