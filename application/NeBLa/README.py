This code is a mindspore implementation of NeBLa(Neural Beer-Lambert for 3D Reconstruction of Oral Structures from Panoramic Radiographs) which is avaliable at https://github.com/sihwa-park/nebla.
In this repository, you'll find the core model classes used in the generation module.

image_encoder.py: The final encoder-decoder model for refinement is a 3D-UNet model, and the corresponding code can be found at pytorch-3dunet.
point_embedder.py: The point embedder
mlp.py: The MLP model 