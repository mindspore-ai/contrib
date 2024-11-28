# SciNet (MindSpore)

This code is a mindspore implementation of SciNet which is avaliable at https://github.com/fd17/SciNet_PyTorch ,introduced in the paper https://arxiv.org/abs/1807.10300 by Raban Iten, Tony Metger, Henrik Wilming, Lidia del Rio, and Renato Renner. The model uses a modified variational autoencoder (VAE) approach with neural networks to automatically discover physical properties from observational data.

## Requirements

- Python 3.9
- MindSpore 2.3
- Matplotlib
- Jupyter Notebook

### Running on Huawei Cloud ModelArts

This implementation can be directly run on Huawei Cloud’s ModelArts platform using the *Guizhou 1* node. For this, select the environment image:

````
mindspore_2.3.0-cann_8.0.rc1-py_3.9-euler_2.10.7-aarch64-snt9b
````

## Usage

The SciNet architecture is defined in `models.py`. You can find example use cases in various Jupyter notebooks. The `Generate_Trainingdata.ipynb` notebook can be used to generate training data for a pendulum, as outlined in the paper. The `Training.ipynb` and `Analysis.ipynb` notebooks demonstrate the model’s training process and how to analyze it once training is complete.