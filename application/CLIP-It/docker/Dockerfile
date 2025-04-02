FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    git 

RUN pip install -U \
    einops

# CLIP
RUN pip install -U \
    ftfy \
    regex \
    tqdm

RUN pip install -U \
    git+https://github.com/openai/CLIP.git