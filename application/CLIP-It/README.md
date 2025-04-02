# CLIP-It MindSpore

MindSpore implementation of the paper "CLIP-It! Language-Guided Video Summarization".

## 概述

这是一个从PyTorch版本 [srpkdyy/CLIP-It](https://github.com/srpkdyy/CLIP-It) 迁移到MindSpore的实现。当前版本主要用于测试功能结构迁移是否正确，已暂时替换了CLIP模型部分为模拟实现。

## 环境要求

- MindSpore >= 1.10.0
- NumPy >= 1.19.0

## 安装

1. 安装 MindSpore:
```bash
pip install mindspore