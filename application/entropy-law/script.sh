#!/bin/bash

# 下载文件并重命名为 instag_mix.txt
curl -L -o instag_mix.txt "https://huggingface.co/datasets/AndrewZeng/deita_sota_pool/resolve/main/instag_mix.json?download=true"

# 运行 formatter.py 将文件标准化为 instag_mix_formatter.json
python formatter.py --input instag_mix.txt --output instag_mix_formatter.json

# 使用 ZIP.py 进行选择操作，并保存结果为 selected_data.json
python ZIP.py --data_path instag_mix_formatter.json --save_path selected_data.json --budget 1000

# 打印输出结果文件路径
echo "Process completed. The output is saved in selected_data.json"
