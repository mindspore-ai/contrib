"""
    This module is used to convert the image data to .mindrecord file.
"""
import os.path
import glob
from mindspore.mindrecord import FileWriter


data_record_path = r"convert_dataset_to_mindrecord/data_to_mindrecord/reasoning/test.mindrecord"
writer = FileWriter(file_name=data_record_path, shard_num=1)

# 定义schema
data_schema = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
writer.add_schema(data_schema, "train_schema")

# 添加数据索引字段
indexes = ["file_name", "label"]
writer.add_index(indexes)

# 数据准备
pneumonia_dir = r"convert_dataset_to_mindrecord/data_to_mindrecord/reasoning/"
pneumonia_file_list = glob.glob(os.path.join(pneumonia_dir, "*.jpeg"))

data = []
for file_name in pneumonia_file_list:
    with open(file_name, "rb") as f:
        bytes_data = f.read()
    data.append({"file_name": file_name.split("\\")[1], "label": 0, "data": bytes_data})

# 数据写入
writer.write_raw_data(data)

# 生成本地数据
writer.commit()
