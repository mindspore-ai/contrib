"""
使用MindSpore实现的权重转换脚本
将PyTorch模型转换为小块MindSpore参数文件，解决内存溢出问题
"""

import os
import torch
import mindspore as ms
from mindspore import Tensor, Parameter
import numpy as np
import argparse
import shutil
import gc

def convert_pt_to_ms_chunks(pt_file, ckpt_dir, max_chunk_size=50*1024*1024):  # 默认50MB一个块
    """
    将PyTorch模型(.pt)转换为MindSpore模型(.ckpt)，并拆分为多个小文件
    Args:
        pt_file: PyTorch模型文件路径
        ckpt_dir: 输出的MindSpore参数目录
        max_chunk_size: 每个块的最大大小（字节）
    """
    print(f"正在加载PyTorch模型: {pt_file}")
    
    # 加载PyTorch模型，使用CPU以减少内存使用
    pt_checkpoint = torch.load(pt_file, map_location='cpu')
    
    # 确保输出目录存在
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 按层次组织参数
    param_groups = {}
    
    # 收集和分组参数
    for key, value in pt_checkpoint.items():
        if isinstance(value, torch.Tensor):
            # 按第一个点分隔获取组名
            parts = key.split('.', 1)
            group_name = parts[0]
            
            if group_name not in param_groups:
                param_groups[group_name] = {}
                
            # 将参数添加到相应的组
            param_groups[group_name][key] = value
    
    # 清理一下内存
    del pt_checkpoint
    gc.collect()
    
    print(f"将模型拆分为 {len(param_groups)} 个参数组")
    
    # 记录所有已保存的文件
    all_files = []
    
    # 转换并保存每一组，如果太大则进一步拆分
    for group_name, params in param_groups.items():
        # 估计当前组的大小
        group_size = sum(p.numel() * p.element_size() for p in params.values())
        
        if group_size > max_chunk_size:
            # 需要进一步拆分
            print(f"参数组 {group_name} 过大 ({group_size/1024/1024:.2f}MB)，进一步拆分")
            
            # 当前块的参数和大小
            current_chunk = {}
            current_size = 0
            chunk_index = 0
            
            for key, value in params.items():
                param_size = value.numel() * value.element_size()
                
                # 如果单个参数超过最大块大小，需要将其分割
                if param_size > max_chunk_size:
                    print(f"参数 {key} 过大 ({param_size/1024/1024:.2f}MB)，将其分割成多个块")
                    
                    # 将参数转换为NumPy数组
                    np_value = value.detach().cpu().numpy()
                    
                    # 计算需要多少块
                    num_chunks = int(np.ceil(param_size / max_chunk_size))
                    
                    # 计算每个块的元素数量
                    elements_per_chunk = int(np.ceil(value.numel() / num_chunks))
                    
                    # 清理内存
                    del value
                    gc.collect()
                    
                    # 分割参数并保存
                    for i in range(num_chunks):
                        start_idx = i * elements_per_chunk
                        end_idx = min((i + 1) * elements_per_chunk, np_value.size)
                        
                        # 创建块名称
                        safe_key = key.replace('.', '_').replace('[', '_').replace(']', '_')
                        chunk_name = f"{safe_key}_part{i}"
                        
                        # 提取部分并保存
                        chunk_file = os.path.join(ckpt_dir, f"{chunk_name}.ckpt")
                        
                        # 创建一个小的MindSpore参数列表
                        ms_params = []
                        
                        # 展平数组获取这一段
                        flat_array = np_value.flatten()[start_idx:end_idx]
                        
                        # 创建参数
                        ms_tensor = Tensor(flat_array)
                        
                        # 创建附加信息，记录原始形状和位置
                        info_param = {
                            'original_key': key,
                            'original_shape': np_value.shape,
                            'part': (i, num_chunks, start_idx, end_idx),
                            'is_chunk': True
                        }
                        
                        # 保存参数和元数据
                        ms_params.append(Parameter(ms_tensor, name=chunk_name))
                        
                        # 保存这一小块
                        ms.save_checkpoint(ms_params, chunk_file)
                        
                        # 记录文件
                        all_files.append(f"{chunk_name}.ckpt")
                        print(f"  保存参数 {key} 的第 {i+1}/{num_chunks} 块到 {chunk_file}")
                        
                        # 清理临时变量
                        del ms_tensor, ms_params
                        gc.collect()
                    
                    # 完全清理NumPy数组
                    del np_value
                    gc.collect()
                    
                else:
                    # 如果当前块加上这个参数会超过最大大小，保存当前块并开始一个新块
                    if current_size + param_size > max_chunk_size and current_chunk:
                        # 保存当前块
                        chunk_file = os.path.join(ckpt_dir, f"{group_name}_chunk{chunk_index}.ckpt")
                        ms_params = []
                        
                        for param_key, param_value in current_chunk.items():
                            # 转换参数
                            np_value = param_value.detach().cpu().numpy()
                            ms_tensor = Tensor(np_value)
                            ms_params.append(Parameter(ms_tensor, name=param_key))
                        
                        # 保存这一组参数
                        ms.save_checkpoint(ms_params, chunk_file)
                        
                        all_files.append(f"{group_name}_chunk{chunk_index}.ckpt")
                        print(f"  保存参数块 {group_name}_chunk{chunk_index} 到 {chunk_file}")
                        
                        # 重置当前块
                        current_chunk = {}
                        current_size = 0
                        chunk_index += 1
                        
                        # 清理
                        del ms_params
                        gc.collect()
                    
                    # 将参数添加到当前块
                    current_chunk[key] = value
                    current_size += param_size
            
            # 保存最后一个块（如果有）
            if current_chunk:
                chunk_file = os.path.join(ckpt_dir, f"{group_name}_chunk{chunk_index}.ckpt")
                ms_params = []
                
                for param_key, param_value in current_chunk.items():
                    # 转换参数
                    np_value = param_value.detach().cpu().numpy()
                    ms_tensor = Tensor(np_value)
                    ms_params.append(Parameter(ms_tensor, name=param_key))
                
                # 保存这一组参数
                ms.save_checkpoint(ms_params, chunk_file)
                
                all_files.append(f"{group_name}_chunk{chunk_index}.ckpt")
                print(f"  保存最后参数块 {group_name}_chunk{chunk_index} 到 {chunk_file}")
                
                # 清理
                del ms_params
                gc.collect()
        
        else:
            # 组不太大，直接保存
            ms_params = []
            
            for key, value in params.items():
                # 转换参数
                np_value = value.detach().cpu().numpy()
                ms_tensor = Tensor(np_value)
                ms_params.append(Parameter(ms_tensor, name=key))
            
            # 保存这一组参数
            group_file = os.path.join(ckpt_dir, f"{group_name}.ckpt")
            ms.save_checkpoint(ms_params, group_file)
            
            all_files.append(f"{group_name}.ckpt")
            print(f"保存参数组 {group_name} 到 {group_file}")
            
            # 清理
            del ms_params
            gc.collect()
    
    # 创建索引文件记录所有拆分的文件
    index_file = os.path.join(ckpt_dir, "index.txt")
    with open(index_file, 'w') as f:
        for file_name in all_files:
            f.write(f"{file_name}\n")
    
    # 创建元数据文件，记录分块信息和参数映射
    metadata_file = os.path.join(ckpt_dir, "metadata.json")
    import json
    metadata = {
        "num_files": len(all_files),
        "files": all_files,
        "max_chunk_size": max_chunk_size,
        "original_model": pt_file
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"模型转换完成! 参数已拆分为 {len(all_files)} 个文件，保存到目录: {ckpt_dir}")
    print(f"索引文件已创建: {index_file}")
    print(f"元数据文件已创建: {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description='将PyTorch模型转换为MindSpore格式（小块）')
    parser.add_argument('--input', type=str, default='models/med_flamingo/model.pt',
                        help='输入的PyTorch模型文件路径(.pt)')
    parser.add_argument('--output_dir', type=str, default='models/med_flamingo/ckpt',
                        help='输出的MindSpore参数文件目录')
    parser.add_argument('--max_chunk_size', type=int, default=50*1024*1024, 
                        help='每个参数块的最大大小（字节），默认50MB')
    parser.add_argument('--overwrite', action='store_true',
                        help='是否覆盖已存在的输出目录，默认为False')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    if os.path.exists(args.output_dir):
        if args.overwrite:
            # 尝试清空目录
            try:
                print(f"尝试清空目录: {args.output_dir}")
                for filename in os.listdir(args.output_dir):
                    file_path = os.path.join(args.output_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"警告: 无法删除 {file_path}. 原因: {e}")
            except Exception as e:
                print(f"警告: 无法清空目录 {args.output_dir}. 原因: {e}")
                print("将使用现有目录，但可能会出现文件混合问题")
        else:
            print(f"输出目录 {args.output_dir} 已存在。")
            response = input("是否使用现有目录？可能会导致文件混合 (y/n): ")
            if response.lower() != 'y':
                print("退出程序。请指定新的输出目录或使用--overwrite参数")
                return
            
    else:
        # 目录不存在，创建新目录
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    
    convert_pt_to_ms_chunks(args.input, args.output_dir, args.max_chunk_size)

if __name__ == '__main__':
    main() 