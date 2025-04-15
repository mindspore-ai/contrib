import os
import json
from concurrent import futures
from pathlib import Path
import pickle
import argparse
import numpy as np

def options():
    parser = argparse.ArgumentParser(description='STCrowd converting ...')
    parser.add_argument('--path_root',type=str,default='STCrowd/STCrowd/STCrowd_official/')
    parser.add_argument('--split_file',type=str,default='split.json')   # the split file 
    parser.add_argument('--split',type=str,default='train')             # train / val 
    args = parser.parse_args()
    return args

def process_single_scene(load_dict, idx, data_root):
    """处理单个场景帧"""
    try:
        frame_info = load_dict['frames'][idx]
        info = {
            'image': {
                'image_idx': idx,
                'image_path': str(data_root / frame_info['images'][0]['image_name'].lstrip('/'))
            },
            'point_cloud': {
                'path': str(data_root / frame_info['frame_name'].lstrip('/')),
                'num_features': 4
            },
            'annos': {
                'position': [],
                'dimensions': [],
                'occlusion': [],
                'rotation': [],
                'tracking': {
                    'velocity': [],
                    'tracking_id': []
                }
            }
        }

        # 处理目标物体信息
        valid_items = [
            item for item in frame_info['items']
            if item['boundingbox']['z'] >= 1.2  # 过滤坐姿目标
        ]

        # 处理跟踪信息
        prev_frame_items = []
        if idx > 0:
            prev_frame_items = load_dict['frames'][idx-1]['items']

        for item in valid_items:
            # 位置和尺寸
            info['annos']['position'].append([
                item['position']['x'],
                item['position']['y'],
                item['position']['z']
            ])
            info['annos']['dimensions'].append([
                item['boundingbox']['x'],
                item['boundingbox']['y'],
                item['boundingbox']['z']
            ])
            
            # 运动信息
            velocity = [0.0, 0.0]
            prev_item = next((i for i in prev_frame_items if i['id'] == item['id']), None)
            if prev_item:
                velocity = [
                    prev_item['position']['x'] - item['position']['x'],
                    prev_item['position']['y'] - item['position']['y']
                ]
            info['annos']['tracking']['velocity'].append(velocity)
            info['annos']['tracking']['tracking_id'].append(item['id'])

            # 其他属性
            info['annos']['occlusion'].append(item['occlusion'])
            info['annos']['rotation'].append(item['rotation'])

        return info
    except Exception as e:
        print(f"Error processing frame {idx}: {str(e)}")
        return None

def load_group(data_root, group_file):
    """加载单个group数据"""
    try:
        with open(group_file, 'r') as f:
            load_dict = json.load(f)
            
        with futures.ThreadPoolExecutor() as executor:
            tasks = [
                executor.submit(process_single_scene, load_dict, idx, data_root)
                for idx in range(load_dict['total_number'])
            ]
            return [task.result() for task in futures.as_completed(tasks) if task.result()]
            
    except Exception as e:
        print(f"Error loading {group_file}: {str(e)}")
        return []

def create_dataset_info(data_root, split_list):
    """创建数据集信息文件"""
    anno_dir = data_root / 'anno'
    all_infos = []
    
    for group_id in split_list:
        group_file = anno_dir / f"{group_id}.json"
        if not group_file.exists():
            print(f"Warning: Missing group file {group_file}")
            continue
            
        group_infos = load_group(data_root, group_file)
        all_infos.extend(group_infos)
        
    return all_infos

def main():
    # 解析参数
    args = options()
    data_root = Path(args.path_root)
    
    # 加载split配置
    split_file = data_root / args.split_file
    try:
        with open(split_file, 'r') as f:
            split_config = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Split file {split_file} not found")
        
    # 生成数据集信息
    dataset_info = create_dataset_info(data_root, split_config[args.split])
    
    # 保存结果
    output_file = data_root / f"STCrowd_infos_{args.split}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(dataset_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Successfully saved {len(dataset_info)} entries to {output_file}")

if __name__ == "__main__":
    main()