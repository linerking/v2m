import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path

class VideoDataset(Dataset):
    def __init__(self, annotations, video_dir):
        self.annotations = annotations
        self.video_dir = video_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        video_path = os.path.join(self.video_dir, anno['video_name'])
        
        # 这里应该添加视频加载和预处理的代码
        # 为简化示例，这里只返回路径和标签
        return {
            'video_path': video_path,
            'label': anno['label'],
            'motion': anno['motion']
        }

def load_annotations(anno_path):
    with open(anno_path, 'r') as f:
        return json.load(f)

def get_data_loader(config_path):
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 获取注释文件路径和视频目录
    anno_path = Path(config['data']['annotation_path'])
    video_dir = Path(config['data']['video_dir'])

    # 加载注释
    annotations = load_annotations(anno_path)

    # 创建数据集
    dataset = VideoDataset(annotations, video_dir)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['training']['shuffle'],
        num_workers=config['training']['num_workers']
    )

    return dataloader

# 使用示例
if __name__ == '__main__':
    config_path = 'path/to/your/config.yaml'
    dataloader = get_data_loader(config_path)
    for batch in dataloader:
        print(batch)
        break  # 只打印第一个批次
