# 假设的数据集类
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm

# 定义 emotion 到索引的映射
EMOTION_TO_INDEX = {
    'joy': 0, 'trust': 1, 'fear': 2, 'surprise': 3,
    'sadness': 4, 'disgust': 5, 'anger': 6, 'anticipation': 7
}

class V2MDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = self.load_all_data()

    def load_all_data(self):
        data = []
        video_names = [f.split('_features')[0] for f in os.listdir(self.config['visual_anno_path']) if f.endswith('_features.npy')]
        for video_name in tqdm(video_names, desc="Loading data"):
            # 检查视频名称是否大于700
            if int(video_name) > 700:
                continue  # 跳过大于700的视频，留作测试集
            fixed_token = np.load(os.path.join(self.config['visual_anno_path'], f"{video_name}_features.npy"), allow_pickle=True)
            
            motion_file = os.path.join(self.config['motion_anno_path'], f"{video_name}_motion.json")
            if not os.path.exists(motion_file):
                print(f"Motion file not found: {motion_file}")
                continue
            with open(motion_file, 'r') as f:
                motion_data = json.load(f)
                motion_tokens = [scene['motion'] for scene in motion_data]
            
            emotion_file = os.path.join(self.config['emotion_anno_path'], f"{video_name}.mp4.json")
            if not os.path.exists(emotion_file):
                print(f"Emotion file not found: {emotion_file}")
                continue
            with open(emotion_file, 'r') as f:
                emotion_data = json.load(f)
                emotion_tokens = []
                for scene in emotion_data['sep_scene']:
                    filtered_emotions = {k: v for k, v in scene['emotion'].items() if k in EMOTION_TO_INDEX}
                    top_emotion = max(filtered_emotions.items(), key=lambda x: x[1])[0]
                    emotion_tokens.append(EMOTION_TO_INDEX[top_emotion])
            
            variable_token = np.load(os.path.join(self.config['caption_feature_anno_path'], f"{video_name}.npy"), allow_pickle=True)
            target = np.load(os.path.join(self.config['target_path'], f"{video_name}.npy"), allow_pickle=True)

            assert len(fixed_token) == len(motion_tokens) == len(emotion_tokens) == len(variable_token) == len(target), \
                f"Inconsistent scene numbers for video {video_name}"

            for i in range(len(fixed_token)):
                if motion_tokens[i] != 0 and len(fixed_token[i]) == 10:
                    data.append({
                        'fixed_token': fixed_token[i],
                        'two_numbers': [motion_tokens[i], emotion_tokens[i]],
                        'variable_token': variable_token[i],
                        'target': target[i]
                    })
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['fixed_token'], dtype=torch.float),
            torch.tensor(item['two_numbers'], dtype=torch.float),
            torch.tensor(item['variable_token'], dtype=torch.float),
            torch.tensor(item['target'], dtype=torch.float)
        )

def collate_fn(batch):
    fixed_tokens, two_numbers, variable_tokens, targets = zip(*batch)

    # 将列表转换为张量
    fixed_tokens = torch.stack(fixed_tokens)
    two_numbers = torch.stack(two_numbers)
    
    # 对 variable_tokens 进行填充
    max_variable_length = max(len(vt) for vt in variable_tokens)
    padded_variable_tokens = [torch.nn.functional.pad(vt, (0, 0, 0, max_variable_length - len(vt))) for vt in variable_tokens]
    variable_tokens = torch.stack(padded_variable_tokens)
    
    # 添加 EOS 到目标序列，然后进行填充
    eos_embedding = torch.ones(targets[0].shape[-1])  # EOS 是全1向量
    targets_with_eos = [torch.cat([t.squeeze(0), eos_embedding.unsqueeze(0)], dim=0) for t in targets]
    max_target_length = max(t.size(0) for t in targets_with_eos)
    padded_targets = [torch.nn.functional.pad(t, (0, 0, 0, max_target_length - t.size(0))) for t in targets_with_eos]
    targets_with_eos_padded = torch.stack(padded_targets)

    # 创建输入掩码和目标掩码
    input_masks, target_masks = create_mask(fixed_tokens, two_numbers, variable_tokens, targets_with_eos_padded)

    return fixed_tokens, two_numbers, variable_tokens, targets_with_eos_padded, input_masks, target_masks

def create_mask(fixed_tokens, two_numbers, variable_tokens, targets):
    batch_size = fixed_tokens.size(0)
    fixed_length = fixed_tokens.size(1)
    variable_length = variable_tokens.size(1)
    total_input_length = fixed_length + 1 + variable_length  # 1 是为 two_numbers 预留的位置
    target_length = targets.size(1)

    # 创建输入掩码
    input_mask = torch.ones(batch_size, total_input_length, dtype=torch.bool)

    # 创建目标掩码
    target_mask = torch.zeros(batch_size, target_length, dtype=torch.bool)

    # 找到每个序列的实际长度（非填充的部分）
    variable_lengths = (variable_tokens.abs().sum(dim=-1) != 0).sum(dim=1)
    target_lengths = (targets.abs().sum(dim=-1) != 0).sum(dim=1)

    # 更新掩码以反映实际长度
    for i in range(batch_size):
        input_mask[i, fixed_length + 1 + variable_lengths[i]:] = False
        target_mask[i, :target_lengths[i]] = True  # 包括 EOS token

    return input_mask, target_mask

def load_data(config):
    dataset = V2MDataset(config)
    print("dataset created")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    print("dataloader created")
    return dataloader
