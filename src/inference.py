import torch
from v2m import V2MTransformer  # 确保这个导入与您的模型定义文件相匹配
from tqdm import tqdm
import numpy as np
import json
import os
import random
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
EMOTION_TO_INDEX = {
    'joy': 0, 'trust': 1, 'fear': 2, 'surprise': 3,
    'sadness': 4, 'disgust': 5, 'anger': 6, 'anticipation': 7
}
def load_and_select_scene(config):
    video_names = [f.split('_features')[0] for f in os.listdir(config['visual_anno_path']) if f.endswith('_features.npy')]
    valid_scenes = []

    for video_name in tqdm(video_names, desc="Loading data"):
        # 检查视频名称是否大于700
        if int(video_name) <= 700:
            continue  # 跳过大于700的视频，留作测试集

        fixed_token = np.load(os.path.join(config['visual_anno_path'], f"{video_name}_features.npy"), allow_pickle=True)
        
        motion_file = os.path.join(config['motion_anno_path'], f"{video_name}_motion.json")
        if not os.path.exists(motion_file):
            print(f"Motion file not found: {motion_file}")
            continue
        with open(motion_file, 'r') as f:
            motion_data = json.load(f)
            motion_tokens = [scene['motion'] for scene in motion_data]
            scene_times = [(scene['start_time'], scene['end_time']) for scene in motion_data]
        
        emotion_file = os.path.join(config['emotion_anno_path'], f"{video_name}.mp4.json")
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
        
        variable_token = np.load(os.path.join(config['caption_feature_anno_path'], f"{video_name}.npy"), allow_pickle=True)

        assert len(fixed_token) == len(motion_tokens) == len(emotion_tokens) == len(variable_token) == len(scene_times), \
            f"Inconsistent scene numbers for video {video_name}"
        
        # 找出符合条件的场景索引
        for i in range(len(motion_tokens)):
            if motion_tokens[i] != 0 and len(fixed_token[i]) == 10:
                valid_scenes.append({
                    'video_name': video_name,
                    'scene_index': i,
                    'fixed_token': fixed_token[i],
                    'motion_token': motion_tokens[i],
                    'emotion_token': emotion_tokens[i],
                    'variable_token': variable_token[i],
                    'start_time': scene_times[i][0],
                    'end_time': scene_times[i][1]
                })

    # 从所有符合条件的场景中随机选择一个
    if valid_scenes:
        selected_scene = random.choice(valid_scenes)
        return selected_scene
    else:
        return None
        
    
def load_model(model_path, device):
    # 初始化模型（确保参数与训练时一致）
    model = V2MTransformer(input_dim=768, output_dim=1536, nhead=4, num_decoder_layers=4, dim_feedforward=512, dropout=0.1)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model

def prepare_input(fixed_tokens, two_numbers, variable_tokens, device):
    # 将输入数据转换为张量并移动到指定设备
    fixed_tokens = torch.tensor(fixed_tokens, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
    two_numbers = torch.tensor(two_numbers, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
    variable_tokens = torch.tensor(variable_tokens, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
    
    # 创建输入掩码
    total_length = fixed_tokens.size(1) + 1 + variable_tokens.size(1)
    input_mask = torch.ones(1, total_length, dtype=torch.bool).to(device)
    
    return fixed_tokens, two_numbers, variable_tokens, input_mask

def inference(model, fixed_tokens, two_numbers, variable_tokens, device):
    # 准备输入数据
    fixed_tokens, two_numbers, variable_tokens, input_mask = prepare_input(fixed_tokens, two_numbers, variable_tokens, device)
    
    # 执行推理
    with torch.no_grad():
        output = model(fixed_tokens, two_numbers, variable_tokens, input_mask)
    
    return output

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model_path = "/home/yihan/v2m/best_model.pth"  # 替换为您的模型路径
    model = load_model(model_path, device)
    with open('config.json', 'r') as f:
        config = json.load(f)
    # 准备输入数据（这里只是示例，请根据实际情况修改）
    selected_scene = load_and_select_scene(config)
    fixed_tokens = selected_scene['fixed_token']
    two_numbers = [selected_scene['motion_token'], selected_scene['emotion_token']]
    variable_tokens = selected_scene['variable_token']
    video_name = selected_scene['video_name']
    start_time = selected_scene['start_time']
    end_time = selected_scene['end_time']
    print(f"Selected scene: {video_name}, start_time: {start_time}, end_time: {end_time}")
    print(fixed_tokens.shape, variable_tokens.shape)
    
    # 执行推理
    output = inference(model, fixed_tokens, two_numbers, variable_tokens, device)
    mask = torch.ones((output.shape[0], output.shape[1]), dtype=torch.float32, device=device)
    
    cfg_condition = {
        'description': [
            output,  # 保持为 tensor
            mask     # 保持为 tensor
        ]
    }
    # 处理输出
    print("Output shape:", output.shape)
    # 这里可以添加更多的后处理步骤，比如将输出转换回原始格式等
    from v2m_models.audiocraft.models.musicgen import MusicGen
    from v2m_models.audiocraft.data.audio import audio_write
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=10,cfg_coef=7,two_step_cfg=True)
    audio = model.generate(descriptions=["any thing"], cfg_conditions=cfg_condition)
    for idx, one_wav in enumerate(audio):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{video_name}_{start_time}_{end_time}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
if __name__ == "__main__":
    main()
