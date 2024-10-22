import json
import os
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
import torch
from tqdm import tqdm

# 读取配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

emotion_anno_path = config['emotion_anno_path']
output_path = config['caption_feature_anno_path']

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 初始化 T5 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5EncoderModel.from_pretrained('t5-base').to(device)

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取所有非填充 token 的 embeddings
    embeddings = outputs.last_hidden_state[0]  # 移除批次维度
    mask = inputs['attention_mask'][0]  # 移除批次维度
    valid_embeddings = embeddings[mask.bool()]
    return valid_embeddings.cpu().numpy()

# 遍历目录中的所有 JSON 文件
for filename in tqdm(os.listdir(emotion_anno_path)):
    if filename.endswith('.json'):
        file_path = os.path.join(emotion_anno_path, filename)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 获取所有场景的 content
        contents = [scene['content'] for scene in data['sep_scene']]
        
        # 将每个 content 转换为 embedding 列表
        all_embeddings = [get_embeddings(content) for content in contents]
        
        # 生成输出文件名（使用数字命名）
        output_filename = f"{os.path.splitext(filename)[0].split('.')[0]}.npy"
        output_file_path = os.path.join(output_path, output_filename)
        
        # 保存为 .npy 文件
        np.save(output_file_path, np.array(all_embeddings, dtype=object), allow_pickle=True)

print("处理完成。所有 embedding 文件已保存到", output_path)
