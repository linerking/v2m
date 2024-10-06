import os
import json
import torch
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import logging

def extract_visual_features(video_dir, scene_dir, output_dir, log_path, max_frames=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(filename=log_path, level=logging.ERROR, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    for video_file in tqdm(os.listdir(video_dir)):
        try:
            if not video_file.endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(video_dir, video_file)
            scene_file = os.path.splitext(video_file)[0] + '.txt'
            scene_path = os.path.join(scene_dir, scene_file)

            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"未找到对应的场景文件 {scene_file}")

            with open(scene_path, 'r') as f:
                scenes = [list(map(float, line.strip().split())) for line in f]
            
            all_scene_features = []
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for start_time, end_time in scenes:
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                total_frames = end_frame - start_frame + 1
                if total_frames < max_frames:
                    frame_numbers = range(start_frame, end_frame + 1)
                else:
                    frame_numbers = np.linspace(start_frame, end_frame, max_frames, dtype=int)
                
                scene_images = []
                
                for frame_number in frame_numbers:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError(f"无法读取帧，视频：{video_file}，时间：{frame_number / fps}")
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb).resize((224, 224), Image.LANCZOS)
                    scene_images.append(image)
                
                if not scene_images:
                    raise ValueError(f"场景没有有效帧，视频：{video_file}，开始时间：{start_time}，结束时间：{end_time}")
                
                # 处理场景的所有帧
                inputs = processor(images=scene_images, return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                pooled_output = outputs.pooler_output.cpu().numpy()
                all_scene_features.append(pooled_output)
            
            cap.release()
            
            if all_scene_features:
                output_file = os.path.splitext(video_file)[0] + '_features.npy'
                np.save(os.path.join(output_dir, output_file), np.array(all_scene_features))
                print(f"已提取并保存 {len(all_scene_features)} 个场景的视觉特征，文件名：{output_file}")
            else:
                raise ValueError(f"视频 {video_file} 没有提取到任何有效特征")

        except Exception as e:
            logging.error(f"处理视频 {video_file} 时出错: {str(e)}")
            print(f"处理视频 {video_file} 时出错，已记录日志。继续处理下一个视频。")
            continue

# 从config.json中获取路径
config = json.load(open('config.json', 'r'))
video_dir = config['raw_video_path']
scene_dir = config['scene_anno_path']
output_dir = config['visual_anno_path']
log_path = config['visual_anno_log_path']

# 调用函数处理所有视频
extract_visual_features(video_dir, scene_dir, output_dir, log_path, max_frames=10)
