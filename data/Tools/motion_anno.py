import numpy as np
import cv2
import os
import json
from pathlib import Path
import torch
import torchvision.models.optical_flow as optical_flow
import time
import gc
import logging
from tqdm import tqdm

def setup_logging(log_file):
    # 确保日志文件的父目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_optical_flow(video_path, start_frame, end_frame, sample_rate=1, batch_size=15):
    model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT)
    model = model.cuda().eval()

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_flow = 0.0
    total_pairs = 0
    frames = []
    frame_count = 0

    try:
        for frame_num in range(start_frame, end_frame, sample_rate):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))
            frames.append(frame)
            frame_count += 1

            if len(frames) == batch_size + 1:
                batch = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float().cuda() / 255.0
                with torch.no_grad():
                    flows = model(batch[:-1], batch[1:])[-1]
                
                flow_magnitudes = torch.norm(flows, dim=1)
                batch_avg_flow = flow_magnitudes.mean().item()
                total_flow += batch_avg_flow * batch_size
                total_pairs += batch_size

                frames = [frames[-1]]  # 保留最后一帧用于下一个batch
                torch.cuda.empty_cache()  # 清理GPU内存

        # 处理剩余的帧
        if len(frames) > 1:
            batch = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float().cuda() / 255.0
            with torch.no_grad():
                flows = model(batch[:-1], batch[1:])[-1]
            
            flow_magnitudes = torch.norm(flows, dim=1)
            batch_avg_flow = flow_magnitudes.mean().item()
            total_flow += batch_avg_flow * (len(frames) - 1)
            total_pairs += len(frames) - 1

    except Exception as e:
        logging.error(f"Error processing frames: {str(e)}")
    finally:
        cap.release()
        torch.cuda.empty_cache()

    if total_pairs == 0:
        if frame_count < 2:
            logging.warning(f"Scene in {video_path} has only {frame_count} frame(s). Setting motion value to 0.")
            return 0.0  # 或者其他表示无运动的值
        else:
            raise ValueError(f"No valid frame pairs were processed despite reading {frame_count} frames.")

    average_flow = total_flow / total_pairs
    return average_flow

def process_video(video_path, scene_file, motion_anno_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        motion_annotations = []
        
        with open(scene_file, 'r') as f:
            scenes = f.readlines()
            
            for line in tqdm(scenes, desc=f"Processing scenes in {video_path.stem}"):
                parts = line.strip().split()
                if len(parts) != 2:
                    logging.error(f"Error: Line in {scene_file} is not in the correct format. Skipping video.")
                    return
                start_time, end_time = map(float, parts)
                start_frame = int(start_time * fps)
                end_frame = min(int(end_time * fps), frame_count - 1)
                motion_value = calculate_optical_flow(video_path, start_frame, end_frame)
                motion_annotations.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "motion": float(motion_value)
                })
        
        output_file = motion_anno_path / f'{video_path.stem}_motion.json'
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(motion_annotations, f, indent=4)

    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")
        return

def process_motion_annotations(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    raw_video_path = Path(config['raw_video_path'])
    scene_anno_path = Path(config['scene_anno_path'])
    motion_anno_path = Path(config['motion_anno_path'])
    
    # 确保 motion_anno_path 目录存在
    motion_anno_path.mkdir(parents=True, exist_ok=True)
    
    log_file = motion_anno_path / 'motion_annotation.log'
    setup_logging(log_file)
    
    # 获取所有场景文件
    scene_files = list(scene_anno_path.glob('*.txt'))
    
    # 获取已经标注的视频列表
    annotated_videos = set(file.stem for file in motion_anno_path.glob('*_motion.json'))
    print(len(annotated_videos))
    # 过滤出未标注的视频
    unannotated_scene_files = [file for file in scene_files if file.stem + '_motion' not in annotated_videos]
    
    if len(unannotated_scene_files) == len(scene_files):
        logging.warning("警告：所有视频都未被标注。请检查是否存在问题。")
    print(len(unannotated_scene_files))
    print(len(scene_files))
    
    for scene_file in tqdm(unannotated_scene_files, desc="处理未标注的视频"):
        video_name = scene_file.stem
        video_path = raw_video_path / f"{video_name}.mp4"
        
        if not video_path.exists():
            logging.error(f"错误：未找到视频文件 {video_path}。跳过。")
            continue
        
        process_video(video_path, scene_file, motion_anno_path)
    
    logging.info(f"总视频数：{len(scene_files)}，已标注视频数：{len(annotated_videos)}，未标注视频数：{len(unannotated_scene_files)}")

if __name__ == "__main__":
    config_path = 'config.json'
    start_time = time.time()
    process_motion_annotations(config_path)
    end_time = time.time()
    print(f"总处理时间：{end_time - start_time:.2f} 秒")