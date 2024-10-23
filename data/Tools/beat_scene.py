import json
from collections import defaultdict

def read_scene_file(file_path):
    scenes = []
    with open(file_path, 'r') as f:
        for line in f:
            start, end = map(float, line.strip().split())
            scenes.append((start, end))
    return scenes

def read_beat_file(file_path):
    beats = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            beats.append(float(line.strip()))
    return beats

def calculate_bpm(beats):
    if len(beats) < 2:
        return 0
    intervals = [beats[i+1] - beats[i] for i in range(len(beats)-1)]
    average_interval = sum(intervals) / len(intervals)
    bpm = 60 / average_interval
    return round(bpm, 0)

def calculate_scene_bpms(scene_file, beat_file):
    scenes = read_scene_file(scene_file)
    beats = read_beat_file(beat_file)
    
    scene_beats = defaultdict(list)
    for i, (start, end) in enumerate(scenes):
        scene_beats[i] = [b for b in beats if start <= b < end]
    
    scene_bpms = {}
    for scene, scene_beats in scene_beats.items():
        bpm = calculate_bpm(scene_beats)
        scene_bpms[scene] = bpm
    
    return scene_bpms

# 读取配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

import os

scene_anno_path = config['scene_anno_path']
beat_anno_path = config['beat_annotation_path']
bpm_anno_path = config['bpm_anno_path']

# 确保 bpm_anno_path 目录存在
os.makedirs(bpm_anno_path, exist_ok=True)

# 遍历 scene_anno_path 中的所有文件
for filename in os.listdir(scene_anno_path):
    if filename.endswith('.txt'):
        scene_file = os.path.join(scene_anno_path, filename)
        beat_file = os.path.join(beat_anno_path, filename)
        
        # 计算 BPM
        scene_bpms = calculate_scene_bpms(scene_file, beat_file)
        
        # 将结果保存到 bpm_anno_path
        bpm_file = os.path.join(bpm_anno_path, filename)
        with open(bpm_file, 'w') as f:
            for scene, bpm in scene_bpms.items():
                f.write(f"{bpm}\n")
        
        print(f"已处理文件 {filename}")

print("所有文件处理完成")