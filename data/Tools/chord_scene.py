import json
import os
from collections import defaultdict

def read_scene_file(file_path):
    scenes = []
    with open(file_path, 'r') as f:
        for line in f:
            start, end = map(float, line.strip().split())
            scenes.append((start, end))
    return scenes

def read_chord_file(file_path):
    chords = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, chord = parts
                chords.append((float(start), float(end), chord))
    return chords

def annotate_scene_chords(scene_file, chord_file):
    scenes = read_scene_file(scene_file)
    chords = read_chord_file(chord_file)
    
    scene_chords = defaultdict(set)
    for i, (scene_start, scene_end) in enumerate(scenes):
        for chord_start, chord_end, chord in chords:
            if (chord_start < scene_end and chord_end > scene_start) or \
               (scene_start <= chord_start < scene_end):
                scene_chords[i].add(chord)
    
    return scene_chords

# 读取配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

# 遍历场景标注文件并处理
scene_anno_path = config['scene_anno_path']
chord_anno_path = config['chord_annotation_path']
output_dir = config['chord_scene_anno_path']
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(scene_anno_path):
    if filename.endswith('.txt'):
        file_number = filename.split('.')[0]
        scene_file = os.path.join(scene_anno_path, filename)
        chord_file = os.path.join(chord_anno_path, filename)
        
        if os.path.exists(chord_file):
            scene_chords = annotate_scene_chords(scene_file, chord_file)
            
            # 将结果写入文件
            output_file = os.path.join(output_dir, f'{file_number}.txt')
            with open(output_file, 'w') as f:
                for scene, chords in scene_chords.items():
                    f.write(f"{', '.join(sorted(chords))}\n")
            
            print(f"文件 {filename} 的结果已保存到 {output_file}")
        else:
            print(f"警告：未找到对应的和弦文件 {chord_file}")

print("所有场景和弦标注完成")