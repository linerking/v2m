import json
from collections import defaultdict
import os
import sys
import numpy as np
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
   
from v2m_models.audiocraft.models.musicgen import MusicGen
def read_chord_scene(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def read_bpm(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f]

def read_emotion(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return [scene['emotion'] for scene in data['sep_scene']]

def read_instrument(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def read_key(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

def get_top_emotions(emotion_dict, top_n=1):
    filtered_emotions = {k: v for k, v in emotion_dict.items() if k not in ['positive', 'negative']}
    sorted_emotions = sorted(filtered_emotions.items(), key=lambda x: x[1], reverse=True)
    return [emotion for emotion, _ in sorted_emotions[:top_n]]
def get_top_instruments(instruments, top_n=3):
    sorted_instruments = sorted(instruments, key=lambda x: x['probability'], reverse=True)
    return [inst['instrument'] for inst in sorted_instruments[:top_n]]

def describe_scene(scene_index, chord, bpm, emotion, instruments, key):
    top_emotions = get_top_emotions(emotion)
    top_instruments = get_top_instruments(instruments)
    
    description = f"Create a song in the key of {key} with a BPM of {bpm}. "
    
    # 处理和弦
    chords = [c.strip() for c in chord.split(',') if c.strip() != 'N']
    if chords:
        description += f"The main chords are {', '.join(chords)}. "
    
    description += f"The dominant instruments are {', '.join(top_instruments)}. "
    description += f"The scene evokes emotions of {', '.join(top_emotions)}."
    
    return description

def main():
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    chord_scene_dir = config['chord_scene_anno_path']
    bpm_dir = config['bpm_anno_path']
    emotion_dir = config['emotion_anno_path']
    instrument_dir = config['instrument_anno_path']
    key_dir = config['key_annotation_path']
    target_dir = config['target_path']

    # 添加这行代码来创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(chord_scene_dir)):
        base_name = os.path.splitext(file_name)[0]
        
        chord_file = os.path.join(chord_scene_dir, file_name)
        bpm_file = os.path.join(bpm_dir, f"{base_name}.txt")
        emotion_file = os.path.join(emotion_dir, f"{base_name}.mp4.json")
        instrument_file = os.path.join(instrument_dir, f"{base_name}.json")
        key_file = os.path.join(key_dir, f"{base_name}.txt")
        
        chords = read_chord_scene(chord_file)
        bpms = read_bpm(bpm_file)
        emotions = read_emotion(emotion_file)
        instruments = read_instrument(instrument_file)
        key = read_key(key_file)
        cfg_conditions = []
        for i in range(len(chords)):
            description = describe_scene(i, chords[i], bpms[i], emotions[i], instruments, key)
            cfg_condition = model.get_cfg_conditions([description])
            cfg_conditions.append(cfg_condition['description'][0])
        #保存
        np.save(os.path.join(target_dir, f"{base_name}.npy"), cfg_conditions)
if __name__ == "__main__":
    main()
