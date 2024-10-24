import os
import json
from collections import Counter
import pickle

def build_emotion_mapping(emotion_tokens_path):
    emotion_counter = Counter()
    
    for file_name in os.listdir(emotion_tokens_path):
        if file_name.endswith('.mp4.json'):
            with open(os.path.join(emotion_tokens_path, file_name), 'r') as f:
                emotion_data = json.load(f)
                for scene in emotion_data['sep_scene']:
                    # 排除 'positive' 和 'negative'
                    filtered_emotions = {k: v for k, v in scene['emotion'].items() if k not in ['positive', 'negative']}
                    emotion_counter.update(filtered_emotions.keys())
    
    # 构建情绪到索引的映射
    emotion_to_index = {emotion: index for index, emotion in enumerate(sorted(emotion_counter.keys()))}
    return emotion_to_index

def main():
    # 从 config 中读取数据路径
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    emotion_tokens_path = config['emotion_anno_path']
    
    # 构建情绪映射
    emotion_to_index = build_emotion_mapping(emotion_tokens_path)
    print("Emotion to index mapping:")
    for emotion, index in emotion_to_index.items():
        print(f"{emotion}: {index}")
    
    # 保存情绪映射到文件
    # with open('emotion_mapping.pkl', 'wb') as f:
    #     pickle.dump(emotion_to_index, f)
    
    print("\nEmotion mapping has been saved to 'emotion_mapping.pkl'")
    print(f"Total number of unique emotions: {len(emotion_to_index)}")

if __name__ == "__main__":
    main()

