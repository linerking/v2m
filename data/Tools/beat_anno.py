#!/home/yihan/anaconda3/envs/v2m/bin/python

import os
import json
import numpy as np
import essentia.standard as es
import logging

def extract_audio_from_video(video_path):
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    try:
        os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}"')
        return audio_path
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {e}")
        return None

def detect_beats(audio_path):
    try:
        loader = es.MonoLoader(filename=audio_path)
        audio = loader()
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)
        print(f"Detected tempo: {bpm} BPM")
        
        return beats  # 只返回beats，不再返回downbeats
    except Exception as e:
        logging.error(f"Error detecting beats in {audio_path}: {e}")
        return []

def save_beats_to_txt(beats, output_path):
    try:
        with open(output_path, 'w') as f:
            f.write("# Beats\n")
            for beat in beats:
                f.write(f"{beat:.2f}\n")
    except Exception as e:
        logging.error(f"Error saving beats to {output_path}: {e}")

def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        raw_video_path = config["raw_video_path"]
        beat_annotation_path = config["beat_annotation_path"]
        beat_anno_log_path = config["beat_anno_log_path"]
        
        logging.basicConfig(
            filename=beat_anno_log_path,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        os.makedirs(beat_annotation_path, exist_ok=True)
        
        for filename in os.listdir(raw_video_path):
            if filename.lower().endswith('.mp4'):
                video_path = os.path.join(raw_video_path, filename)
                audio_path = extract_audio_from_video(video_path)
                if not audio_path:
                    continue
                beats = detect_beats(audio_path)
                output_filename = os.path.splitext(filename)[0] + '.txt'
                output_path = os.path.join(beat_annotation_path, output_filename)
                save_beats_to_txt(beats, output_path)
                os.remove(audio_path)
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()