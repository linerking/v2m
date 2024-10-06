import os
import json
import madmom
import numpy as np
import logging

def extract_audio_from_video(video_path):
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    try:
        os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}"')
        return audio_path
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {e}")
        return None

def detect_chords(audio_path):
    try:
        # 使用 madmom 的 CNNChordFeatureProcessor 和 CRFChordRecognitionProcessor 进行和弦检测
        proc = madmom.features.chords.CRFChordRecognitionProcessor()
        act = madmom.features.chords.CNNChordFeatureProcessor()(audio_path)
        chords = proc(act)
        return chords
    except Exception as e:
        logging.error(f"Error detecting chords in {audio_path}: {e}")
        return []

def save_chords_to_txt(chords, output_path):
    try:
        with open(output_path, 'w') as f:
            for chord in chords:
                start_time, end_time, chord_label = chord
                f.write(f"{start_time:.2f}\t{end_time:.2f}\t{chord_label}\n")
    except Exception as e:
        logging.error(f"Error saving chords to {output_path}: {e}")

def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        raw_video_path = config["raw_video_path"]
        chord_annotation_path = config["chord_annotation_path"]
        chord_anno_log_path = config["chord_anno_log_path"]

        logging.basicConfig(
            filename=chord_anno_log_path,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        os.makedirs(chord_annotation_path, exist_ok=True)
        
        for filename in os.listdir(raw_video_path):
            if filename.lower().endswith('.mp4'):
                video_path = os.path.join(raw_video_path, filename)
                audio_path = extract_audio_from_video(video_path)
                if not audio_path:
                    continue
                chords = detect_chords(audio_path)
                output_filename = os.path.splitext(filename)[0] + '.txt'
                output_path = os.path.join(chord_annotation_path, output_filename)
                save_chords_to_txt(chords, output_path)
                os.remove(audio_path)
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()