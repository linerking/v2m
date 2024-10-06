#!/home/yihan/anaconda3/envs/v2m/bin/python

import os
import json
import logging
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

def extract_audio_from_video(video_path, logger):
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    try:
        result = os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}"')
        if result != 0:
            logger.error(f"Failed to extract audio from {video_path}")
            return None
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {e}", exc_info=True)
        return None

def detect_key_changes(audio_path, logger):
    try:
        # 使用 madmom 的 CNNKeyRecognitionProcessor 进行key检测
        proc = CNNKeyRecognitionProcessor()
        act = proc(audio_path)
        key_label = key_prediction_to_label(act)

        # 打印 key_label 的内容以调试
        logger.debug(f"Detected key label: {key_label}")
        
        return key_label
    except Exception as e:
        logger.error(f"Error detecting key changes in {audio_path}: {e}", exc_info=True)
        return None

def save_key_changes_to_txt(key_label, output_path, logger):
    try:
        with open(output_path, 'w') as f:
            f.write(f"{key_label}\n")
    except Exception as e:
        logger.error(f"Error saving key changes to {output_path}: {e}", exc_info=True)

def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        raw_video_path = config["raw_video_path"]
        key_annotation_path = config["key_annotation_path"]
        key_anno_log_path = config["key_anno_log_path"]

        # 设置日志配置
        logging.basicConfig(
            filename=key_anno_log_path,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger()

        os.makedirs(key_annotation_path, exist_ok=True)

        for filename in os.listdir(raw_video_path):
            if filename.lower().endswith('.mp4'):
                video_path = os.path.join(raw_video_path, filename)
                audio_path = extract_audio_from_video(video_path, logger)
                if not audio_path:
                    continue
                key_label = detect_key_changes(audio_path, logger)
                if key_label is None:
                    continue
                output_filename = os.path.splitext(filename)[0] + '.txt'
                output_path = os.path.join(key_annotation_path, output_filename)
                save_key_changes_to_txt(key_label, output_path, logger)
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.error(f"Error removing audio file {audio_path}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)

if __name__ == "__main__":
    main()