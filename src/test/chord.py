import madmom
import logging
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
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
if __name__ == "__main__":
    logger = logging.getLogger(__name__)  
    audio_path = "/home/yihan/v2m/1.wav"
    chords = detect_chords(audio_path)
    key_changes = detect_key_changes(audio_path, logger)
    print(chords)
    print(key_changes)