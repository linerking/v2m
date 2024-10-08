import sys
import os
sys.path.append('/home/yihan/v2m/v2m_models')
from audiocraft.models.musicgen import MusicGen
from audiocraft.data.audio import audio_write

musicgen_model = MusicGen.get_pretrained("facebook/musicgen-melody")
descriptions = ["a piano with C major and 60 bpm", "a guitar with C minor and 120 bpm"]

# 调用 generate_with_chroma，不提供 melody_wavs
cfg_conditions = musicgen_model.get_cfg_conditions(descriptions)
musicgen_model.set_generation_params(duration=15,cfg_coef=10)
audio = musicgen_model.generate(descriptions,cfg_conditions=cfg_conditions)
for idx, one_wav in enumerate(audio):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), musicgen_model.sample_rate, strategy="loudness", loudness_compressor=True)
print(cfg_conditions)  
