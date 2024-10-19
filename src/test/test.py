import sys
import os
sys.path.append('/home/yihan/v2m/v2m_models')
from audiocraft.models.musicgen import MusicGen
from audiocraft.data.audio import audio_write

musicgen_model = MusicGen.get_pretrained("facebook/musicgen-melody")
descriptions = ["a piano with the key Bb minor and 60 bpm have chrods C#:maj D#:maj D#:min A#:min one by one", "a guitar with the key Bb minor and 120 bpm have chrods C#:maj D#:maj D#:min A#:min one by one"]

# 调用 generate_with_chroma，不提供 melody_wavs
cfg_conditions = musicgen_model.get_cfg_conditions(descriptions)
print(cfg_conditions['description'][0].shape)
musicgen_model.set_generation_params(duration=15,cfg_coef=10,two_step_cfg=True)
audio = musicgen_model.generate(descriptions,cfg_conditions=cfg_conditions)
for idx, one_wav in enumerate(audio):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), musicgen_model.sample_rate, strategy="loudness", loudness_compressor=True)
print(cfg_conditions)  
