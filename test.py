import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/v2m_models/")
from v2m_models.audiocraft.models.musicgen import MusicGen
from v2m_models.audiocraft.data.audio import audio_write
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=10,cfg_coef=15)
audio = model.generate(descriptions=["a viola song with key Bb minor having chord A#:min using a bpm of 120 in a fear emotion"])
for idx, one_wav in enumerate(audio):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{2}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

