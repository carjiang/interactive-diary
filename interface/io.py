import argparse
import numpy as np
import sounddevice as sd
import whisper
from gtts import gTTS
import os


# result = m.transcribe("tests/test_audio.m4a")


SAMPLERATE = 16000
CHANNELS = 1
sd.default.samplerate = SAMPLERATE
sd.default.channels = CHANNELS


def audio_to_text(audio_file, output_path):
    print(os.getcwd())
    audio = np.load(audio_file)
    audio_mono = audio.flatten().astype(np.float32)
    model = whisper.load_model("base", download_root="/app/models")

    result = model.transcribe(audio_mono, fp16=False, condition_on_previous_text=False)
    with open(output_path, 'w') as f:
        f.write(result["text"])


def save_text_to_speech(text, output_path, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_path)
