import argparse
import numpy as np
import sounddevice as sd
import whisper
from gtts import gTTS
import os
from openai import OpenAI


SAMPLERATE = 16000
CHANNELS = 1
sd.default.samplerate = SAMPLERATE
sd.default.channels = CHANNELS


def audio_to_text(audio_file, output_path):
    print(os.getcwd())
    audio = np.load(audio_file)
    audio_mono = audio.flatten().astype(np.float32)
    model = whisper.load_model("base", download_root="/app/models")

    result = model.transcribe(audio_mono, fp16=False,
                              condition_on_previous_text=False)
    with open(output_path, 'w') as f:
        f.write(result["text"])


def save_text_to_speech(text, output_path, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_path)

# for GPT wrapper ablation study, not used in main code


def generate_gpt_response(text, output_path):
    client = OpenAI()

    instruction = f"You are a interactive diary assistant, who is grounded, empathetic, and delightful. Given the following diary entry, respond to the writer by reflecting back what you hear with clarity and brevity, then follow up an appropriate empathetic response. For example, if the writer is having a positive time, you can celebrate their wins, or if the writer is having a tough time, you can console them, redirect them to the proper resources, or ask them to clarify their thoughts.\n\nDiary Entry: {text}\n\nResponse:"
    response = client.responses.parse(
        model="gpt-5.4-nano",
        input=instruction
    )
    with open(output_path, 'w') as f:
        f.write(response.output_text)
