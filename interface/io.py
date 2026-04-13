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

    result = model.transcribe(audio_mono, fp16=False,
                              condition_on_previous_text=False)
    with open(output_path, 'w') as f:
        f.write(result["text"])


def generate_response(diary_text, output_path):
    # insert some call to model
    response = "This is a response to your diary entry!"
    with open(output_path, 'w') as f:
        f.write(response)


def save_text_to_speech(text, output_path, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="hello")
    parser.add_argument("--output", default="/app/tts_output.mp3")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    save_text_to_speech(args.text, args.output, args.lang)
    print(f"Saved speech to {args.output}")


if __name__ == "__main__":
    main()
