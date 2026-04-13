import argparse
import os
import subprocess
import time
import tempfile
import numpy as np
import sounddevice as sd
from gtts import gTTS


import pygame

DURATION = 60 * 10  # max recording duration of 10 minutes
SAMPLERATE = 16000
CHANNELS = 1
SPEECH_ENABLED = False  # no speech by default


def text_to_speech(text, output_path, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_path)


def play_audio(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():  # wait for audio to stop
        time.sleep(0.1)


# pretty print new section
def print_section(text):
    print("\n|"+("-"*min(150, len(text)-2))+"|")
    print(text)

# speaks to user


def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        text_to_speech(text, tmp.name)
        play_audio(tmp.name)

# records audio from microphone and save to numpy file


def record_and_save(output_path):
    try:
        audio = sd.rec(int(DURATION * SAMPLERATE),
                       samplerate=SAMPLERATE, channels=CHANNELS)

        sd.wait()
        print("Recording stopped after 10 minutes.")
    except KeyboardInterrupt:
        sd.stop()
        print("Recording stopped by user.")
        if SPEECH_ENABLED:
            speak("Recording stopped by user.")

    np.save(output_path, audio)


# return (abs file path of host, container file path)
def get_container_path(filename):
    abs_output = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)
    container_output = f"/app/{os.path.relpath(abs_output, os.getcwd())}"
    return abs_output, container_output


# asks container to load numpy file and convert to text
def stt_container(audio_path):
    with tempfile.NamedTemporaryFile(suffix=".txt", dir=".") as tmp:
        output_path = tmp.name
        _, container_audio = get_container_path(audio_path)
        _, container_output = get_container_path(output_path)
        subprocess.run([
            "docker", "compose", "exec", "-T", "app",
            "python", "-c",
            f"from interface.io import audio_to_text; audio_to_text('{container_audio}', '{container_output}')"
        ], check=True, stdout=subprocess.DEVNULL)  # hide standard output

        with open(output_path, 'r') as f:
            text = f.read().strip()
    return text


def speech_to_text():
    with tempfile.NamedTemporaryFile(suffix=".npy", dir=".") as tmp:
        audio_path = tmp.name
        record_and_save(audio_path)
        print("Recording saved. Transcribing...")
        text = stt_container(audio_path)
    return text


def get_response_container(text):
    with tempfile.NamedTemporaryFile(suffix=".txt", dir=".") as tmp:
        output_path = tmp.name
        _, container_output = get_container_path(output_path)
        subprocess.run([
            "docker", "compose", "exec", "-T", "app",
            "python", "-c",
            f"from interface.io import generate_response; generate_response('{text}','{container_output}')"
        ], check=True, stdout=subprocess.DEVNULL)  # hide standard output

        with open(output_path, 'r') as f:
            response = f.read().strip()
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="store_true",
                        help="Disable speech and only use text input/output")
    parser.add_argument("--output", default="tmp/tts_output.mp3")
    args = parser.parse_args()

    if args.text:
        SPEECH_ENABLED = False
    else:
        SPEECH_ENABLED = True

    print_section(
        "Welcome to your Interactive Diary! You can speak or type your diary entries, and I'll respond with empathy and understanding.")
    if SPEECH_ENABLED:
        speak("Welcome to your Interactive Diary!")

    # first round
    log_text = f"Hey, what's up? Feel free to {'speak' if SPEECH_ENABLED else 'type'} your diary entry."
    print_section(log_text)
    if SPEECH_ENABLED:
        speak(log_text)
        print("Listening for your diary entry... Stop recording with Ctrl+C when done.")
        text = speech_to_text()
    else:
        text = input("Your diary entry: ")
    print("Your Response" + text)

    print("Generating response...")
    response = get_response_container(text)
    print_section(response)
    speak(response)


if __name__ == "__main__":
    main()
