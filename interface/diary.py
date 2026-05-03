from rich.panel import Panel
from rich import print
import io
import queue
from gtts import gTTS
import sounddevice as sd
import numpy as np
import json
import tempfile
import time
import subprocess
import argparse
import os

# isort: off
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame # keep this after os.environ.
# isort: on

DURATION = 60 * 10  # max recording duration of 10 minutes
SAMPLERATE = 16000
CHANNELS = 1
SPEECH_ENABLED = False  # no speech by default


def tts(text, output_path, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_path)


def play_audio(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():  # wait for audio to stop
        time.sleep(0.1)


# pretty print new section
def print_assistant(text):
    print()
    print(Panel(f"{text}", title="Interactive Diary", style="bold green"))
    
def print_diary(text):
    print()
    print(Panel(f"{text}", style="bold blue"))


def speak(text):
    mp3_buffer = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_buffer, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


# records audio from microphone and save to numpy file


def record_and_save(output_path):
    q = queue.Queue()
    recording = []
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            # If the main loop is busy, drop this chunk instead of blocking
            # the audio callback thread.
            pass

    print("Recording... Press Ctrl+C to stop.")
    stream = None
    try:
        stream = sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            callback=callback
        )
        stream.start()
        while True:
            if time.time() - start_time >= DURATION:
                print(f"\nReached max recording duration ({DURATION} seconds).")
                break
            try:
                # Use a timeout so Python can process Ctrl+C promptly.
                data = q.get(timeout=0.1)
                recording.append(data)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("\nStopped recording.")
    finally:
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    # Keep any final chunks that were queued right before Ctrl+C.
    while not q.empty():
        recording.append(q.get_nowait())

    if recording:
        audio = np.concatenate(recording, axis=0)
    else:
        audio = np.empty((0, CHANNELS), dtype=np.float32)

    np.save(output_path, audio)
    # play_recorded_audio(output_path)


def play_recorded_audio(audio_path):
    audio = np.load(audio_path).astype(np.float32)
    sd.play(audio, samplerate=SAMPLERATE)
    sd.wait()

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
            f"from interface.io_audio import audio_to_text; audio_to_text('{container_audio}', '{container_output}')"
        ], check=True, stdout=subprocess.DEVNULL)  # hide standard output

        with open(output_path, 'r') as f:
            text = f.read().strip()
    return text


def stt():
    with tempfile.NamedTemporaryFile(suffix=".npy", dir=".") as tmp:
        audio_path = tmp.name
        record_and_save(audio_path)
        print("Recording saved. Transcribing...")
        text = stt_container(audio_path)
    return text


def gpt_diary_response(text):
    with tempfile.NamedTemporaryFile(suffix=".txt", dir=".") as tmp:
        output_path = tmp.name
        _, container_output = get_container_path(output_path)
        subprocess.run([
            "docker", "compose", "exec", "-T", "app",
            "python", "-c",
            f"from interface.io_audio import generate_gpt_response; generate_gpt_response({json.dumps(text)}, {json.dumps(container_output)})"
        ], check=True, stdout=subprocess.DEVNULL)  # hide standard output

        with open(output_path, 'r') as f:
            response = f.read().strip()
    return response


def start_diary(speech_enabled=True):
    SPEECH_ENABLED = speech_enabled
    print_assistant(
        "Welcome to Interactive Diary!")
    if speech_enabled:
        speak("Welcome to Interactive Diary!")
        
    print_assistant(
        "Would you like me to coach, or just listen?"
    )
    if speech_enabled:
        speak("Would you like me to coach, or just listen?")
    
    diary_style = input(
        "[COACH/listen]: ").strip().lower()
    listen = (diary_style in ("listen", "just listen", "listening", "just listening", "just be listening") ) or (diary_style[0] == "l")
    diary_style = "listen" if listen else "COACH"
    print_diary(f"You chose: {diary_style} mode.")
    
    return listen


def get_entry(speech_enabled=True, first_entry=False):
    SPEECH_ENABLED = speech_enabled
    log_text = f"Hey, what's up? Feel free to {'speak' if SPEECH_ENABLED else 'type'} your diary entry."
    print_assistant(log_text)
    if SPEECH_ENABLED:
        speak(log_text)
        print("Listening for your diary entry... Stop recording with Ctrl+C when done.")
        text = stt()
    else:
        text = input("Your diary entry: ")
    print_diary(f"Your diary entry: {text}")

    # text formatting?
    return text

def diary_response(
    raw_text: str,
    user_id: str,
    session_id: str,
    top_k: int,
    listen: bool,
    ablation: bool,
) -> str:
    call = "gpt" if ablation else "rag"
    with tempfile.NamedTemporaryFile(suffix=".txt", dir=".") as tmp:
        output_path = tmp.name
        _, container_output = get_container_path(output_path)
        subprocess.run([
            "docker", "compose", "exec", "-T", "app",
            "python", "-c",
            (
                f"from interface.io_rag import generate_{call}_response; "
                f"generate_{call}_response({json.dumps(raw_text)}, "
                f"{json.dumps(user_id)}, "
                f"{json.dumps(session_id)}, "
                f"{int(top_k)}, "
                f"{listen}, "
                f"{json.dumps(container_output)})"
            ),
        ], check=True, stdout=subprocess.DEVNULL)  # hide standard output

        with open(output_path, "r") as f:
            response = f.read().strip()
    return response

def put_reply(text, speech_enabled=True):
    SPEECH_ENABLED = speech_enabled
    print_assistant(f"Thanks for sharing!\n\n{text}")
    if SPEECH_ENABLED:
        speak("Thanks for sharing! " + text)

def should_continue_diary(speech_enabled=True):
    keep_going = input(
        "\nWould you like to write another entry? [Y/n]: ").strip().lower()
    y = keep_going not in ("n", "no", "stop", "quit", "exit", "q")
    if not y:
        print_assistant("Bye-bye! Please fill out the survey. :)")
        if speech_enabled:
            speak("Bye-bye! Please fill out the survey!")
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="store_true",
                        help="Disable speech and only use text input/output")
    args = parser.parse_args()
    SPEECH_ENABLED = not args.text

    start_diary()
    get_entry()
    put_reply("This is a response to your diary entry!")


if __name__ == "__main__":
    main()
