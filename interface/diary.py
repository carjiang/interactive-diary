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
    callback_warnings = []
    start_time = time.monotonic()
    chunk_count = 0
    total_frames = 0

    def callback(indata, frames, time_info, status):
        # Keep callback work minimal to avoid audio-thread stalls.
        if status:
            callback_warnings.append(str(status))
        q.put(indata.copy())

    print("Recording... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=1024,
            callback=callback
        ):
            while True:
                if time.monotonic() - start_time >= DURATION:
                    print(f"\nReached max recording duration ({DURATION} seconds).")
                    break

                for _ in range(10):
                    try:
                        chunk = q.get_nowait()
                        recording.append(chunk)
                        chunk_count += 1
                        total_frames += chunk.shape[0]
                    except queue.Empty:
                        break
                # Small sleep avoids a busy-spin while staying responsive.
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopped recording.")
    finally:
        # Keep any final chunks queued right before stopping.
        for _ in range(10):
            try:
                recording.append(q.get_nowait())
            except queue.Empty:
                break

    if callback_warnings:
        unique_warnings = sorted(set(callback_warnings))
        print("Audio input warnings:")
        for warning in unique_warnings:
            print(f"- {warning}")

    recorded_seconds = total_frames / SAMPLERATE if total_frames > 0 else 0.0
    wall_seconds = time.monotonic() - start_time
    # print(
    #     f"Recording diagnostics: chunks={chunk_count}, "
    #     f"frames={total_frames}, "
    #     f"audio_seconds={recorded_seconds:.3f}, "
    #     f"wall_seconds={wall_seconds:.3f}"
    # )

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


# def gpt_diary_response(text):
#     with tempfile.NamedTemporaryFile(suffix=".txt", dir=".") as tmp:
#         output_path = tmp.name
#         _, container_output = get_container_path(output_path)
#         subprocess.run([
#             "docker", "compose", "exec", "-T", "app",
#             "python", "-c",
#             f"from interface.io_audio import generate_gpt_response; generate_gpt_response({json.dumps(text)}, {json.dumps(container_output)})"
#         ],
#             check=True,
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#         )  # hide standard output

#         with open(output_path, 'r') as f:
#             response = f.read().strip()
#     return response


def start_diary(speech_enabled=True):
    SPEECH_ENABLED = speech_enabled
    print_assistant(
        "Welcome to Interactive Diary!")
    if speech_enabled:
        speak("Welcome to Interactive Diary!")
        
    # print_assistant(
    #     "Would you like me to coach, or just listen?"
    # )
    # if speech_enabled:
    #     speak("Would you like me to coach, or just listen?")
    
    # diary_style = input(
    #     "[COACH/listen]: ").strip().lower()
    # listen = (diary_style in ("listen", "just listen", "listening", "just listening", "just be listening") ) or (diary_style[0] == "l")
    # diary_style = "listen" if listen else "COACH"
    # print_diary(f"You chose: {diary_style} mode.")


def get_entry(speech_enabled=True, first_entry=False):
    SPEECH_ENABLED = speech_enabled
    if first_entry:
        print_assistant("Hey, what's up?" )
        if SPEECH_ENABLED:
            speak("Hey, what's up?")
    if SPEECH_ENABLED:
        print("Listening for your diary entry... Stop recording with Ctrl+C when done.")
        text = stt()
    else:
        text = input("Your diary entry: ")
    print_diary(f"Your diary entry: {text}")
    return text

_SERVER_URL = "http://localhost:8765"


def diary_response(
    raw_text: str,
    user_id: str,
    session_id: str,
    top_k: int,
    ablation: bool,
) -> str:
    import urllib.request

    endpoint = "gpt" if ablation else "rag"
    payload = json.dumps({
        "diary_entry": raw_text,
        "user_id": user_id,
        "session_id": session_id,
        "top_k": top_k,
    }).encode()
    req = urllib.request.Request(
        f"{_SERVER_URL}/{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read().decode().strip()

def put_reply(text, speech_enabled=True):
    SPEECH_ENABLED = speech_enabled
    print_assistant(f"Thanks for sharing!\n\n{text}")
    if SPEECH_ENABLED:
        speak("Thanks for sharing! " + text)

def get_ablation_preference():
    preference = input(
        "\nWhich response do you prefer? [A/B]: ").strip().lower()
    prefer_A = preference in ("a", "response a", "option a")
    if not prefer_A and preference not in ("b", "response b", "option b"):
        print("Invalid input. Please enter A or B.")
        return get_ablation_preference()
    return prefer_A

def should_continue_diary(ablation_key, speech_enabled=True):
    keep_going = input(
        "\nWould you like to write another entry? [Y/n]: ").strip().lower()
    y = keep_going not in ("n", "no", "stop", "quit", "exit", "q")
    if not y:
        prompt = f"Bye-bye! Your ablation key is {ablation_key}. Please fill out the survey."
        print_assistant(prompt + " :)")
        if speech_enabled:
            speak(prompt + " smiley face.")
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
