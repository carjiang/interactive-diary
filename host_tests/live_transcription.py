import sys
import os
import subprocess
import numpy as np
import sounddevice as sd
DURATION = 5
SAMPLERATE = 16000
CHANNELS = 1


def main(docker_image):
    # Record on host
    print("Recording audio now... Interrupt with Ctrl+C when done.")
    audio = sd.rec(int(DURATION * SAMPLERATE),
                   samplerate=SAMPLERATE, channels=CHANNELS, dtype="int16")
    sd.wait()
    np.save("temp_audio.npy", audio)

    # Send to container for transcription
    print("Sending to Docker instance for transcription...")
    result = subprocess.run([
        "docker", "run", "--rm",
        "-v", f"{os.getcwd()}:/app",  # loads current directory into app
        docker_image,
        "python", "-c",
        "import numpy as np; import whisper; "
        "audio = np.load('/app/temp_audio.npy'); "
        "print('Audio loaded:', audio.shape); "
        "m = whisper.load_model('tiny'); "
        "print('Model loaded'); "
        "result = m.transcribe(audio.flatten().astype('float32') / 32768); "
        "print('TRANSCRIPTION:', result['text'])"
    ], capture_output=True, text=True)

    print(result.stdout)


if __name__ == "__main__":
    assert len(
        sys.argv) > 1, "Provide your_username/your_image_name as a command line argument"
    docker_image = sys.argv[1]
    main(docker_image)
