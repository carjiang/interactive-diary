import sys
import os
import subprocess
import tempfile
import numpy as np
import sounddevice as sd

DURATION = 5
SAMPLERATE = 16000
CHANNELS = 1


def main(docker_image):
    print("Recording audio now... Interrupt with Ctrl+C when done.")
    audio = sd.rec(int(DURATION * SAMPLERATE),
                   samplerate=SAMPLERATE, channels=CHANNELS, dtype="int16")
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp, audio)
        tmp_path = tmp.name

    try:
        print("Sending to Docker instance for transcription...")
        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{tmp_path}:/app/temp_audio.npy:ro",
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
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    assert len(
        sys.argv) > 1, "Provide your_username/your_image_name as a command line argument"
    docker_image = sys.argv[1]
    main(docker_image)
