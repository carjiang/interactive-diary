import whisper


def test_transcribe():
    m = whisper.load_model("tiny")
    result = m.transcribe("tests/test_audio.m4a")
    print(result)
    assert result["text"] == " This is a portion of a test audio."
