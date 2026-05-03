# starts diary application, passes mode of diary
def start_diary(speech_enabled=True) -> bool:
    pass

# gets diary application from user


def get_entry(speech_enabled=True) -> str:
    pass

# puts reply to diary entry


def diary_response(
    raw_text: str,
    user_id: str,
    session_id: str,
    top_k: int,
    listen: bool,
    ablation: bool,
) -> str:
    pass

def put_reply(text, speech_enabled=True):
    pass
