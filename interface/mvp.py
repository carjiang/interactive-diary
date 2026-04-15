import json
import interface.diary as diary


def run_diary():
    start_diary()
    entry = get_entry()
    action_states = extract_events(entry)
    output_txt = thought_trace(action_states)
    output(output_txt)


def start_diary():
    diary.start_diary()


def get_entry() -> str:
    return diary.get_entry()


def extract_events(entry: str) -> json:
    pass


def thought_trace(action_state_pairs: json) -> str:
    pass


def output(output_txt: str):
    diary.put_reply(output_txt)


if __name__ == "__main__":
    run_diary()
