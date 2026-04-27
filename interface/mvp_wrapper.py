import json
from diary import start_diary, get_entry, gpt_diary_response, put_reply


def run_diary():
    start_diary()
    entry = get_entry()
    output_txt = gpt_diary_response(entry)
    put_reply(output_txt)


if __name__ == "__main__":
    run_diary()