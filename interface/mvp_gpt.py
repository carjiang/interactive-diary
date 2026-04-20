import json
from diary import start_diary, get_entry, gpt_diary_response, put_reply, ask_to_continue, end_diary


def run_diary():
    start_diary()
    entry = get_entry()
    output_txt = gpt_diary_response(entry)
    put_reply(output_txt)
    while ask_to_continue(): # loops if user has more to say
        entry = get_entry()
        output_txt = gpt_diary_response(entry)
        put_reply(output_txt)
    end_diary()

if __name__ == "__main__":
    run_diary()
