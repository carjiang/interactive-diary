import os
os.environ["WANDB_MODE"] = "dryrun"
import sys
import json
import argparse
import pandas as pd
from collections import Counter
from tqdm import tqdm
import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')

from rich import print
from rich.panel import Panel
from rich import box

sys.path.append("..")
from thought_trace.agents.load_model import load_model
from thought_trace.tracer import load_tracer_model, get_tracer_parser

MODEL = 'gpt-4o-mini'
# MODEL = 'monkey'

PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TOMI_QUESTION_TYPES = [
#     'reality', 'memory',
#     'first_order_0_tom', 'first_order_1_tom', 'first_order_0_no_tom', 'first_order_1_no_tom',
#     'second_order_0_tom', 'second_order_1_tom', 'second_order_0_no_tom', 'second_order_1_no_tom'
# ]

# ANSWER_PROMPT_WITH_THEREFORE = "Therefore, the short one-sentence answer specifying the most detailed location including both the container and the place (e.g., from A in B) without any explanation is:" # one-sentence
# ANSWER_PROMPT = "The short one-sentence answer specifying the most detailed location including both the container and the place (e.g., from A in B) without any explanation is:" # one-sentence


def main(text):
    parser = get_tracer_parser()
    # parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max_questions_per_type', type=int, default=50)
    parser.add_argument('--tomi-set', type=str, default="paraphrased_tomi", help='ToMi subset to test')
    parser.add_argument('--input_file', type=str, help='Input file to test', default='test')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model', type=str,
                        help='Model to use to answer final question.')
    parser.add_argument('--use-cot',
                        type=bool,
                        default=False,
                        help='whether to run the model with zero-shot cot',
    )
    parser.add_argument('--run-id', type=str, required=True, help='Run ID')
    parser.add_argument('--print', action='store_true', help='whether to print the outputs')
    parser.add_argument('--existing-savepoint', default=None, help='path to existing savepoint')
    parser.add_argument('--reasoning-effort', type=str, help='Reasoning effort')
    args = parser.parse_args([
                "--model", MODEL,
                "--use-tracing",
                "--tracing-model", MODEL,
                # "--print",
                "--run-id", "tracer-first-run",
                "--dataset", "tomi",
                "--tracer-type", "tracer",
            ])
    # TOMI_DIR = os.path.join(PROJECT_BASE, "revised_tomi", args.tomi_set)
    # SAVED_INPUTS = os.path.join(TOMI_DIR, "saved_inputs")
    # os.makedirs(SAVED_INPUTS, exist_ok=True)
    tracer_name = args.tracing_model.replace("/", "-")
    # file_id = f"tracer-{tracer_name}_runid-{args.run_id}_use_cot-{args.use_cot}_num-qtypes-{str(args.max_questions_per_type)}"

    # args.answer_prompt = ANSWER_PROMPT
    # args.answer_prompt_with_therefore = ANSWER_PROMPT_WITH_THEREFORE

    agent = load_tracer_model(args=args)

    # text = 'Elizabeth stepped into the hallway. Benjamin arrived at the hallway. The box has the persimmon. The box is positioned in the hallway. Elizabeth conveyed the persimmon to the treasure chest. The treasure chest is stored in the hallway. Benjamin exited the hallway. Elizabeth walked out of the hallway. Benjamin tiptoed into the office. Benjamin is annoyed by the turnip.'
    # text = 'Yesterday morning I texted Maya that the meeting was moved, and my dog Rumi heard the notification. Later, I realized Maya had noticed the change before I did, so we agreed to call the new coordinator.'
    # text = 'I didn\'t expect today to turn into such a mess. It started in history class when Jake interrupted me during my presentation—again. I tried to ignore it at first, but when he laughed and made that comment about me “trying too hard,” something just snapped. I fired back, louder than I meant to, and suddenly the whole class went quiet. The teacher stepped in before it got worse, but the damage was already done. Now I keep replaying it in my head—his smirk, my voice shaking, everyone staring. Part of me is still angry, but another part feels embarrassed for losing control. I don\'t know if I should apologize tomorrow or just avoid him. I wish things could go back to normal, but I have a feeling they won\'t be that simple.'

    aggregate, hypotheses_list = agent.trace(text)
    # if MODEL == "gpt-4o-mini":
    #     agent.tracer_model.save_transcript_json()

    print("AGGREGATED", aggregate)
    print("FINISHED AGGREGATED")
    hypotheses_dicts = [h.dump() for h in hypotheses_list]
    print(hypotheses_dicts)
    return aggregate

if __name__ == '__main__':
    text = 'Yesterday morning I texted Maya that the meeting was moved, and my dog Rumi heard the notification. Later, I realized Maya had noticed the change before I did, so we agreed to call the new coordinator.'

    main(text)