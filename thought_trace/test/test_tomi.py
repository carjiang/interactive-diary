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
from agents.load_model import load_model
from tracer import load_tracer_model, get_tracer_parser

PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOMI_QUESTION_TYPES = [
    'reality', 'memory',
    'first_order_0_tom', 'first_order_1_tom', 'first_order_0_no_tom', 'first_order_1_no_tom',
    'second_order_0_tom', 'second_order_1_tom', 'second_order_0_no_tom', 'second_order_1_no_tom'
]

ANSWER_PROMPT_WITH_THEREFORE = "Therefore, the short one-sentence answer specifying the most detailed location including both the container and the place (e.g., from A in B) without any explanation is:" # one-sentence
ANSWER_PROMPT = "The short one-sentence answer specifying the most detailed location including both the container and the place (e.g., from A in B) without any explanation is:" # one-sentence


def main(args):
    TOMI_DIR = os.path.join(PROJECT_BASE, "revised_tomi", args.tomi_set)
    SAVED_INPUTS = os.path.join(TOMI_DIR, "saved_inputs")
    os.makedirs(SAVED_INPUTS, exist_ok=True)
    tracer_name = args.tracing_model.replace("/", "-")
    file_id = f"tracer-{tracer_name}_runid-{args.run_id}_use_cot-{args.use_cot}_num-qtypes-{str(args.max_questions_per_type)}"

    args.answer_prompt = ANSWER_PROMPT
    args.answer_prompt_with_therefore = ANSWER_PROMPT_WITH_THEREFORE


    agent = load_tracer_model(args=args)

    # batch = ['Elizabeth stepped into the hallway. Benjamin arrived at the hallway. The box has the persimmon. The box is positioned in the hallway. Elizabeth conveyed the persimmon to the treasure chest. The treasure chest is stored in the hallway. Benjamin exited the hallway. Elizabeth walked out of the hallway. Benjamin tiptoed into the office. Benjamin is annoyed by the turnip.\n\nQuestion: Where does Elizabeth think that Benjamin searches for the persimmon?\nAnswer:']
    batch = ['Yesterday morning I texted Maya that the meeting was moved, and my dog Rumi heard the notification. Later, I realized Maya had noticed the change before I did, so we agreed to call the new coordinator.']
    use_tracings = [True]

    generated_thoughts = agent.batch_trace(batch, use_tracings)

    print(generated_thoughts)




if __name__ == "__main__":
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
                "--model", "gpt-4o-mini",
                "--use-tracing",
                "--tracing-model", "gpt-4o-mini",
                "--print",
                "--run-id", "tracer-first-run",
                "--dataset", "tomi",
                "--tracer-type", "tracer",
            ])
    main(args)
