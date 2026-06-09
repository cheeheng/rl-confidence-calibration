# Google Search Gemini AI is used to create part of the code
# Original template taken from the following source, but heavily modified afterwards
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb#scrollTo=DkIvEkIIkEyB

# Known potential bug
# Upon resumption, train/global_step is 0 during much of the first step, which will lead to inaccuracies when plotting graph if not corrected.

import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
# https://github.com/unslothai/unsloth-zoo/blob/2a80d543b9e22f68e051e32029c8a47005102895/unsloth_zoo/vllm_utils.py#L20
# Override due to occasional OOM
os.environ["UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE"] = "1"
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import math
import argparse
import re
import random
import pandas as pd
import time
import torch
import wandb

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
from utils import verify_correctness, extract_xml_tag, find_token_length, LLM_LONG_NAME, rl_format_regex, confidence_is_valid, derived_model_name_sft, derived_model_name_rl

# Configurable parameters
LLM_SHORT_NAME = "qwen2.5-3b"
DATASET_NAME = "multi-armed-bandit-64"
REWARD_FUNCTION = "brier-1"
LORA_RANK = 32
NUM_GENERATIONS = 8
QUESTIONS_PER_BATCH = 64
MAX_OUTPUT_TOKENS = 2048
NUM_TRAINING_STEPS = 500
NUM_CPU_THREADS = 8 # Lower this value if RAM is used up or CPU does not have enough cores
UNSLOTH_GRPO_MINI_BATCH_SIZE = 8
UNSLOTH_MAX_CHUNK_LENGTH = 4096

LLM_CONFIDENCE = True
CONFIDENCE_ANALYSIS = True
USE_WANDB = False

WANDB_TEAM_NAME = ""
WANDB_PROJECT_NAME = "rl-confidence-calibration"
WANDB_RESUME_ID = ""

# logit did not work well - with reward hacking when I tested, even though theoretically, it should not occur
CONFIDENCE_TRANSFORMATION_MODE = "linear"
    
def get_dataset(filename):
    data = pd.read_csv(filename)
    #print(data.iloc[0]['prompt_chat_template'])
    #assert False
    
    preprocessed_data = pd.DataFrame({
        "prompt": data["prompt_string"],
        "answer": data['ground_truth'],
        "group": data["group"]
    })
    
    #print(preprocessed_data.iloc[0]["prompt"])
    #assert False
    # Found this from Google Search Gemini AI
    return Dataset.from_pandas(preprocessed_data)

# Taken from Google Search Gemini AI 
# and https://www.geeksforgeeks.org/python/python-check-for-float-string/
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def T(confidence: float) -> float:
    if CONFIDENCE_TRANSFORMATION_MODE == "linear":
        return (confidence + 0.5) / 101
    else:
        assert False

def f_underconfidence(k, c):
    return (k*c + math.log1p(-k*c / (1+k))) / (k - math.log1p(k))

def g_underconfidence(k, c):
    return (k*c + (k+1)*math.log1p(-k*c / (1+k))) / (k - math.log1p(k))

def f_overconfidence(k, c):
    return ((k+1)*math.log(c*k+1) - c*k) / ((k+1)*math.log(k+1) - k)

def g_overconfidence(k, c):
    return (math.log(c*k+1) - c*k) / ((k+1)*math.log(k+1) - k)

def f(confidence):
    if REWARD_FUNCTION == "rlvr":
        return 1
    elif REWARD_FUNCTION == "log-1":
        return 1 + math.log(confidence)
    elif REWARD_FUNCTION == "log-no-correctness-reward":
        return math.log(confidence)
    elif REWARD_FUNCTION == "log-1-div-ln-202":
        return 1 + (math.log(confidence) / math.log(202))
    elif REWARD_FUNCTION == "brier-no-correctness-reward":
        return - (1 - confidence) ** 2
    elif REWARD_FUNCTION == "brier-1":
        return 1 - (1 - confidence) ** 2
    elif REWARD_FUNCTION == "brier-2":
        return 1 - 2 * ((1 - confidence) ** 2)
    elif REWARD_FUNCTION == "brier-log-hybrid":
        return confidence
    elif REWARD_FUNCTION == "underconfidence-1":
        return f_underconfidence(1, confidence)
    elif REWARD_FUNCTION == "underconfidence-4":
        return f_underconfidence(4, confidence)
    elif REWARD_FUNCTION == "overconfidence-1":
        return f_overconfidence(1, confidence)
    elif REWARD_FUNCTION == "overconfidence-4":
        return f_overconfidence(4, confidence)
    elif REWARD_FUNCTION == "overconfidence-1000":
        return f_overconfidence(1000, confidence)
    else:
        assert False
    
def g(confidence):
    if REWARD_FUNCTION == "rlvr":
        return 0
    elif REWARD_FUNCTION == "log-1":
        return math.log(1-confidence)
    elif REWARD_FUNCTION == "log-no-correctness-reward":
        return math.log(1-confidence)
    elif REWARD_FUNCTION == "log-1-div-ln-202":
        return (math.log(1-confidence) / math.log(202))
    elif REWARD_FUNCTION == "brier-no-correctness-reward":
        return - confidence ** 2
    elif REWARD_FUNCTION == "brier-1":
        return - confidence ** 2
    elif REWARD_FUNCTION == "brier-2":
        return - 2 * (confidence ** 2)
    elif REWARD_FUNCTION == "brier-log-hybrid":
        return confidence + math.log(1-confidence)
    elif REWARD_FUNCTION == "underconfidence-1":
        return g_underconfidence(1, confidence)
    elif REWARD_FUNCTION == "underconfidence-4":
        return g_underconfidence(4, confidence)
    elif REWARD_FUNCTION == "overconfidence-1":
        return g_overconfidence(1, confidence)
    elif REWARD_FUNCTION == "overconfidence-4":
        return g_overconfidence(4, confidence)
    elif REWARD_FUNCTION == "overconfidence-1000":
        return g_overconfidence(1000, confidence)
    else:
        assert False

'''
for i in range(101):
    p = T(i)
    print(i, p, f(p), g(p), p*f(p) + (1-p)*g(p))
assert False
'''

# initialize with empty set first, will be updated in main
group_set = {} 
rolling_average_confidence_by_group = None
rolling_average_accuracy_by_group = None
rolling_average_batch_confidence = 0.5
rolling_average_batch_accuracy = 0.5

# Reward functions
overall_correct = 0
overall_wrong = 0
overall_valid = 0
batches_completed = 0
def correctness_reward_func(prompts, completions, answer, group, **kwargs) -> list[float]:
    global overall_correct, overall_wrong, overall_valid, rolling_average_confidence_by_group, rolling_average_accuracy_by_group, group_set
    global rolling_average_batch_confidence, rolling_average_batch_accuracy
    global batches_completed
    alpha = 0.2 # exponential average smoothing constant
    group_list = list(group)
    
    this_time_total_confidence = 0
    this_time_valid_confidences = 0
    responses = [completion for completion in completions]
    q = prompts[0]
    extracted_responses = [(extract_xml_tag(r, "answer"), extract_xml_tag(r, "confidence")) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0][0]}", f"\nConfidence:\n{extracted_responses[0][1]}", f"\n")
    
    rewards = []
    confidence_list = []
    this_time_total_confidence_by_group = {group: 0 for group in group_set}
    this_time_valid_confidence_by_group = {group: 0 for group in group_set}
    this_time_total_questions_by_group = {group: 0 for group in group_set}
    this_time_total_correct_by_group = {group: 0 for group in group_set}
    for r, a, group in zip(extracted_responses, answer, group_list):
        score = 0
        if group not in this_time_total_confidence_by_group:
            this_time_total_confidence_by_group[group] = 0
            this_time_valid_confidence_by_group[group] = 0
            this_time_total_questions_by_group[group] = 0
            this_time_total_correct_by_group[group] = 0
        
        # Bug fix: Include experiment_id = run_id (from wandb) to reduce evaluation conflicts due to race condition
        correct = verify_correctness(DATASET_NAME, r[0], str(a), experiment_id = run_id)
    
        if confidence_is_valid(r[1]):
            confidence = float(r[1])
            confidence_list.append(confidence)
            overall_valid += 1
            confidence = T(confidence)
            this_time_total_confidence += confidence
            this_time_valid_confidences += 1
            
            this_time_total_confidence_by_group[group] += confidence
            this_time_valid_confidence_by_group[group] += 1
        else:
            # Assume the worst possible confidence score to penalize LLM for incorrect formatting of confidence
            if correct:
                confidence = T(0)
            else:
                confidence = T(100)

        this_time_total_questions_by_group[group] += 1
        
        if correct:
            score += f(confidence)
            overall_correct += 1
            this_time_total_correct_by_group[group] += 1
        else:
            score += g(confidence)
            overall_wrong += 1
        rewards.append(score)
    
    print(extracted_responses)
    print(sorted(confidence_list))
    total_responses = len(extracted_responses)
    if USE_WANDB:
        run.log({"batch_statistics": {
            "effective_batch_size": total_responses,
            "valid_confidence_values": this_time_valid_confidences,
            # In an older version of the code, this was misspelt as 'proprotion_of_valid_confidence_values'
            "proportion_of_valid_confidence_values": this_time_valid_confidences / total_responses,
        }})

    if this_time_valid_confidences != 0:
        average_batch_confidence = this_time_total_confidence / this_time_valid_confidences
        rolling_average_batch_confidence = (1-alpha) * rolling_average_batch_confidence + alpha * average_batch_confidence
        print("This batch: %d confidence values valid, average implied confidence %.4f, rolling average %.4f" % 
            (this_time_valid_confidences, average_batch_confidence, rolling_average_batch_confidence))
    else:
        average_batch_confidence = None
    if USE_WANDB:
        run.log({"batch_statistics": {
            "average_confidence": average_batch_confidence,
            "rolling_average_confidence": rolling_average_batch_confidence,
        }})
    
    # Edit: Fix bug where some statistics would not show since some groups may not show up at small batch sizes
    for group in group_set:
        if this_time_valid_confidence_by_group[group] != 0:
            average_confidence = this_time_total_confidence_by_group[group] / this_time_valid_confidence_by_group[group]
        
            rolling_average_confidence_by_group[group] = (1-alpha) * rolling_average_confidence_by_group[group] + alpha * average_confidence
            print("Group %s: %d confidence values valid, average implied confidence %.4f, rolling average %.4f" % (group, this_time_valid_confidence_by_group[group], 
                average_confidence, rolling_average_confidence_by_group[group]))
        else:
            # To maintain consistency with previous versions, we do not update rolling averages here.
            average_confidence = None
        if USE_WANDB:
            run.log({"group_%s_statistics" % group: {
                "average_confidence": average_confidence,
                "rolling_average_confidence": rolling_average_confidence_by_group[group],
            }})
    
    this_time_correct = 0
    for group in this_time_total_questions_by_group.keys():
        this_time_correct += this_time_total_correct_by_group[group]
    batch_accuracy = this_time_correct / total_responses
    rolling_average_batch_accuracy = (1-alpha) * rolling_average_batch_accuracy + alpha * batch_accuracy
    
    print()
    # Older versions of the code did not include rolling average batch accuracy
    print("This batch: %d out of %d correct, overall accuracy %.4f, rolling average %.4f" % 
        (this_time_correct, total_responses, batch_accuracy, rolling_average_batch_accuracy))
    if USE_WANDB:
        run.log({"batch_statistics": {
            "correct_responses": this_time_correct,
            "total_responses": total_responses,
            "accuracy": batch_accuracy,
            "rolling_average_accuracy": rolling_average_batch_accuracy,
        }})

    for group in group_set:
        if USE_WANDB:
            run.log({"group_%s_statistics" % group: {
                "group_responses": this_time_total_questions_by_group[group],
            }})
        if this_time_total_questions_by_group[group] != 0:
            average_accuracy = this_time_total_correct_by_group[group] / this_time_total_questions_by_group[group]
        
            rolling_average_accuracy_by_group[group] = (1-alpha) * rolling_average_accuracy_by_group[group] + alpha * average_accuracy
            print("Group %s: %d questions, %d correct, average accuracy %.4f, rolling average %.4f" % (group, this_time_total_questions_by_group[group], 
                this_time_total_correct_by_group[group], average_accuracy, rolling_average_accuracy_by_group[group]))
        else:
            average_accuracy = None
            
        if USE_WANDB:
            run.log({"group_%s_statistics" % group: {
                "correct_responses": this_time_total_correct_by_group[group],
                "accuracy": average_accuracy,
                "rolling_average_accuracy": rolling_average_accuracy_by_group[group],
            }})
    
    batches_completed += 1
    print()
    # Earlier versions printed this for easier debugging, now it is unnecessary since there is wandb
    #print("Correct:", overall_correct, "Wrong:", overall_wrong, "Valid confidence scores:", overall_valid)
    print("Training batches/steps completed:", batches_completed)
    return rewards

def valid_confidence_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    q = prompts[0]
    extracted_responses = [(extract_xml_tag(r, "answer"), extract_xml_tag(r, "confidence")) for r in responses]
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0][0]}", f"\nConfidence:\n{extracted_responses[0][1]}", f"\n")
    
    rewards = []
    for r, a in zip(extracted_responses, answer):
        if confidence_is_valid(r[1]):
            rewards.append(1)
        else:
            rewards.append(0)
    return rewards

def is_valid_answer_format(r: str) -> bool:
    if DATASET_NAME == "addition":
        return r.strip().isnumeric()
    elif DATASET_NAME.startswith("multi-armed-bandit"):
        answer = r.strip()
        max_num = 6 if DATASET_NAME == "multi-armed-bandit-22222" else 3
        return answer in [str(i) for i in range(1, max_num + 1)]
    elif DATASET_NAME in ["hotpotqa", "hotpotqa-modified", "deepmath-103k", "bigmath"]:
        return len(r) <= 1000 # discourage overly long final answers
    elif DATASET_NAME in ["noisy-ground-truth-sequential", "noisy-ground-truth-random"]:
        if not r.strip().isnumeric():
            return False
        return int(r) >= 0 and int(r) <= 999
    else:
        assert False

def answer_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    extracted_responses = [extract_xml_tag(r, "answer") for r in responses]
    return [0.5 if is_valid_answer_format(r) else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = RL_FORMAT_REGEX
    #pattern = r"^<reasoning>(.|\n)*<\/reasoning>(.|\n)*<answer>(.|\n)*<\/answer>\n<confidence>(.|\n)*<\/confidence>"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    #print(responses, matches)
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    # There are formatting bugs that were missed out in earlier experiments.
    # The bug is minor and it is not expected to significantly impact experimental results since formatting instructions are generally followed.
    # The code below has been fixed for completeness, ensuring that every XML-like tag has a score for inclusion.

    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.1
    if text.count("\n</reasoning>\n") == 1:
        count += 0.1
    #if text.count("<answer>\n") == 1:
    # Missing "\n" in <answer> that allows answer to be in the front, which is not originally intended.
    if text.count("\n<answer>\n") == 1:
        count += 0.1
    if text.count("\n</answer>\n") == 1:
        count += 0.1
    if text.count("\n<confidence>\n") == 1:
        count += 0.1
    if text.count("\n</confidence>") == 1:
        count += 0.1

    # This part is added: <confidence_analysis> tags missed out in earlier experiments.
    if CONFIDENCE_ANALYSIS:
        if text.count("\n<confidence_analysis>\n") == 1:
            count += 0.1
        if text.count("\n</confidence_analysis>\n") == 1:
            count += 0.1
        
    #count -= len(text.split("\n</answer>\n")[-1])*0.001
    #count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    #print("To avoid confusion, I have added this line of code")
    #print(completions[0])
    #print("To avoid confusion, I have added this line of code")
    #print(kwargs)
    #assert False
    contents = [completion for completion in completions]
    return [count_xml(c) for c in contents]
    
if __name__ == "__main__":
    # https://docs.python.org/3/library/argparse.html
    # Argparse template taken from https://github.com/zhytk/RAREval-data-processing/blob/main/few_shot.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME, dest='dataset_name', type=str)
    parser.add_argument("--llm", default=LLM_SHORT_NAME, dest='llm_short_name', type=str)
    parser.add_argument("--reward_function", default=REWARD_FUNCTION, dest='reward_function', type=str)
    parser.add_argument("--max_output_tokens", default=MAX_OUTPUT_TOKENS, dest='max_output_tokens', type=int)
    parser.add_argument("--lora_rank", default=LORA_RANK, dest='lora_rank', type=int)
    parser.add_argument("--num_generations", default=NUM_GENERATIONS, dest='num_generations', type=int)
    parser.add_argument("--questions_per_batch", default=QUESTIONS_PER_BATCH, dest='questions_per_batch', type=int)
    parser.add_argument("--num_training_steps", default=NUM_TRAINING_STEPS, dest='num_training_steps', type=int)
    parser.add_argument("--num_cpu_threads", default=NUM_CPU_THREADS, dest='num_cpu_threads', type=int)
    parser.add_argument("--unsloth_grpo_mini_batch_size", default=UNSLOTH_GRPO_MINI_BATCH_SIZE, dest='unsloth_grpo_mini_batch_size', type=int)
    parser.add_argument("--unsloth_max_chunk_length", default=UNSLOTH_MAX_CHUNK_LENGTH, dest='unsloth_max_chunk_length', type=int)
    parser.add_argument("--llm_confidence", default=LLM_CONFIDENCE, dest='llm_confidence', action=argparse.BooleanOptionalAction)
    parser.add_argument("--confidence_analysis", default=CONFIDENCE_ANALYSIS, dest='confidence_analysis', action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_wandb", default=USE_WANDB, dest='use_wandb', action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_team_name", default=WANDB_TEAM_NAME, dest='wandb_team_name', type=str)
    parser.add_argument("--wandb_project_name", default=WANDB_PROJECT_NAME, dest='wandb_project_name', type=str)
    parser.add_argument("--wandb_resume_id", default=WANDB_RESUME_ID, dest='wandb_resume_id', type=str)
    args = parser.parse_args()
    print(args)
    
    DATASET_NAME = args.dataset_name
    LLM_SHORT_NAME = args.llm_short_name
    REWARD_FUNCTION = args.reward_function
    MAX_OUTPUT_TOKENS = args.max_output_tokens
    LORA_RANK = args.lora_rank
    NUM_GENERATIONS = args.num_generations
    QUESTIONS_PER_BATCH = args.questions_per_batch
    NUM_TRAINING_STEPS = args.num_training_steps
    NUM_CPU_THREADS = args.num_cpu_threads
    UNSLOTH_GRPO_MINI_BATCH_SIZE = args.unsloth_grpo_mini_batch_size
    UNSLOTH_MAX_CHUNK_LENGTH = args.unsloth_max_chunk_length
    LLM_CONFIDENCE = args.llm_confidence
    CONFIDENCE_ANALYSIS = args.confidence_analysis
    USE_WANDB = args.use_wandb
    WANDB_TEAM_NAME = args.wandb_team_name
    WANDB_PROJECT_NAME = args.wandb_project_name
    WANDB_RESUME_ID = args.wandb_resume_id

    RL_FORMAT_REGEX = rl_format_regex(CONFIDENCE_ANALYSIS)
    model_sft_filename = derived_model_name_sft(DATASET_NAME, LLM_SHORT_NAME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)
    derived_model_name = derived_model_name_rl(DATASET_NAME, LLM_SHORT_NAME, REWARD_FUNCTION, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)
    rl_model_output_dir = "models/rl/%s" % derived_model_name

    timestamp = time.time_ns()
    resumed_run = False
    if USE_WANDB:
        # Taken from wandb quickstart guide
        wandb_init_kwargs = {
            # Set the wandb entity where your project will be logged (generally your team name).
            "entity": WANDB_TEAM_NAME if WANDB_TEAM_NAME != "" else None,
            # Set the wandb project where this run will be logged.
            "project": WANDB_PROJECT_NAME if WANDB_PROJECT_NAME != "" else None,
            # Name of the run
            "name": derived_model_name + ("_%d" % timestamp),
            # Track hyperparameters and run metadata.
            # vars() converts Namespace to dict() - according to Google Search Gemini AI
            "config": {"cmdline_args": vars(args)},
        }
        
        if WANDB_RESUME_ID != "":
            run = wandb.init(**wandb_init_kwargs, id=WANDB_RESUME_ID, resume="must", allow_val_change=False)
            resumed_run = True
        else:
            # Older versions of the code did not have this.
            # Now, wandb will allow resumption of run if resume_id is provided
            run = wandb.init(**wandb_init_kwargs)
    
        run_id = wandb.run.id
    else:
        run_id = wandb.util.generate_id()

    # Preprocess dataset and intialize groups
    # Note: This is where the statistics are initialized
    print("Timestamp (ns): %d" % timestamp)
    print("Run ID: %s" % run_id)
    dataset = get_dataset("datasets/rl-train/%s.csv" % model_sft_filename)
    group_set = set(dataset["group"])

    if resumed_run:
        print("Resuming run")

        # First, check the checkpoint index.
        rl_model_output_dir_path = Path(rl_model_output_dir)
        #print(rl_model_output_dir_path)
        checkpoint_str = "checkpoint-"
        checkpoint_indices = list(rl_model_output_dir_path.glob(checkpoint_str + "*"))
        batches_completed = -1
        for path in checkpoint_indices:
            try:
                checkpoint_index = int(path.name[len(checkpoint_str):])
                assert checkpoint_index >= 1
                batches_completed = max(checkpoint_index, batches_completed)
            except ValueError:
                print("Warning: checkpoint index invalid in %s" % path.name)
                pass

        print("Batches/Steps completed: %d" % batches_completed)

        if batches_completed == -1:
            assert False, "Checkpoint not found"

        api = wandb.Api()
        api_run = api.run("%s/%s/%s" % (WANDB_TEAM_NAME, WANDB_PROJECT_NAME, WANDB_RESUME_ID))
        history_df = api_run.history(samples=NUM_TRAINING_STEPS*1000)

        # For debugging purposes only
        DEBUG_WANDB_RESUME = False
        if DEBUG_WANDB_RESUME:
            for i in range(len(history_df)):
                row = history_df.iloc[i]
                step_number = row["train/global_step"]
                assert step_number == step_number # Ensure not NaN
                for column in row.keys():
                    if row[column] == row[column]: # Checks for not NaN
                        print(i, step_number, column, row[column])

        run_summary = {}
        # Whatever that is train/ is only updated when the global step number has incremented.
        # Of course, the only exception is train/global_step
        # The others are updated before the global step number is incremented.
        # Ignore _step, _runtime and _timestamp, these are logs when wandb.log is called
        # This code assumes sorting by runtime/timestamp.
        # This code assumes save_steps >= 2 because during much of the first step after resumption, the value train/global_step is 0.
        history_df = history_df[(history_df["train/global_step"] == batches_completed) | (history_df["train/global_step"] == batches_completed-1)]
        history_df = history_df.sort_values(by='_runtime')
        for i in range(len(history_df)):
            row = history_df.iloc[i]
            step_number = row["train/global_step"]
            assert step_number == step_number # Ensure not NaN
            for column in row.keys():
                if column == "train/global_step" or column in ["_step", "_runtime", "_timestamp"]:
                    continue
                if column[:6] == "train/" and step_number == batches_completed-1:
                    # Wrong step number
                    continue
                if column[:6] != "train/" and step_number == batches_completed:
                    # Wrong step number
                    continue

                if row[column] == row[column]: # Checks for not NaN
                    run_summary[column] = row[column]
                    print(i, step_number, column, row[column])


        rolling_average_batch_accuracy = run_summary["batch_statistics.rolling_average_accuracy"]
        rolling_average_batch_confidence = run_summary["batch_statistics.rolling_average_confidence"]

        rolling_average_confidence_by_group = {}
        rolling_average_accuracy_by_group = {}
        print(run_summary.keys())
        for group in group_set:
            group_statistics_name = "group_%s_statistics" % group
            group_confidence_name = group_statistics_name + ".rolling_average_confidence"
            group_accuracy_name = group_statistics_name + ".rolling_average_accuracy"
            if group_confidence_name not in run_summary.keys():
                assert group_accuracy_name not in run_summary.keys()
                rolling_average_confidence_by_group[group] = 0.5
                rolling_average_accuracy_by_group[group] = 0.5
                continue

            rolling_average_confidence_by_group[group] = run_summary[group_confidence_name]
            rolling_average_accuracy_by_group[group] = run_summary[group_accuracy_name]

        #print(run_summary.keys())
        #assert False
    else:
        rolling_average_batch_confidence = 0.5
        rolling_average_batch_accuracy = 0.5

        overall_correct = 0
        overall_wrong = 0
        overall_valid = 0
        batches_completed = 0

        rolling_average_confidence_by_group = {group: 0.5 for group in group_set}
        rolling_average_accuracy_by_group = {group: 0.5 for group in group_set}
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_LONG_NAME[LLM_SHORT_NAME])

    dataset = dataset.map(find_token_length, batched=True, num_proc=NUM_CPU_THREADS, fn_kwargs={"tokenizer": tokenizer})
    max_prompt_length = max(list(dataset["token_length"]))
    print("Longest prompt in dataset has %d token(s)." % max_prompt_length)
    
    # Delay creation of model to ease debugging of dataset preprocessing
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "models/sft/%s" % model_sft_filename,
        #model_name = LLM_LONG_NAME[LLM_SHORT_NAME],
        max_seq_length = MAX_OUTPUT_TOKENS + max_prompt_length + 5,
        load_in_4bit = False, # False for LoRA 16bit
        load_in_fp8 = False,
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = LORA_RANK,
        gpu_memory_utilization = 0.8, # Reduce if out of memory,
        dtype = torch.float16,
        device_map = "balanced",
        #unsloth_tiled_mlp = True,
    )
    
    if not os.path.exists("models/rl"):
        os.mkdir("models/rl")

    effective_batch_size = QUESTIONS_PER_BATCH * NUM_GENERATIONS # only 1 device for now

    learning_rate = 1e-5

    training_args = GRPOConfig(
        beta = 0.0,
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = learning_rate,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        #warmup_ratio = 0.1,
        warmup_steps = 25,
        lr_scheduler_type = "constant",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = NUM_GENERATIONS,
        gradient_accumulation_steps = QUESTIONS_PER_BATCH, # Increase to 4 for smoother training
        steps_per_generation = QUESTIONS_PER_BATCH,
        num_generations = NUM_GENERATIONS, # Decrease if out of memory
        max_completion_length = MAX_OUTPUT_TOKENS,
        max_prompt_length = max_prompt_length + 5, # will be deprecated in trl 0.26.0, removed in 0.28.0
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = NUM_TRAINING_STEPS,
        # Save frequency must be at most once every two steps because program resumption logic only suppors save_steps >= 2.
        save_steps = 5, # More frequent saves to reduce lost iterations in future, older versions have save_steps=20
        save_total_limit = 3,
        # We find that the training instability in Llama 3.1 (8B) Instruct and Llama 3.2 (3B) Instruct may be caused by too high a gradient norm limit.
        # We should have done that for the other models we test - but we left the max_grad_norm as 0.1 for Qwen 2.5 (3B) Instruct for reproducibility.
        #max_grad_norm = 0.1 if LLM_SHORT_NAME == "qwen2.5-3b" else 0.01,
        # Note that this line has been changed for consistency with experimental settings in paper
        max_grad_norm = 0.1 if LLM_SHORT_NAME in ["qwen2.5-3b", "llama3.2-3b"] else 0.01,
        output_dir = rl_model_output_dir,
        loss_type = 'dr_grpo',
        scale_rewards = False,
        mask_truncated_completions = True,
        importance_sampling_level="sequence",
        unsloth_grpo_mini_batch = 1 + (effective_batch_size - 1) // UNSLOTH_GRPO_MINI_BATCH_SIZE,
        unsloth_logit_chunk_multiplier = 1 + (max_prompt_length + 4) // UNSLOTH_MAX_CHUNK_LENGTH,
        # wandb integration
        report_to = "wandb" if USE_WANDB else "none",
        run_name = "rl_confidence_calibration-" + run_id,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            strict_format_reward_func,
            answer_reward_func,
            valid_confidence_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train(resume_from_checkpoint = resumed_run)

    if USE_WANDB:
        run.finish()
