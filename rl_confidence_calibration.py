# Google Search Gemini AI is used to assist in creating part of the code, with human supervision and verification.
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

from unsloth import FastLanguageModel, FastVisionModel

import math
import argparse
import re
import random
import pandas as pd
import time
import torch
import wandb
import yaml

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from vllm.config import ReasoningConfig
from vllm.sampling_params import StructuredOutputsParams
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from pathlib import Path
from collections import Counter
from functools import lru_cache
from peft import LoraConfig, get_peft_model
from utils import find_token_length, LLM_LONG_NAME, normalize_confidence, parse_and_grade_response, is_multimodal, model_supports_float16
from utils import get_confidence_reward, get_rl_train_path, AnswerFormat, get_rl_temperature, get_rl_top_p, get_rl_top_k, get_rl_min_p

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

# Callback tutorial from https://discuss.huggingface.co/t/how-to-write-custom-trainercallback-functions-with-custom-arguments/151063
# Retrieved with the help of Google Search Gemini AI
class RLStatsCallback(TrainerCallback):
    def __init__(self):
        # initialize with empty set first, will be updated in main
        self.group_set = {} 
        self.rolling_average_confidence_by_group = None
        self.rolling_average_accuracy_by_group = None
        self.rolling_average_batch_confidence = 0.5
        self.rolling_average_batch_accuracy = 0.5

        # Reward functions
        self.overall_correct = 0
        self.overall_wrong = 0
        self.overall_valid = 0
        self.batches_completed = 0

        # Statistics for current batch
        self.sub_batch_count = 0
        self.this_time_total_confidence_by_group = {}
        self.this_time_valid_confidence_by_group = {}
        self.this_time_total_questions_by_group = {}
        self.this_time_total_correct_by_group = {}
        self.this_batch_shortened_extracted_responses = []
        self.this_batch_confidence_list = []
        self.this_time_total_confidence = 0
        self.this_time_valid_confidences = 0
        self.sub_batch_count = 0
        
        self.wandb_run = None
    
    def reset_current_batch_statistics(self):
        self.this_time_total_confidence_by_group = {group: 0 for group in self.group_set}
        self.this_time_valid_confidence_by_group = {group: 0 for group in self.group_set}
        self.this_time_total_questions_by_group = {group: 0 for group in self.group_set}
        self.this_time_total_correct_by_group = {group: 0 for group in self.group_set}
        self.this_batch_shortened_extracted_responses = []
        self.this_batch_confidence_list = []
        self.this_time_total_confidence = 0
        self.this_time_valid_confidences = 0
        self.sub_batch_count = 0
    
    # Override from parent class
    # https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/callback#transformers.TrainerCallback
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        if len(self.this_batch_shortened_extracted_responses) == 0:
            # No batch
            return
    
        # Note: In earlier versions, alpha was always set to 0.2, with initial value at 0.5
        # Now, we allow adjusting the smoothing factor alpha in the config file, with overwrites in the beginning to ensure one value does not overly affect the others.
        
        # In general, we take exponential moving average with smoothing factor of SMOOTHING_FACTOR_ALPHA (Brown, 1956)
        # If there are not enough steps completed, we take the arithmetic mean across all steps.
        # If some values are not defined, we assume the determined average applies evenly across all previous steps.
        alpha = max(SMOOTHING_FACTOR_ALPHA, 1 / (self.batches_completed + 1))
    
        # Changed: extracted responses are now shortened so that responses with incorrect format do not take up too many bytes of the output stream
        print("INFO: Responses in the list below are truncated to 100 characters and confidence values are truncated to 5 characters.")
        print(self.this_batch_shortened_extracted_responses)
        print(dict(sorted(Counter(self.this_batch_confidence_list).items())))
        #total_responses = len(extracted_responses)
        total_responses = len(self.this_batch_shortened_extracted_responses)
        
        # Initialize additional log parameters
        logs["batch_statistics"] = {}
        for group in self.group_set:
            logs["group_%s_statistics" % group] = {}
        
        logs["batch_statistics"].update({
            "effective_batch_size": total_responses,
            "valid_confidence_values": self.this_time_valid_confidences,
            # In an older version of the code, this was misspelt as 'proprotion_of_valid_confidence_values'
            "proportion_of_valid_confidence_values": self.this_time_valid_confidences / total_responses,
        })

        if self.this_time_valid_confidences != 0:
            average_batch_confidence = self.this_time_total_confidence / self.this_time_valid_confidences
            self.rolling_average_batch_confidence = (1-alpha) * self.rolling_average_batch_confidence + alpha * average_batch_confidence
            print("This step: %d confidence values valid, average implied confidence %.4f, rolling average %.4f" % 
                (self.this_time_valid_confidences, average_batch_confidence, self.rolling_average_batch_confidence))
        else:
            average_batch_confidence = None

        logs["batch_statistics"].update({
            "average_confidence": average_batch_confidence,
            "rolling_average_confidence": self.rolling_average_batch_confidence,
        })
        
        # Edit: Fix bug where some statistics would not show since some groups may not show up at small batch sizes
        for group in self.group_set:
            if self.this_time_valid_confidence_by_group[group] != 0:
                average_confidence = self.this_time_total_confidence_by_group[group] / self.this_time_valid_confidence_by_group[group]
            
                self.rolling_average_confidence_by_group[group] = (1-alpha) * self.rolling_average_confidence_by_group[group] + alpha * average_confidence
                print("Group %s: %d confidence values valid, average implied confidence %.4f, rolling average %.4f" % (group, self.this_time_valid_confidence_by_group[group], 
                    average_confidence, self.rolling_average_confidence_by_group[group]))
            else:
                # To maintain consistency with previous versions, we do not update rolling averages here.
                average_confidence = None

            logs["group_%s_statistics" % group].update({
                "average_confidence": average_confidence,
                "rolling_average_confidence": self.rolling_average_confidence_by_group[group],
            })
        
        self.this_time_correct = 0
        for group in self.this_time_total_questions_by_group.keys():
            self.this_time_correct += self.this_time_total_correct_by_group[group]
        batch_accuracy = self.this_time_correct / total_responses
        self.rolling_average_batch_accuracy = (1-alpha) * self.rolling_average_batch_accuracy + alpha * batch_accuracy
        
        print()
        # Older versions of the code did not include rolling average batch accuracy
        print("This step: %d out of %d correct, overall accuracy %.4f, rolling average %.4f" % 
            (self.this_time_correct, total_responses, batch_accuracy, self.rolling_average_batch_accuracy))

        logs["batch_statistics"].update({
            "correct_responses": self.this_time_correct,
            "total_responses": total_responses,
            "accuracy": batch_accuracy,
            "rolling_average_accuracy": self.rolling_average_batch_accuracy,
        })

        for group in self.group_set:
            if self.this_time_total_questions_by_group[group] != 0:
                average_accuracy = self.this_time_total_correct_by_group[group] / self.this_time_total_questions_by_group[group]
            
                self.rolling_average_accuracy_by_group[group] = (1-alpha) * self.rolling_average_accuracy_by_group[group] + alpha * average_accuracy
                print("Group %s: %d questions, %d correct, average accuracy %.4f, rolling average %.4f" % (group, self.this_time_total_questions_by_group[group], 
                    self.this_time_total_correct_by_group[group], average_accuracy, self.rolling_average_accuracy_by_group[group]))
            else:
                average_accuracy = None
                
            logs["group_%s_statistics" % group].update({
                "group_responses": self.this_time_total_questions_by_group[group],
                "correct_responses": self.this_time_total_correct_by_group[group],
                "accuracy": average_accuracy,
                "rolling_average_accuracy": self.rolling_average_accuracy_by_group[group],
            })
        
        if self.wandb_run is not None:
            self.wandb_run.log({"batch_statistics": logs["batch_statistics"]})
            for group in self.group_set:
                self.wandb_run.log({"group_%s_statistics" % group: logs["group_%s_statistics" % group]})
        
        self.batches_completed += 1
        print()
        # Earlier versions printed this for easier debugging, now it is unnecessary since there is wandb
        #print("Correct:", overall_correct, "Wrong:", overall_wrong, "Valid confidence scores:", overall_valid)
        print("Training batches completed:", self.batches_completed)
        rl_stats_callback.reset_current_batch_statistics()
        print(logs)

rl_stats_callback = RLStatsCallback()

def correctness_reward_func(prompts, completions, answer, group, **kwargs) -> list[float]:
    global rl_stats_callback
    
    rl_stats_callback.sub_batch_count += 1

    group_list = list(group)

    responses = [completion for completion in completions]
    
    batch_size = len(prompts)
    assert len(completions) == batch_size and len(answer) == batch_size and len(group) == batch_size
    
    graded_responses = [None for i in range(batch_size)]
    for i in range(batch_size):
        graded_responses[i] = cached_parse_and_grade_response(responses[i], DATASET_NAME, answer[i], experiment_id = run_id)
        #print(responses[i])
        #print(graded_responses[i])
    
    q = prompts[0]
    #extracted_responses = [(extract_xml_tag(r, "answer"), extract_xml_tag(r, "confidence")) for r in responses]
    extracted_responses = [(r["answer"], r["raw_confidence"]) for r in graded_responses]
    shortened_extracted_responses = [(answer[:100], confidence) for answer, confidence in extracted_responses] # for display of output only, not for response grading
    
    # https://stackoverflow.com/questions/2347265/why-does-behave-unexpectedly-on-lists
    # This is intended - shortened_extracted_responses is appended in-place to this_batch_shortened_extracted_responses
    rl_stats_callback.this_batch_shortened_extracted_responses += shortened_extracted_responses
    
    if rl_stats_callback.sub_batch_count == 1:
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0][0]}", f"\nConfidence:\n{extracted_responses[0][1]}", f"\n")
    
    rewards = []
    for r, a, group in zip(graded_responses, answer, group_list):            
        correct = r["correct"]
        unnormalized_confidence = r["confidence"]
        confidence = normalize_confidence(unnormalized_confidence)

        score = get_confidence_reward(REWARD_FUNCTION, confidence, correct)
        rl_stats_callback.this_time_total_questions_by_group[group] += 1
        if correct:
            rl_stats_callback.overall_correct += 1
            rl_stats_callback.this_time_total_correct_by_group[group] += 1
        else:
            rl_stats_callback.overall_wrong += 1
        
        # Valid confidence: defined as when both the answer and confidence can be read (most important to do proper grading)
        valid_confidence = "answer" not in r["invalid_fields"] and "confidence" not in r["invalid_fields"]
        if valid_confidence:
            rl_stats_callback.this_batch_confidence_list.append(unnormalized_confidence)
            rl_stats_callback.overall_valid += 1
            rl_stats_callback.this_time_total_confidence += confidence
            rl_stats_callback.this_time_valid_confidences += 1
            rl_stats_callback.this_time_total_confidence_by_group[group] += confidence
            rl_stats_callback.this_time_valid_confidence_by_group[group] += 1

        rewards.append(score)
    
    #print("Sub-batch %d inference of %d prompt(s) completed" % (sub_batch_count, len(shortened_extracted_responses)))
    return rewards

def format_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    batch_size = len(prompts)
    graded_responses = [None for i in range(batch_size)]
    rewards = []
    for i in range(batch_size):
        graded_responses[i] = cached_parse_and_grade_response(responses[i], DATASET_NAME, answer[i], experiment_id = run_id)
        if graded_responses[i]['invalid_format']:
            rewards.append(0)
        else:
            rewards.append(FORMATTING_REWARD)
    return rewards

if __name__ == "__main__":
    # https://docs.python.org/3/library/argparse.html
    # Argparse template taken from https://github.com/zhytk/RAREval-data-processing/blob/main/few_shot.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", dest='config_yaml', type=str)
    args = parser.parse_args()
    print(args)
    
    EXPERIMENT_NAME = args.config_yaml
    
    with open(Path("config/rl/") / (EXPERIMENT_NAME + ".yaml"), 'r') as file:
        config = yaml.safe_load(file)
        print(config)
        
        LLM_SHORT_NAME = config['llm_short_name']
        ENABLE_THINKING = config['enable_thinking']
        THINKING_BUDGET = config['thinking_budget']
        USE_FORMAT_ENFORCER = config['use_format_enforcer']
        
        DATASET_NAME = config['train_dataset_name']
        REWARD_FUNCTION = config['reward_scheme']
        
        LORA_RANK = config['lora_rank']
        NUM_CPU_THREADS_PER_DEVICE = config['num_cpu_threads_per_device']
        VLLM_GPU_MEMORY_UTILIZATION = config['vllm_gpu_memory_utilization']
        
        NUM_TRAINING_BATCHES = config['num_training_batches']

        NUM_DEVICES = config['num_devices']
        RESPONSES_PER_QUESTION = config['responses_per_question']
        QUESTIONS_PER_BATCH = config['questions_per_batch']
        OPTIMIZER_STEPS_PER_GENERATION = config['optimizer_steps_per_generation']
        PER_DEVICE_TRAIN_BATCH_SIZE = config['per_device_train_batch_size']
        MAX_OUTPUT_TOKENS = config['max_output_tokens']
        UNSLOTH_GRPO_MINI_BATCH_SIZE = config['unsloth_grpo_mini_batch_size']
        UNSLOTH_MAX_CHUNK_LENGTH = config['unsloth_max_chunk_length']
        
        LEARNING_RATE = float(config['learning_rate'])
        FORMATTING_REWARD = config['formatting_reward']
        RANDOM_SEED = config['random_seed_rl']
        SMOOTHING_FACTOR_ALPHA = float(config['smoothing_factor_alpha'])
        
        USE_WANDB = config['use_wandb']
        WANDB_TEAM_NAME = config['wandb_team_name']
        WANDB_PROJECT_NAME = config['wandb_project_name']
        WANDB_RESUME_ID = config['wandb_resume_id']
        
        # These three optional parameters were only added when the first Qwen 3 (4B) HotpotQA-Modified runs have completed and the first Qwen 3 (4B) DeepMath-103K are ongoing.
        # The original experiment parameters are used as defaults if not explicitly given. This is to ensure compatibility of the configuration file with older versions.
        if 'loss_type' not in config:
            LOSS_TYPE = 'dr_grpo'
        else:
            LOSS_TYPE = config['loss_type']
        
        if 'save_batches' not in config:
            SAVE_BATCHES = 5
        else:
            SAVE_BATCHES = config['save_batches']
            
        if 'save_total_limit' not in config:
            SAVE_TOTAL_LIMIT = 3
        else:
            SAVE_TOTAL_LIMIT = config['save_total_limit']

    if "WORLD_SIZE" in os.environ:
        pytorch_world_size = int(os.environ["WORLD_SIZE"])
    else:
        pytorch_world_size = 1
    
    print("Number of GPUs (num_devices in yaml file):", NUM_DEVICES)
    print("PyTorch world size:", pytorch_world_size)
    print("Number of detected CUDA devices:", torch.cuda.device_count())
    assert NUM_DEVICES == pytorch_world_size

    # References for the equations:
    # 1. https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation
    # 2. Unsloth source code (Licensed Under Apache 2.0)
    # 3. https://huggingface.co/docs/trl/v0.29.1/en/grpo_trainer#trl.GRPOConfig
    # 4. https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/grpo-long-context
    # In our equations, we assume the use of exactly 1 GPU
    effective_batch_size = QUESTIONS_PER_BATCH * RESPONSES_PER_QUESTION
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE
    
    assert effective_batch_size % (per_device_train_batch_size * NUM_DEVICES) == 0
    steps_per_generation = effective_batch_size // per_device_train_batch_size // NUM_DEVICES
    
    assert steps_per_generation % OPTIMIZER_STEPS_PER_GENERATION == 0
    gradient_accumulation_steps = steps_per_generation // OPTIMIZER_STEPS_PER_GENERATION
    effective_training_batch_size = per_device_train_batch_size * steps_per_generation * NUM_DEVICES # 1 device for now
    effective_generation_batch_size = per_device_train_batch_size * gradient_accumulation_steps * NUM_DEVICES # 1 device for now
    
    num_training_steps = NUM_TRAINING_BATCHES * OPTIMIZER_STEPS_PER_GENERATION
    
    print("Effective Batch Size: %d" % effective_batch_size)
    print("Effective (Optimizer) Step Size: %d" % (effective_batch_size // OPTIMIZER_STEPS_PER_GENERATION))
    print("Effective Training Minibatch Size: %d" % effective_training_batch_size)
    print("Effective Generation Batch Size: %d" % effective_generation_batch_size)
    print("Training Batch(es): %d" % NUM_TRAINING_BATCHES)
    print("Training (Optimizer) Step(s): %d" % num_training_steps)

    rl_model_output_dir = "models/rl/%s" % EXPERIMENT_NAME

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
            "name": EXPERIMENT_NAME + ("_%d" % timestamp),
            # Track hyperparameters and run metadata.
            # vars() converts Namespace to dict() - according to Google Search Gemini AI
            "config": {"cmdline_args": vars(args), "config_params": config},
        }
        
        if WANDB_RESUME_ID != "" and WANDB_RESUME_ID is not None:
            rl_stats_callback.wandb_run = wandb.init(**wandb_init_kwargs, id=WANDB_RESUME_ID, resume="must", allow_val_change=False)
            resumed_run = True
        else:
            # Older versions of the code did not have this.
            # Now, wandb will allow resumption of run if resume_id is provided
            rl_stats_callback.wandb_run = wandb.init(**wandb_init_kwargs)
    
        run_id = wandb.run.id
    else:
        run_id = wandb.util.generate_id()

    # Preprocess dataset and intialize groups
    # Note: This is where the statistics are initialized
    print("Timestamp (ns): %d" % timestamp)
    print("Run ID: %s" % run_id)
    #dataset = get_dataset("datasets/rl-train/%s.csv" % model_sft_filename)
    dataset = get_dataset(get_rl_train_path(LLM_SHORT_NAME, ENABLE_THINKING, DATASET_NAME))
    rl_stats_callback.group_set = set(dataset["group"])
    rl_stats_callback.reset_current_batch_statistics()

    if resumed_run:
        print("Resuming run")

        # First, check the checkpoint index.
        rl_model_output_dir_path = Path(rl_model_output_dir)
        #print(rl_model_output_dir_path)
        checkpoint_str = "checkpoint-"
        checkpoint_indices = list(rl_model_output_dir_path.glob(checkpoint_str + "*"))
        steps_completed = -1
        for path in checkpoint_indices:
            try:
                checkpoint_index = int(path.name[len(checkpoint_str):])
                assert checkpoint_index >= 1
                steps_completed = max(checkpoint_index, steps_completed)
            except ValueError:
                print("Warning: checkpoint index invalid in %s" % path.name)
                pass

        print("Steps completed: %d" % steps_completed)

        if steps_completed == -1:
            assert False, "Checkpoint not found"

        api = wandb.Api()
        api_run = api.run("%s/%s/%s" % (WANDB_TEAM_NAME, WANDB_PROJECT_NAME, WANDB_RESUME_ID))
        history_df = api_run.history(samples=num_training_steps*1000)

        # For debugging purposes only
        DEBUG_WANDB_RESUME = True
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
        # This code assumes save_steps >= 2 batches because during much of the first step after resumption, the value train/global_step is 0.
        history_df = history_df[(history_df["train/global_step"] >= steps_completed - OPTIMIZER_STEPS_PER_GENERATION) & (history_df["train/global_step"] <= steps_completed)]
        history_df = history_df.sort_values(by='_runtime') # Therefore, train/global_step is expected to increase as runtime increases unless interrupted
        # Take only the most recent value if there is overlap in the numbers.
        for i in range(len(history_df)):
            row = history_df.iloc[i]
            step_number = row["train/global_step"]
            assert step_number == step_number # Ensure not NaN
            for column in row.keys():
                if column == "train/global_step" or column in ["_step", "_runtime", "_timestamp"]:
                    continue
                    
                # Only profiling is 0-indexed
                if not column.startswith("profiling/") and step_number == steps_completed - OPTIMIZER_STEPS_PER_GENERATION:
                    # If profiling/ is not the prefix, then the step number is 1-indexed
                    # Wrong step number
                    continue
                if column.startswith("profiling/") and step_number == steps_completed:
                    # If profiling/ is the prefix, then the step number is 0-indexed
                    # Wrong step number
                    continue

                if row[column] == row[column]: # Checks for not NaN
                    run_summary[column] = row[column]
                    print(i, step_number, column, row[column])


        rl_stats_callback.rolling_average_batch_accuracy = run_summary["batch_statistics.rolling_average_accuracy"]
        rl_stats_callback.rolling_average_batch_confidence = run_summary["batch_statistics.rolling_average_confidence"]

        rl_stats_callback.rolling_average_confidence_by_group = {}
        rl_stats_callback.rolling_average_accuracy_by_group = {}
        print(run_summary)
        for group in rl_stats_callback.group_set:
            group_statistics_name = "group_%s_statistics" % group
            group_confidence_name = group_statistics_name + ".rolling_average_confidence"
            group_accuracy_name = group_statistics_name + ".rolling_average_accuracy"
            if group_confidence_name not in run_summary.keys():
                assert group_accuracy_name not in run_summary.keys()
                rl_stats_callback.rolling_average_confidence_by_group[group] = 0.5
                rl_stats_callback.rolling_average_accuracy_by_group[group] = 0.5
                continue

            rl_stats_callback.rolling_average_confidence_by_group[group] = run_summary[group_confidence_name]
            rl_stats_callback.rolling_average_accuracy_by_group[group] = run_summary[group_accuracy_name]

        #print(run_summary.keys())
        #assert False
    else:
        rl_stats_callback.rolling_average_batch_confidence = 0.5
        rl_stats_callback.rolling_average_batch_accuracy = 0.5

        rl_stats_callback.overall_correct = 0
        rl_stats_callback.overall_wrong = 0
        rl_stats_callback.overall_valid = 0
        steps_completed = 0

        rl_stats_callback.rolling_average_confidence_by_group = {group: 0.5 for group in rl_stats_callback.group_set}
        rl_stats_callback.rolling_average_accuracy_by_group = {group: 0.5 for group in rl_stats_callback.group_set}
    
    assert steps_completed % OPTIMIZER_STEPS_PER_GENERATION == 0
    rl_stats_callback.batches_completed = steps_completed // OPTIMIZER_STEPS_PER_GENERATION
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_LONG_NAME[LLM_SHORT_NAME])

    dataset = dataset.map(find_token_length, batched=True, num_proc=NUM_CPU_THREADS_PER_DEVICE, fn_kwargs={"tokenizer": tokenizer})
    max_prompt_length = max(list(dataset["token_length"]))
    print("Longest prompt in dataset has %d token(s)." % max_prompt_length)

    unsloth_fast_inference_supported = False if LLM_SHORT_NAME.startswith("gemma4-") else True
    max_seq_length = MAX_OUTPUT_TOKENS + max_prompt_length + 5
    
    reasoning_parser = None
    if ENABLE_THINKING:
        if LLM_SHORT_NAME.startswith("qwen3-"):
            reasoning_parser = "qwen3"
        elif LLM_SHORT_NAME.startswith("gemma4-"):
            # Incomplete
            reasoning_parser = "gemma4"
        else:
            assert False, "reasoning_parser not configured"
    
    load_in_fp8 = LLM_SHORT_NAME.endswith("fp8")
    
    if model_supports_float16(LLM_SHORT_NAME):
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
        
    if is_multimodal(LLM_SHORT_NAME):
        ModelClass = FastVisionModel
    else:
        ModelClass = FastLanguageModel
    
    model, tokenizer = ModelClass.from_pretrained(
        model_name = LLM_LONG_NAME[LLM_SHORT_NAME],
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        load_in_fp8 = load_in_fp8, # Only load in FP8 for models already in FP8
        fast_inference = unsloth_fast_inference_supported, # Enable vLLM fast inference
        max_lora_rank = LORA_RANK,
        gpu_memory_utilization = VLLM_GPU_MEMORY_UTILIZATION, # Reduce if out of memory,
        dtype = dtype,
        device_map = "balanced",
        unsloth_tiled_mlp = True,
        text_only = True,
        #reasoning_config = ReasoningConfig(reasoning_parser = reasoning_parser) # not supported
    )
    
    model = ModelClass.get_peft_model(
        model,
        r = LORA_RANK,
        lora_alpha = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        use_rslora = False,
        loftq_config = None,
        random_state = RANDOM_SEED,
    )

    if not os.path.exists("models/rl"):
        os.makedirs("models/rl")
    
    # Since the optimal size of the cache depends on the effective batch size, there is no choice but to place the code here
    @lru_cache(maxsize=effective_batch_size*2, typed=True)
    def cached_parse_and_grade_response(response: str, dataset_name: str, ground_truth: str, experiment_id: str = None):
        return parse_and_grade_response(response, dataset_name, ground_truth, experiment_id)
    
    # Attempt to ensure diversity in generation: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation
    generation_kwargs = {
        "temperature": get_rl_temperature(LLM_SHORT_NAME, ENABLE_THINKING),
        "top_p": get_rl_top_p(LLM_SHORT_NAME, ENABLE_THINKING),
        "top_k": get_rl_top_k(LLM_SHORT_NAME, ENABLE_THINKING),
        "min_p": get_rl_min_p(LLM_SHORT_NAME, ENABLE_THINKING),
    }
    
    answer_format_schema = AnswerFormat.model_json_schema()
    print(answer_format_schema)
    if USE_FORMAT_ENFORCER:
        generation_kwargs["structured_outputs"] = {"json": answer_format_schema, "strict_mode": True}
    
    ''' # It looks like TRL does not support a separate thinking token budget for now because this requires --reasoning_parser from VLLM
    # https://github.com/huggingface/trl/issues/3592
    if ENABLE_THINKING: 
        generation_kwargs["thinking_token_budget"] = THINKING_BUDGET
    '''
    REWARD_CALLS_PER_GENERATION = 1
    
    training_args = GRPOConfig(
        beta = 0.0,
        use_vllm = True, # use vLLM for fast inference!
        vllm_mode = "colocate",
        vllm_gpu_memory_utilization = VLLM_GPU_MEMORY_UTILIZATION,
        vllm_max_model_length = max_seq_length,
        learning_rate = LEARNING_RATE,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        #warmup_steps = 25,
        lr_scheduler_type = "constant",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps, # Increase to 4 for smoother training
        steps_per_generation = steps_per_generation,
        num_generations = RESPONSES_PER_QUESTION, # Decrease if out of memory
        max_completion_length = MAX_OUTPUT_TOKENS,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = num_training_steps,
        # Save frequency must be at most once every two steps because program resumption logic only supports save_steps >= 2.
        # Save frequency must be aligned to the batch to ensure that partially trained batches do not get saved
        # More frequent saves to reduce lost iterations in future, older versions have save_steps=20 (batches, with 1 step per batch)
        # Update: Now, this parameter will be a configurable parameter in the configuration file, with default value commented out.
        #save_steps = 5 * OPTIMIZER_STEPS_PER_GENERATION,
        save_steps = SAVE_BATCHES * OPTIMIZER_STEPS_PER_GENERATION,
        # save_total_limit is now a configurable parameter, with default value commented out.
        #save_total_limit = 3,
        save_total_limit = SAVE_TOTAL_LIMIT,
        max_grad_norm = 0.1,
        output_dir = rl_model_output_dir,
        #loss_type = 'dr_grpo',
        loss_type = LOSS_TYPE,
        scale_rewards = False,
        mask_truncated_completions = True,
        importance_sampling_level = "sequence",
        unsloth_grpo_mini_batch = None if UNSLOTH_GRPO_MINI_BATCH_SIZE is None else (1 + (effective_generation_batch_size - 1) // UNSLOTH_GRPO_MINI_BATCH_SIZE),
        # Bug 1: unsloth_logit_chunk_multiplier is not calculated correctly. Only affects the memory usage in the implementation. 
        # A change in unsloth_logit_chunk_multiplier is not expected to affect the output as it only controls the chunking of logits.
        # Bug 1 is only discovered while testing with Gemma 4 (E2B) Instruct with DeepMath-103K dataset where there was a CUDA out of memory error.
        # unsloth_logit_chunk_multiplier is found to be 1 despite the output length limit.
        # At that time, Qwen 3 (4B) RL training with HotpotQA Modified (first set of experiments) is already underway, but the output is not expected to be affected by the bug.
        # Qwen 3 (4B) RL training with HotpotQA Modified is allowed to proceed with the bug but subsequent experiments will run with the bug fixed.
        # Buggy line has been commented out.
        # unsloth_logit_chunk_multiplier = None if UNSLOTH_MAX_CHUNK_LENGTH is None else (1 + (max_prompt_length + 4) // UNSLOTH_MAX_CHUNK_LENGTH),
        unsloth_logit_chunk_multiplier = None if UNSLOTH_MAX_CHUNK_LENGTH is None else (1 + (max_seq_length + 4) // UNSLOTH_MAX_CHUNK_LENGTH),
        # wandb integration
        report_to = "wandb" if USE_WANDB else "none",
        run_name = "rl_confidence_calibration-" + run_id,
        generation_kwargs = generation_kwargs,
        # Bug 2: vllm_enable_sleep_mode = True is not supported while using Unsloth (affects Gemma 4 (E2B) Instruct but not Qwen 3 (4B))
        # As a result of the bug, Gemma 4 (E2B) Instruct inference appears to be taken from the base model, causing the training to run off-policy.
        # Bug 2 is only discovered while testing with Gemma 4 (E2B) Instruct with HotpotQA-Modified dataset where training performance remained the same regardless of the reward scheme. 
        # At that time, Qwen 3 (4B) RL training with HotpotQA Modified (first set of experiments) is already underway, but these experiments are unaffected by the bug.
        # Qwen 3 (4B) RL training with HotpotQA Modified is allowed to proceed with the bug (unaffected by the bug) but subsequent experiments will run with the bug fixed.
        # Buggy line has been commented out.
        #vllm_enable_sleep_mode = False if unsloth_fast_inference_supported else True,
        vllm_enable_sleep_mode = False,
    )
    print(training_args)

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [correctness_reward_func, format_reward_func],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.add_callback(rl_stats_callback)
    trainer.train(resume_from_checkpoint = resumed_run)

    if USE_WANDB:
        rl_stats_callback.wandb_run.finish()

