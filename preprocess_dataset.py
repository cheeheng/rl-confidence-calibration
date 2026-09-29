# Google Search Gemini AI is used to make part of the code

import os
import argparse

# Helps ensure reproducibility: https://discuss.vllm.ai/t/two-different-runs-give-different-answers/2025
# But still need to ensure library version consistency and same hardware
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from pathlib import Path
from vllm.tokenizers import get_tokenizer
from typing import Union, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from utils import LLM_LONG_NAME, get_system_prompt, get_rl_train_path

import random
import pandas as pd

LLM_SHORT_NAME = "qwen3-0.6b"
DATASET_NAME = "bigmath"
ENABLE_THINKING = False

if __name__ == '__main__':
    random.seed(856506232)
    # https://docs.python.org/3/library/argparse.html
    # Argparse template taken from https://github.com/zhytk/RAREval-data-processing/blob/main/few_shot.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME, dest='dataset_name', type=str)
    parser.add_argument("--llm", default=LLM_SHORT_NAME, dest='llm_short_name', type=str)
    parser.add_argument("--enable_thinking", default=ENABLE_THINKING, dest='enable_thinking', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print(args)
    
    DATASET_NAME = args.dataset_name
    LLM_SHORT_NAME = args.llm_short_name
    ENABLE_THINKING = args.enable_thinking
    
    dataset = pd.read_csv("datasets/train/%s.csv" % DATASET_NAME)
    
    # Shuffle pandas dataset: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    dataset = dataset.sample(frac=1, random_state=574136846).reset_index(drop=True)
    
    N = len(dataset)
    
    model_name = LLM_LONG_NAME[LLM_SHORT_NAME]
    tokenizer = get_tokenizer(model_name)

    rl_chat_templates = []
    ground_truths = []
    groups = []
    for i in range(N):
        question = dataset.iloc[i]["question"]

        rl_chat_templates.append([
            {"role" : "system", "content" : get_system_prompt(ENABLE_THINKING)},
            {"role" : "user", "content" : question},
        ])
        ground_truths.append(dataset.iloc[i]["ground_truth"])
        groups.append(dataset.iloc[i]["group"])

    rl_prompts = tokenizer.apply_chat_template(rl_chat_templates, tokenize = False, add_generation_prompt = True, enable_thinking = ENABLE_THINKING)
    
    assert len(rl_prompts) == N
    assert len(ground_truths) == N
    assert len(groups) == N
    
    entire_rl_dataset = {
        'prompt_string': rl_prompts,
        'ground_truth': ground_truths,
        'group': groups,
    }
    
    rl_df = pd.DataFrame(entire_rl_dataset)
    
    rl_train_path = Path(get_rl_train_path(LLM_SHORT_NAME, ENABLE_THINKING, DATASET_NAME))
    if not os.path.exists(rl_train_path.parent):
        os.makedirs(rl_train_path.parent)
    rl_df.to_csv(rl_train_path)

