# Google Search Gemini AI is used to make part of the code
# Finetuning code taken from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb#scrollTo=6bZsfBuZDeCL
# SFT finetuning code references https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only

import os
import psutil
import json
import argparse

#from unsloth import FastLanguageModel

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.sampling_params import StructuredOutputsParams
from typing import Union, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from utils import LLM_LONG_NAME, derived_model_name_sft

import random
import torch
import pandas as pd

LLM_SHORT_NAME = "qwen2.5-3b"
DATASET_NAME = "multi-armed-bandit-64"
LORA_RANK = 32
LLM_CONFIDENCE = True
CONFIDENCE_ANALYSIS = True

if __name__ == '__main__':
    random.seed(856506232)
    
    # https://docs.python.org/3/library/argparse.html
    # Argparse template taken from https://github.com/zhytk/RAREval-data-processing/blob/main/few_shot.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME, dest='dataset_name', type=str)
    parser.add_argument("--llm", default=LLM_SHORT_NAME, dest='llm_short_name', type=str)
    parser.add_argument("--lora_rank", default=LORA_RANK, dest='lora_rank', type=int)
    parser.add_argument("--llm_confidence", default=LLM_CONFIDENCE, dest='llm_confidence', action=argparse.BooleanOptionalAction)
    parser.add_argument("--confidence_analysis", default=CONFIDENCE_ANALYSIS, dest='confidence_analysis', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print(args)
    
    DATASET_NAME = args.dataset_name
    LLM_SHORT_NAME = args.llm_short_name
    LORA_RANK = args.lora_rank
    LLM_CONFIDENCE = args.llm_confidence
    CONFIDENCE_ANALYSIS = args.confidence_analysis
    
    model_filename = derived_model_name_sft(DATASET_NAME, LLM_SHORT_NAME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)
    df = pd.read_csv('datasets/sft-train/%s.csv' % model_filename)
    
    model_name = LLM_LONG_NAME[LLM_SHORT_NAME]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    labels = []
    delete_indices = []
    for i in range(len(df)):
        row = df.iloc[i]
        label = "<reasoning>\n" + str(row["llm_reasoning"]) + "\n</reasoning>\n<answer>\n" + str(row["llm_answer"]) + "\n</answer>\n"
        if CONFIDENCE_ANALYSIS:
            label += "<confidence_analysis>\n" + str(row["confidence_analysis"]) + "\n</confidence_analysis>\n"
        label += "<confidence>\n" + str(row["confidence"]) + "\n</confidence>"
        #print(label)
        #if i == 3:
        #    assert False
        labels.append(label)

    df["response"] = labels
    dataset = Dataset.from_pandas(df)
    
    def preprocess_function(example):
        template = {
            "prompt": eval(example["prompt_chat_template"]), # Insecure but no choice due to the chat format dump in single quotes
            "completion": [{"role": "assistant", "content": example["response"]}],
        }
        #print(template)
        #assert False
        #return {"messages": template}
        return template

    cores = psutil.cpu_count(logical=False)
    print("Using %d CPU core(s) to preprocess the dataset" % cores)
    #print(list(df.columns))
    dataset = dataset.map(preprocess_function, num_proc=cores, remove_columns=list(df.columns))
    #print(dataset)
    
    #for i in range(5):
    #    print(dataset[i])
    #assert False
    
    # We need some kind of creativity in response generation so that more confidence levels are explored.
    
    training_args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        warmup_steps = 5,
        #max_steps = 128,
        num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        #assistant_only_loss=True,
        packing = False, # Can make training 5x faster for short sequences.
        gradient_checkpointing=True,
        completion_only_loss=True,
        max_length=None,
    )
    
    if not os.path.exists("models/sft"):
        os.makedirs("models/sft")

    # Hugging Face model code start
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, peft_config)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Hugging Face model code ends
    
    trainer = SFTTrainer(
        model = model,
        #tokenizer = tokenizer,
        # https://huggingface.co/google/gemma-3-1b-it/discussions/21
        processing_class = tokenizer if LLM_SHORT_NAME in ["gemma3-1b"] else None, 
        train_dataset = dataset,
        args = training_args,
    )
    
    trainer.train()
    
    model.save_pretrained("models/sft/%s" % model_filename, from_pt=True)
