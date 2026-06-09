# Google Search Gemini AI is used to make part of the code

import os
import argparse

# Helps ensure reproducibility: https://discuss.vllm.ai/t/two-different-runs-give-different-answers/2025
# But still need to ensure library version consistency and same hardware
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

for path in ["datasets/sft-train", "datasets/rl-train"]:
    if not os.path.exists(path):
        os.mkdir(path)

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.sampling_params import StructuredOutputsParams
from typing import Union, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from math_verify import parse, verify
from utils import verify_correctness, LLM_LONG_NAME, system_prompt_rl, system_prompt_preprocess, derived_model_name_sft

import random
import torch
import pandas as pd
import json

MAX_OUTPUT_TOKENS = 1024
LLM_SHORT_NAME = "qwen2.5-3b"
LLM_CONFIDENCE = True
CONFIDENCE_ANALYSIS = True
PREPROCESSING_SAMPLE_SIZE = 1 << 10
#M = 1<<10 # Subset of samples to generate for finetuning
#M = 32 # For debugging

DATASET_NAME = "multi-armed-bandit-64"

# Attempts to restore the original string by replacing characters such as \t with their escaped versions.
def add_escape_characters(answer: str) -> str:
    escape_characters = ['\a', '\r', '\t', '\b', '\f', '\v']
    replacement_characters = ['\\a', '\\r', '\\t', '\\b', '\\f', '\\v']
    assert len(escape_characters) == len(replacement_characters)
    
    for i in range(len(escape_characters)):
        answer = answer.replace(escape_characters[i], replacement_characters[i])
    return answer

if __name__ == '__main__':
    random.seed(856506232)
    
    # https://docs.python.org/3/library/argparse.html
    # Argparse template taken from https://github.com/zhytk/RAREval-data-processing/blob/main/few_shot.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME, dest='dataset_name', type=str)
    parser.add_argument("--llm", default=LLM_SHORT_NAME, dest='llm_short_name', type=str)
    parser.add_argument("--max_output_tokens", default=MAX_OUTPUT_TOKENS, dest='max_output_tokens', type=int)
    parser.add_argument("--llm_confidence", default=LLM_CONFIDENCE, dest='llm_confidence', action=argparse.BooleanOptionalAction)
    parser.add_argument("--confidence_analysis", default=CONFIDENCE_ANALYSIS, dest='confidence_analysis', action=argparse.BooleanOptionalAction)
    parser.add_argument("--preprocessing_sample_size", default=PREPROCESSING_SAMPLE_SIZE, dest='preprocessing_sample_size', type=int)
    args = parser.parse_args()
    print(args)
    
    DATASET_NAME = args.dataset_name
    LLM_SHORT_NAME = args.llm_short_name
    MAX_OUTPUT_TOKENS = args.max_output_tokens
    LLM_CONFIDENCE = args.llm_confidence
    CONFIDENCE_ANALYSIS = args.confidence_analysis
    PREPROCESSING_SAMPLE_SIZE = args.preprocessing_sample_size
    M = PREPROCESSING_SAMPLE_SIZE
    
    dataset = pd.read_csv("datasets/train/%s.csv" % DATASET_NAME)
    
    # Shuffle pandas dataset: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    dataset = dataset.sample(frac=1, random_state=574136846).reset_index(drop=True)
    
    N = len(dataset)
    
    model_name = LLM_LONG_NAME[LLM_SHORT_NAME]
    tokenizer = get_tokenizer(model_name)
    
    class AnswerFormat(BaseModel):
        reasoning: str
        
        if DATASET_NAME == "addition":
            answer: int = Field(ge=0)
        elif DATASET_NAME == "multi-armed-bandit-22222":
            answer: int = Field(ge=0, le=5)
        elif DATASET_NAME == "multi-armed-bandit-64":
            answer: int = Field(ge=0, le=2)
        elif DATASET_NAME == "multi-armed-bandit-82":
            answer: int = Field(ge=0, le=2)
        elif DATASET_NAME in ["hotpotqa", "hotpotqa-modified", "deepmath-103k", "bigmath"]:
            answer: str = Field(max_length=1000)
        elif DATASET_NAME in ["noisy-ground-truth-sequential", "noisy-ground-truth-random"]:
            answer: int = Field(ge=0, le=999)
        else:
            assert False
        
        if CONFIDENCE_ANALYSIS:
            confidence_analysis: str
        
        if LLM_CONFIDENCE:
            confidence: int = Field(ge=0, le=100)
        
        model_config = ConfigDict(extra='forbid', strict=True)
    
    answer_format_schema = AnswerFormat.model_json_schema()
    print(answer_format_schema)
    structured_outputs_params = StructuredOutputsParams(json=answer_format_schema, strict_mode=True)

    chat_templates = []
    rl_chat_templates = []
    ground_truths = []
    groups = []
    for i in range(N):
        question = dataset.iloc[i]["question"]

        if i < M:
            chat_templates.append([
                {"role" : "system", "content" : system_prompt_preprocess(LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)},
                {"role" : "user", "content" : question},
            ])
        
        rl_chat_templates.append([
            {"role" : "system", "content" : system_prompt_rl(CONFIDENCE_ANALYSIS)},
            {"role" : "user", "content" : question},
        ])
        ground_truths.append(dataset.iloc[i]["ground_truth"])
        groups.append(dataset.iloc[i]["group"])
        
    # Get tokenizer statistics - find longest token length (with help of Google Search Gemini AI)
    prompt_tokens = tokenizer.apply_chat_template(chat_templates, tokenize = True, add_generation_prompt = True, return_tensors=None)
    #print(prompt_tokens)
    token_lengths = [len(tokens) for tokens in prompt_tokens]
    max_prompt_length = max(token_lengths)
    print("Longest question in finetuning dataset (subset of training dataset) has %d token(s)." % max_prompt_length)
    #assert False
    
    max_model_length = MAX_OUTPUT_TOKENS + max_prompt_length + 5 # Add 5 for a small buffer
    sampling_params = SamplingParams(
        structured_outputs = structured_outputs_params,
        # "maximum number of generated tokens per output sequence"
        # according to documentation from https://docs.vllm.ai/en/latest/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.logprobs
        max_tokens = MAX_OUTPUT_TOKENS, 
    )

    rl_prompts = tokenizer.apply_chat_template(rl_chat_templates, tokenize = False, add_generation_prompt = True)
    
    assert len(rl_prompts) == N
    assert len(ground_truths) == N
    assert len(groups) == N
    
    entire_rl_dataset = {
        'prompt_string': rl_prompts,
        'ground_truth': ground_truths,
        'group': groups,
    }
    
    rl_df = pd.DataFrame(entire_rl_dataset)
    
    # LLM preprocessing for finetuning
    prompts = tokenizer.apply_chat_template(chat_templates, tokenize = False, add_generation_prompt = True)
    llm = LLM(model=model_name, max_model_len=max_model_length, seed=421570851, gpu_memory_utilization=0.8)
    outputs = llm.generate(prompts[:M], sampling_params=sampling_params)
    
    assert len(chat_templates) == M
    assert len(outputs) == M
    
    group_set = set(groups)
    group_counts = {group: 0 for group in group_set}
    group_correct = {group: 0 for group in group_set}
    filtered_outputs = {
        'prompt_chat_template': [],
        'prompt_string': [],
        'llm_reasoning': [],
        'llm_answer': [],
        'ground_truth': [],
        'confidence': [],
        'group': [],
    }
    
    if CONFIDENCE_ANALYSIS:
        filtered_outputs['confidence_analysis'] = []
    
    for i in range(M):
        response = outputs[i].outputs[0].text
        if i < 5:
            print(response)
            print("---------------------------------------------------------------")
        try:
            # This line is intentionally before the json.loads so that formatting issues will be counted as a wrong answer for the group.
            group_counts[groups[i]] += 1
            
            response_json = json.loads(response)
            
            # JSON loads unescapes string, hence we have to add back the escape characters
            response_json['reasoning'] = add_escape_characters(str(response_json['reasoning']))
            response_json['answer'] = add_escape_characters(str(response_json['answer']))
            
            if verify_correctness(DATASET_NAME, str(response_json['answer']), str(ground_truths[i])):
                group_correct[groups[i]] += 1
            
            # Gets rid of UTF-8 unprintable characters error
            # https://stackoverflow.com/questions/27366479/python-3-os-walk-file-paths-unicodeencodeerror-utf-8-codec-cant-encode-s
            # surrogateescape did not work, hence I changed it to ignore when there is an error
            response_json['reasoning'] = response_json['reasoning'].encode('utf8', errors='ignore').decode('ISO-8859-1')
            response_json['answer'] = response_json['answer'].encode('utf8', errors='ignore').decode('ISO-8859-1')
            
            if LLM_CONFIDENCE:
                try:
                    confidence = response_json['confidence']
                    if confidence < 0 or confidence > 100 or confidence != confidence:
                        raise ValueError("confidence value out of range")
                except ValueError:
                    print("Warning: confidence is not valid for response %d - random confidence selected instead" % i)
                    confidence = random.randint(0, 100)
            else:
                confidence = random.randint(0, 100)
            
            if CONFIDENCE_ANALYSIS:
                response_json['confidence_analysis'] = response_json['confidence_analysis'].encode('utf8', errors='ignore').decode('ISO-8859-1')
                filtered_outputs['confidence_analysis'].append(response_json['confidence_analysis'])
        
            filtered_outputs['prompt_chat_template'].append(rl_chat_templates[i])
            filtered_outputs['prompt_string'].append(rl_prompts[i])
            filtered_outputs['llm_reasoning'].append(response_json['reasoning'])
            filtered_outputs['llm_answer'].append(response_json['answer'])
            filtered_outputs['ground_truth'].append(ground_truths[i])
            filtered_outputs['confidence'].append(confidence)
            filtered_outputs['group'].append(groups[i])
        except json.decoder.JSONDecodeError:
            print("Warning: output is not valid json for response %d" % i)
    
    finetuning_df = pd.DataFrame(filtered_outputs)
    print(filtered_outputs['llm_reasoning'][0])
    print(filtered_outputs['llm_answer'][0])
    
    print(finetuning_df)
    print(finetuning_df.iloc[0])
    
    print("%d row(s) generated for finetuning" % len(finetuning_df))
    print("%d row(s) generated for reinforcement learning" % len(rl_df))
    
    for group in group_set:
        if group_counts[group] != 0:
            print("Group %s: %d out of %d correct (%.1f%%)" % (group, group_correct[group], group_counts[group], 100.0 * group_correct[group] / group_counts[group]))

    # Print the outputs
    '''
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    '''
    
    derived_model_file_name = derived_model_name_sft(DATASET_NAME, LLM_SHORT_NAME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)
    finetuning_df.to_csv('datasets/sft-train/%s.csv' % derived_model_file_name)
    rl_df.to_csv('datasets/rl-train/%s.csv' % derived_model_file_name)
