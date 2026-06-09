# Google Search Gemini AI is used to create part of the code
import os
from unsloth import FastLanguageModel
import secrets

# Attempts to help ensure reproducibility: https://discuss.vllm.ai/t/two-different-runs-give-different-answers/2025
# But still need to ensure library version consistency and same hardware
# The results remain non-deterministic when LoRA adapters are used
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# https://github.com/unslothai/unsloth-zoo/blob/2a80d543b9e22f68e051e32029c8a47005102895/unsloth_zoo/vllm_utils.py#L20
# Override due to occasional OOM
os.environ["UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE"] = "1"

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.sampling_params import StructuredOutputsParams
from utils import verify_correctness, LLM_LONG_NAME, system_prompt_rl, rl_format_regex, system_prompt_preprocess
from utils import derived_model_name_rl, derived_model_name_sft, has_valid_xml_tag, add_escape_characters
from datasets import load_dataset
from pathlib import Path
from transformers import TextStreamer
from typing import Union, List, Optional
from pydantic import BaseModel, ConfigDict, Field

import json
import argparse
import re

from utils import extract_xml_tag, find_token_length, confidence_is_valid

MAX_OUTPUT_TOKENS = 1024 # to follow RL max token limit
LORA_RANK = 32

LLM_SHORT_NAME = "qwen2.5-3b"
DATASET_NAME = "multi-armed-bandit-64"
REWARD_SCHEME = "brier-1"
LLM_CONFIDENCE = True
CONFIDENCE_ANALYSIS = True
USE_MODEL = "rl"
USE_UNSLOTH = False
RUN_SUFFIX = ""
RESPONSES_PER_QUESTION = 1
SECOND_CHANCE_ANSWER_TOKEN_LIMIT = 64
USE_JSON = False
SAMPLE_SIZE = -1

NUM_CPU_THREADS = 8

def normalize_confidence(confidence):
    return (confidence + 0.5) / 101

def generate_outputs(llm, prompts, sampling_params, lora_request):
    if USE_UNSLOTH:
        outputs = llm.fast_generate(prompts, sampling_params=sampling_params, lora_request = lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request = lora_request)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME, dest='dataset_name', type=str)
    parser.add_argument("--llm", default=LLM_SHORT_NAME, dest='llm_short_name', type=str)
    parser.add_argument("--max_output_tokens", default=MAX_OUTPUT_TOKENS, dest='max_output_tokens', type=int)
    parser.add_argument("--lora_rank", default=LORA_RANK, dest='lora_rank', type=int)
    parser.add_argument("--num_cpu_threads", default=NUM_CPU_THREADS, dest='num_cpu_threads', type=int)
    parser.add_argument("--reward_scheme", default=REWARD_SCHEME, dest='reward_scheme', type=str)
    parser.add_argument("--run_suffix", default=RUN_SUFFIX, dest='run_suffix', type=str)
    #parser.add_argument("--llm_confidence", default=LLM_CONFIDENCE, dest='llm_confidence', action=argparse.BooleanOptionalAction)
    parser.add_argument("--confidence_analysis", default=CONFIDENCE_ANALYSIS, dest='confidence_analysis', action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_model", default=USE_MODEL, dest='use_model', type=str)
    parser.add_argument("--use_unsloth", default=USE_UNSLOTH, dest='use_unsloth', action=argparse.BooleanOptionalAction)
    parser.add_argument("--responses_per_question", default=RESPONSES_PER_QUESTION, dest='responses_per_question', type=int)
    parser.add_argument("--second_chance_answer_token_limit", default=SECOND_CHANCE_ANSWER_TOKEN_LIMIT, 
        dest="second_chance_answer_token_limit", type=int)
    parser.add_argument("--use_json", default=USE_JSON, dest='use_json', action=argparse.BooleanOptionalAction)
    parser.add_argument("--sample_size", default=SAMPLE_SIZE, dest='sample_size', type=int)
    args = parser.parse_args()
    print(args)
    
    DATASET_NAME = args.dataset_name
    LLM_SHORT_NAME = args.llm_short_name
    MAX_OUTPUT_TOKENS = args.max_output_tokens
    LORA_RANK = args.lora_rank
    NUM_CPU_THREADS = args.num_cpu_threads
    REWARD_SCHEME = args.reward_scheme
    RUN_SUFFIX = args.run_suffix
    #LLM_CONFIDENCE = args.llm_confidence
    CONFIDENCE_ANALYSIS = args.confidence_analysis
    USE_MODEL = args.use_model
    USE_UNSLOTH = args.use_unsloth
    RESPONSES_PER_QUESTION = args.responses_per_question
    SECOND_CHANCE_ANSWER_TOKEN_LIMIT = args.second_chance_answer_token_limit
    USE_JSON = args.use_json
    SAMPLE_SIZE = args.sample_size

    assert RUN_SUFFIX.strip() not in ["base", "rl", "sft"], "--run_suffix cannot be base, rl or sft because it may cause confusion"
    
    if USE_JSON:
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
    else:
        structured_outputs_params = StructuredOutputsParams(regex=rl_format_regex(CONFIDENCE_ANALYSIS), strict_mode=True)
    
    SECOND_CHANCE_ANSWER_STRING_PREFIX = "Final Answer:"
    structured_outputs_answer_params = StructuredOutputsParams(regex="^%s.*" % SECOND_CHANCE_ANSWER_STRING_PREFIX, strict_mode=True)
    structured_outputs_confidence_params = StructuredOutputsParams(choice=[str(i) for i in range(101)], strict_mode=True)

    model_name = LLM_LONG_NAME[LLM_SHORT_NAME]
    tokenizer = get_tokenizer(model_name)
    
    dataset_filename = "datasets/test/%s.csv" % DATASET_NAME
    dataset = load_dataset("csv", data_files=dataset_filename)["train"]
    
    system_prompt = system_prompt_preprocess(LLM_CONFIDENCE, CONFIDENCE_ANALYSIS) if USE_JSON else system_prompt_rl(CONFIDENCE_ANALYSIS)
    def generate_prompt(entry, tokenizer):
        entry["prompt"] = [
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : entry["question"]},
        ]
        entry["prompt"] = tokenizer.apply_chat_template(entry["prompt"], tokenize = False, add_generation_prompt = True)
        #print(entry)
        #assert False
        return entry
    
    N_test = len(dataset)
    
    dataset = dataset.map(generate_prompt, batched=False, num_proc=NUM_CPU_THREADS, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.map(find_token_length, batched=True, num_proc=NUM_CPU_THREADS, fn_kwargs={"tokenizer": tokenizer})
    max_prompt_length = max(list(dataset["token_length"]))
    print("Longest prompt in test dataset has %d token(s)." % max_prompt_length)
    max_model_length = MAX_OUTPUT_TOKENS + max_prompt_length + SECOND_CHANCE_ANSWER_TOKEN_LIMIT + 1024

    chat_templates = []
    if SAMPLE_SIZE != -1:
        N_test = min(N_test, SAMPLE_SIZE)

    for i in range(N_test):
        for j in range(RESPONSES_PER_QUESTION):
            chat_templates.append([
                {"role" : "system", "content" : system_prompt},
                {"role" : "user", "content" : dataset["question"][i]},
            ])
            #print(chat_templates[-1])
            #assert False
    
    sampling_params = SamplingParams(
        temperature = 1.0,
        structured_outputs = structured_outputs_params,
        # "maximum number of generated tokens per output sequence"
        # according to documentation from https://docs.vllm.ai/en/latest/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.logprobs
        max_tokens = MAX_OUTPUT_TOKENS, 
    )

    sampling_params_answer = SamplingParams(
        temperature = 1.0,
        structured_outputs = structured_outputs_answer_params,
        # "maximum number of generated tokens per output sequence"
        # according to documentation from https://docs.vllm.ai/en/latest/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.logprobs
        max_tokens = SECOND_CHANCE_ANSWER_TOKEN_LIMIT, 
    )

    sampling_params_confidence = SamplingParams(
        temperature = 1.0,
        structured_outputs = structured_outputs_confidence_params,
    )

    prompts = tokenizer.apply_chat_template(chat_templates, tokenize = False, add_generation_prompt = True)
    
    if USE_MODEL == "base":
        # _base at the end of the filename signifies base model
        model_rl_filename = derived_model_name_sft(DATASET_NAME, LLM_SHORT_NAME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS) + "_base" 
        print("Loading base model %s" % LLM_LONG_NAME[LLM_SHORT_NAME])
    elif USE_MODEL == "sft":
        # _sft at the end of the filename signifies sft model
        model_sft_filename = derived_model_name_sft(DATASET_NAME, LLM_SHORT_NAME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)
        model_rl_filename = model_sft_filename + "_sft" 
        lora_model_dir = "models/sft/%s" % model_sft_filename
        print("Loading supervised finetuning model %s" % lora_model_dir)
    elif USE_MODEL == "rl":
        model_rl_filename = derived_model_name_rl(DATASET_NAME, LLM_SHORT_NAME, REWARD_SCHEME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)

        lora_model_parent_dir = "models/rl/%s/" % model_rl_filename
        checkpoint_index = 0
        found_directories = [p for p in Path(lora_model_parent_dir).glob("checkpoint-*") if p.is_dir()]
        for directory in found_directories:
            directory_checkpoint_index = str(directory)[len(lora_model_parent_dir + "checkpoint-"):]
            try:
                #print(str(directory), directory_checkpoint_index)
                directory_checkpoint_index = int(directory_checkpoint_index)
                checkpoint_index = max(checkpoint_index, directory_checkpoint_index)
            except ValueError:
                pass

        lora_model_dir = lora_model_parent_dir + "checkpoint-" + str(checkpoint_index)
        print("Loading from LoRA model file %s" % lora_model_dir)

        if RUN_SUFFIX != "":
            model_rl_filename += "_" + str(RUN_SUFFIX)
    else:
        assert False, "--use_model must be either base, sft or rl"

    # Output file name - also used for experiment id to avoid race conditions
    if USE_MODEL in ["base", "sft"]:
        output_filename = derived_model_name_sft(DATASET_NAME, LLM_SHORT_NAME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS) + "_" + USE_MODEL 
    elif USE_MODEL == "rl":
        output_filename = derived_model_name_rl(DATASET_NAME, LLM_SHORT_NAME, REWARD_SCHEME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)
    
    if RUN_SUFFIX != "":
        output_filename += "_" + str(RUN_SUFFIX)

    experiment_id = "eval_" + output_filename
    
    # Check if unsloth works here
    # The LLM inference code is modified from https://github.com/unslothai/unsloth/issues/2551

    base_model_name = LLM_LONG_NAME[LLM_SHORT_NAME]

    if USE_UNSLOTH:
        llm, _ = FastLanguageModel.from_pretrained(
            model_name = base_model_name,
            max_seq_length = max_model_length,
            seed = 467915983,
            gpu_memory_utilization=0.8,
            fast_inference = True, # Uses vLLM
            load_in_4bit = False,
            load_in_8bit = False,
            enable_lora = False if USE_MODEL == "base" else True,
            max_lora_rank = LORA_RANK)
        FastLanguageModel.for_inference(llm)
    else:
        llm = LLM(model=base_model_name, 
            max_model_len=max_model_length, 
            seed = 905244229, 
            gpu_memory_utilization=0.8,
            enable_lora = False if USE_MODEL == "base" else True,
            max_lora_rank = LORA_RANK)

    lora_request = None if USE_MODEL == "base" else LoRARequest(lora_name="lora_adapter", lora_int_id=1+secrets.randbelow((1<<63) - 1), lora_path=lora_model_dir)

    outputs = generate_outputs(llm, prompts, sampling_params, lora_request)
    
    assert len(chat_templates) == N_test * RESPONSES_PER_QUESTION
    
    groups = list(set(dataset["group"]))
    print("List of groups:", groups)
    
    ALL_GROUPS = "overall"
    assert ALL_GROUPS not in groups
    groups.append(ALL_GROUPS)

    assert "metadata" not in groups
    group_statistics = {}
    group_statistics["metadata"] = vars(args)
    group_statistics["metadata"]["num_questions"] = N_test
    for group in groups:
        group_statistics[group] = {'is_correct': [], 'confidences': [], 
            'invalid_counts': {'confidence': 0, 'answer': 0, 'format': 0}}
    
    print("Sanity check (first 10 responses)")
    rerun_answer_idx = []
    rerun_confidence_idx = []
    for i in range(len(chat_templates)):
        #print("Output %d" % i)
        response = outputs[i].outputs[0].text
        #print(response)
        
        group = dataset[i//RESPONSES_PER_QUESTION]["group"]
        ground_truth = str(dataset[i//RESPONSES_PER_QUESTION]["ground_truth"])

        group_idx = len(group_statistics[group]['is_correct'])
        assert group_idx == len(group_statistics[group]['confidences'])

        if USE_JSON:
            try:
                response_json = json.loads(response)
                format_is_valid = True

                response_json['reasoning'] = add_escape_characters(str(response_json['reasoning']))
                response_json['answer'] = add_escape_characters(str(response_json['answer']))
                answer = response_json['answer']
                answer_is_valid = True

                confidence = response_json['confidence']
                valid_confidence_output = confidence_is_valid(confidence)
            except json.decoder.JSONDecodeError:
                answer = "" # Invalid answer
                answer_is_valid = False
                valid_confidence_output = False
                format_is_valid = False
        else:
            format_is_valid = re.fullmatch(rl_format_regex(CONFIDENCE_ANALYSIS), response)

            answer = extract_xml_tag(response, "answer")
            confidence = extract_xml_tag(response, "confidence")
            
            answer_is_valid = has_valid_xml_tag(response, "answer")
            valid_confidence_output = confidence_is_valid(confidence) and has_valid_xml_tag(response, "confidence")
        
        if not format_is_valid:
            group_statistics[group]['invalid_counts']['format'] += 1
            group_statistics[ALL_GROUPS]['invalid_counts']['format'] += 1
        
        if not answer_is_valid:
            group_statistics[group]['invalid_counts']['answer'] += 1
            group_statistics[ALL_GROUPS]['invalid_counts']['answer'] += 1
            rerun_answer_idx.append((i, group_idx))
            answer_is_correct = 0
        else:
            answer_is_correct = 1 if verify_correctness(DATASET_NAME, answer, ground_truth, experiment_id) else 0
        
        if valid_confidence_output:
            confidence = float(confidence)
        else:
            # assume the worst for now
            confidence = 0 if answer_is_correct else 100
            group_statistics[group]['invalid_counts']['confidence'] += 1
            group_statistics[ALL_GROUPS]['invalid_counts']['confidence'] += 1
            rerun_confidence_idx.append((i, group_idx))
        
        if i < 10:
            print("Group:", group)
            print("Question:", dataset[i//RESPONSES_PER_QUESTION]["question"])
            print("Answer:", answer)
            print("Ground Truth:", ground_truth)
            print("Confidence:", confidence)
            #print("Response:", response)
            print("Verdict:", "Correct" if answer_is_correct else "Wrong")
            print()
            
        normalized_confidence = normalize_confidence(confidence)
        group_statistics[group]['is_correct'].append(answer_is_correct)
        group_statistics[ALL_GROUPS]['is_correct'].append(answer_is_correct)
        group_statistics[group]['confidences'].append(normalized_confidence)
        group_statistics[ALL_GROUPS]['confidences'].append(normalized_confidence)

        chat_templates[i].append({"role" : "assistant", "content" : response})
        
        #print("Answer is correct: %s" % ("Yes" if answer_is_correct else "No"))
        #print("Normalized confidence: %.4f" % normalized_confidence)
        #print()
    
    REASK_ANSWER_PROMPT = "Reasoning token limit reached. Please output only your final answer within %d tokens." % SECOND_CHANCE_ANSWER_TOKEN_LIMIT
    if DATASET_NAME in ["bigmath", "deepmath-103k"]:
        REASK_ANSWER_PROMPT += " Express your answer in LaTeX."

    REASK_CONFIDENCE_PROMPT = "Please output your confidence as an integer between 0 and 100 inclusive."

    print("Some outputs may not have the correct format. Therefore, the LLM is tasked to ask for the answer and the confidence if applicable.")
    if len(rerun_answer_idx) > 0:
        print("Asking LLM for answers when original answer is invalid")
        chat_templates_answer = []
        for overall_idx, group_idx in rerun_answer_idx:
            chat_templates[overall_idx].append({"role" : "user", "content" : REASK_ANSWER_PROMPT})
            assert chat_templates[overall_idx][0]["role"] == "system"
            chat_templates_answer.append(chat_templates[overall_idx][1:])
            #chat_templates_answer.append(chat_templates[overall_idx])
        
        prompts_answer = tokenizer.apply_chat_template(chat_templates_answer, tokenize = False, add_generation_prompt = True)
        outputs_answer = generate_outputs(llm, prompts_answer, sampling_params_answer, lora_request)

        for i in range(len(chat_templates_answer)):
            overall_idx, group_idx = rerun_answer_idx[i]
            group = dataset[overall_idx//RESPONSES_PER_QUESTION]["group"]
            ground_truth = dataset[overall_idx//RESPONSES_PER_QUESTION]["ground_truth"]

            response = outputs_answer[i].outputs[0].text
            answer = response[len(SECOND_CHANCE_ANSWER_STRING_PREFIX):]

            # Re-evaluate answer
            answer_is_correct = 1 if verify_correctness(DATASET_NAME, answer, ground_truth, experiment_id) else 0
            if i < 5:
                print("Indices: (%d, %d, %d)" % (overall_idx, group_idx, overall_idx//RESPONSES_PER_QUESTION))
                print("Question: ", dataset[overall_idx//RESPONSES_PER_QUESTION]["question"])
                print("Group:", group)
                print("Response:", response)
                print("Answer:", answer)
                print("Ground truth:", ground_truth)
                print("Verdict:", "Correct" if answer_is_correct else "Wrong")
            chat_templates[overall_idx].append({"role" : "assistant", "content" : response})

            group_statistics[ALL_GROUPS]['is_correct'][overall_idx] = answer_is_correct
            group_statistics[group]['is_correct'][group_idx] = answer_is_correct

    if len(rerun_confidence_idx) > 0:
        print("Asking LLM for confidences when original confidence is invalid")
        chat_templates_confidence = []
        for overall_idx, group_idx in rerun_confidence_idx:
            chat_templates[overall_idx].append({"role" : "user", "content" : REASK_CONFIDENCE_PROMPT})
            assert chat_templates[overall_idx][0]["role"] == "system"
            chat_templates_confidence.append(chat_templates[overall_idx][1:])
            #chat_templates_confidence.append(chat_templates[overall_idx])

        prompts_confidence = tokenizer.apply_chat_template(chat_templates_confidence, tokenize = False, add_generation_prompt = True)
        outputs_confidence = generate_outputs(llm, prompts_confidence, sampling_params_confidence, lora_request)

        for i in range(len(chat_templates_confidence)):
            overall_idx, group_idx = rerun_confidence_idx[i]
            group = dataset[overall_idx//RESPONSES_PER_QUESTION]["group"]

            response = outputs_confidence[i].outputs[0].text
            confidence = int(response)
            if i < 5:
                print("Indices: (%d, %d, %d)" % (overall_idx, group_idx, overall_idx//RESPONSES_PER_QUESTION))
                print("Group:", group)
                print("Confidence:", confidence)

            # Re-evaluate confidence
            normalized_confidence = normalize_confidence(confidence)
            group_statistics[ALL_GROUPS]['confidences'][overall_idx] = normalized_confidence
            group_statistics[group]['confidences'][group_idx] = normalized_confidence
    
    if not os.path.exists("models/evaluate"):
        os.makedirs("models/evaluate")

    with open("models/evaluate/%s.json" % output_filename, "w") as json_file:
        json.dump(group_statistics, json_file, indent=4)

