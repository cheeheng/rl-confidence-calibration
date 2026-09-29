# Google Search Gemini AI is used to assist in creating part of the code, with human supervision and verification.
import os
from unsloth import FastLanguageModel, FastVisionModel
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
from vllm.tokenizers import get_tokenizer
from vllm.sampling_params import StructuredOutputsParams
from datasets import load_dataset
from pathlib import Path
from transformers import TextStreamer
from typing import Union, List, Optional
from pydantic import BaseModel, ConfigDict, Field

import json
import argparse
import re
import yaml

from utils import verify_correctness, LLM_LONG_NAME, get_system_prompt
from utils import get_temperature, get_top_p, get_top_k, get_min_p
from utils import AnswerFormat, find_token_length, is_multimodal
from utils import parse_and_grade_response, sanity_check_config
from utils import normalize_confidence, get_chat_template_format

def generate_outputs(llm, prompts, sampling_params, lora_request):
    if USE_UNSLOTH:
        outputs = llm.fast_generate(prompts, sampling_params=sampling_params, lora_request = lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request = lora_request)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_yaml", dest='eval_yaml', type=str)
    args = parser.parse_args()
    print(args)
    
    EVAL_EXPERIMENT_NAME = args.eval_yaml
    
    with open(Path("config/eval/") / (EVAL_EXPERIMENT_NAME + ".yaml"), 'r') as file:
        config = yaml.safe_load(file)
        print(config)
        
        LLM_SHORT_NAME = config['llm_short_name']
        ENABLE_THINKING = config['enable_thinking']
        THINKING_BUDGET = config['thinking_budget']
        USE_FORMAT_ENFORCER_FIRST_ATTEMPT = config['use_format_enforcer_first_attempt']
        USE_UNSLOTH = config['use_unsloth']
        RANDOM_SEED = config['random_seed_eval']

        EVAL_DATASET_NAME = config['eval_dataset_name']
        USE_MODEL = config['use_model']
        MAX_OUTPUT_TOKENS = config['max_output_tokens']
        SECOND_CHANCE_ANSWER_TOKEN_LIMIT = config['second_chance_answer_token_limit']
        RL_EXPERIMENT_NAME = config['rl_experiment_name']
        
        NUM_CPU_THREADS = config['num_cpu_threads']
        
        SAMPLE_SIZE = config['sample_size']
        RESPONSES_PER_QUESTION = config['responses_per_question']
    
    if RL_EXPERIMENT_NAME is not None and USE_MODEL == "rl":
        with open(Path("config/rl/") / (RL_EXPERIMENT_NAME + ".yaml"), 'r') as file:
            rl_config = yaml.safe_load(file)
        sanity_check_config(config, rl_config)
        
        LORA_RANK = rl_config['lora_rank']
    else:
        rl_config = None

    answer_format_schema = AnswerFormat.model_json_schema()
    print(answer_format_schema)
    structured_outputs_params = StructuredOutputsParams(json=answer_format_schema, strict_mode=True)
    
    SECOND_CHANCE_ANSWER_STRING_PREFIX = "Final Answer:"
    structured_outputs_answer_params = StructuredOutputsParams(regex="^%s.*" % SECOND_CHANCE_ANSWER_STRING_PREFIX, strict_mode=True)
    structured_outputs_confidence_params = StructuredOutputsParams(choice=[str(i) for i in range(101)], strict_mode=True)

    model_name = LLM_LONG_NAME[LLM_SHORT_NAME]
    tokenizer = get_tokenizer(model_name)
    
    dataset_filename = "datasets/test/%s.csv" % EVAL_DATASET_NAME
    dataset = load_dataset("csv", data_files=dataset_filename)["train"]
    
    system_prompt = get_system_prompt(ENABLE_THINKING)
    def generate_prompt(entry, tokenizer):
        entry["prompt"] = [
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : entry["question"]},
        ]
        entry["prompt"] = tokenizer.apply_chat_template(entry["prompt"], tokenize = False, add_generation_prompt = True, enable_thinking = ENABLE_THINKING)
        #print(entry)
        #assert False
        return entry
    
    N_test = len(dataset)
    
    dataset = dataset.map(generate_prompt, batched=False, num_proc=NUM_CPU_THREADS, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.map(find_token_length, batched=True, num_proc=NUM_CPU_THREADS, fn_kwargs={"tokenizer": tokenizer})
    max_prompt_length = max(list(dataset["token_length"]))
    print("Longest prompt in test dataset has %d token(s)." % max_prompt_length)
    max_model_length = MAX_OUTPUT_TOKENS + max_prompt_length + SECOND_CHANCE_ANSWER_TOKEN_LIMIT + 1024
    
    # A bug has been spotted that is expected to affect only Gemma 4 (E2B) Instruct.
    # Extra backslashes are added to unescaped characters to ensure proper escaping of characters, most applicable to Gemma 4 (E2B) Instruct which often outputs broken JSON.
    # This increases the number of tokens in the output, which increases the risk of insufficient context window errors.
    # At the time of bug discovery, Qwen 3 (4B) evaluations are ongoing and are expected to be unaffected by the bug.
    # On the other hand, a Gemma 4 (E2B) Instruct evaluation on correctness-only reward scheme has earlier crashed due to insufficient context window.
    if LLM_SHORT_NAME.startswith("gemma4-"):
        # Add a two times safety margin for the output tokens of the initial response to account for the added escape backslashes to fix defective JSON.
        # Update: A two times safety margin is not enough in extremely rare cases, hence we are increasing it to three times.
        max_model_length += 2*MAX_OUTPUT_TOKENS

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
        temperature = get_temperature(LLM_SHORT_NAME, ENABLE_THINKING),
        top_p = get_top_p(LLM_SHORT_NAME, ENABLE_THINKING),
        top_k = get_top_k(LLM_SHORT_NAME, ENABLE_THINKING),
        min_p = get_min_p(LLM_SHORT_NAME, ENABLE_THINKING),
        structured_outputs = structured_outputs_params if USE_FORMAT_ENFORCER_FIRST_ATTEMPT else None,
        # "maximum number of generated tokens per output sequence"
        # according to documentation from https://docs.vllm.ai/en/latest/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.logprobs
        max_tokens = MAX_OUTPUT_TOKENS, 
    )

    sampling_params_answer = SamplingParams(
        temperature = get_temperature(LLM_SHORT_NAME, ENABLE_THINKING),
        top_p = get_top_p(LLM_SHORT_NAME, ENABLE_THINKING),
        top_k = get_top_k(LLM_SHORT_NAME, ENABLE_THINKING),
        min_p = get_min_p(LLM_SHORT_NAME, ENABLE_THINKING),
        structured_outputs = structured_outputs_answer_params,
        # "maximum number of generated tokens per output sequence"
        # according to documentation from https://docs.vllm.ai/en/latest/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.logprobs
        max_tokens = SECOND_CHANCE_ANSWER_TOKEN_LIMIT, 
    )

    sampling_params_confidence = SamplingParams(
        temperature = get_temperature(LLM_SHORT_NAME, ENABLE_THINKING),
        top_p = get_top_p(LLM_SHORT_NAME, ENABLE_THINKING),
        top_k = get_top_k(LLM_SHORT_NAME, ENABLE_THINKING),
        min_p = get_min_p(LLM_SHORT_NAME, ENABLE_THINKING),
        structured_outputs = structured_outputs_confidence_params,
    )

    prompts = tokenizer.apply_chat_template(chat_templates, tokenize = False, add_generation_prompt = True, enable_thinking = ENABLE_THINKING)
    
    if USE_MODEL == "base":
        # _base at the end of the filename signifies base model
        print("Loading base model %s" % LLM_LONG_NAME[LLM_SHORT_NAME])
    elif USE_MODEL == "rl":
        lora_model_parent_dir = "models/rl/%s/" % RL_EXPERIMENT_NAME
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
    else:
        # SFT support is dropped because it is found to be unnecessary.
        assert False, "--use_model must be either base or rl"

    output_filename = EVAL_EXPERIMENT_NAME
    experiment_id = "eval_" + output_filename
    
    # Check if unsloth works here
    # The LLM inference code is modified from https://github.com/unslothai/unsloth/issues/2551

    base_model_name = LLM_LONG_NAME[LLM_SHORT_NAME]
    
    enable_lora = False if USE_MODEL == "base" else True
    if enable_lora:
        max_lora_rank = LORA_RANK
    else:
        max_lora_rank = None
    
    if is_multimodal(LLM_SHORT_NAME):
        ModelClass = FastVisionModel
    else:
        ModelClass = FastLanguageModel

    if USE_UNSLOTH:
        llm, _ = ModelClass.from_pretrained(
            model_name = base_model_name,
            max_seq_length = max_model_length,
            seed = RANDOM_SEED,
            gpu_memory_utilization=0.85,
            fast_inference = True, # Uses vLLM
            load_in_4bit = False,
            load_in_8bit = False,
            enable_lora = enable_lora,
            text_only = True,
            max_lora_rank = max_lora_rank)
        ModelClass.for_inference(llm)
    else:
        llm = LLM(model=base_model_name, 
            max_model_len=max_model_length, 
            seed = RANDOM_SEED, 
            gpu_memory_utilization=0.85,
            enable_lora = enable_lora,
            max_lora_rank = max_lora_rank,
            # enforce_eager=True, # This is needed for Ministral 3 (3B) Reasoning 2512 to work
            language_model_only=True)

    lora_request = None if USE_MODEL == "base" else LoRARequest(lora_name="lora_adapter", lora_int_id=1+secrets.randbelow((1<<31) - 1), lora_path=lora_model_dir)

    outputs = generate_outputs(llm, prompts, sampling_params, lora_request)
    
    assert len(chat_templates) == N_test * RESPONSES_PER_QUESTION
    
    groups = list(set(dataset["group"]))
    print("List of groups:", groups)
    
    ALL_GROUPS = "overall"
    assert ALL_GROUPS not in groups
    groups.append(ALL_GROUPS)

    assert "metadata" not in groups
    group_statistics = {"metadata": {}}
    group_statistics["metadata"]["eval_config"] = config
    group_statistics["metadata"]["rl_config"] = rl_config
    group_statistics["metadata"]["dataset_stats"] = {"num_questions": N_test}
    for group in groups:
        group_statistics[group] = {'is_correct': [], 'confidences': [], 
            'invalid_counts': {'confidence': 0, 'answer': 0, 'format': 0}}
    
    print("Sanity check (first 10 responses)")
    rerun_answer_idx = []
    rerun_confidence_idx = []
    final_answers = [None for i in range(len(chat_templates))]
    for i in range(len(chat_templates)):
        #print("Output %d" % i)
        response = outputs[i].outputs[0].text
        #print(response)
        
        group = dataset[i//RESPONSES_PER_QUESTION]["group"]
        ground_truth = str(dataset[i//RESPONSES_PER_QUESTION]["ground_truth"])

        group_idx = len(group_statistics[group]['is_correct'])
        assert group_idx == len(group_statistics[group]['confidences'])
        
        response_json = parse_and_grade_response(response, EVAL_DATASET_NAME, ground_truth, experiment_id)
        format_is_valid = not response_json['invalid_format']
        answer_is_valid = 'answer' not in response_json['invalid_fields']
        valid_confidence_output = 'confidence' not in response_json['invalid_fields']
        answer_is_correct = bool(response_json['correct'])
        answer = response_json['answer']
        confidence = int(response_json['confidence'])
        
        final_answers[i] = answer # To record so that abstention can be judeged
        
        if not format_is_valid:
            group_statistics[group]['invalid_counts']['format'] += 1
            group_statistics[ALL_GROUPS]['invalid_counts']['format'] += 1
        
        if not answer_is_valid:
            group_statistics[group]['invalid_counts']['answer'] += 1
            group_statistics[ALL_GROUPS]['invalid_counts']['answer'] += 1
            rerun_answer_idx.append((i, group_idx))
        
        if not valid_confidence_output:
            group_statistics[group]['invalid_counts']['confidence'] += 1
            group_statistics[ALL_GROUPS]['invalid_counts']['confidence'] += 1
            rerun_confidence_idx.append((i, group_idx))
        
        if i < 10:
            print("Group:", group)
            print("Question:", dataset[i//RESPONSES_PER_QUESTION]["question"])
            print("Answer:", answer)
            print("Ground Truth:", ground_truth)
            print("Confidence:", confidence)
            print("Response:", response)
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
    if EVAL_DATASET_NAME in ["bigmath", "deepmath-103k"]:
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
        
        # No thinking after initial token time limit to avoid cheating
        prompts_answer = tokenizer.apply_chat_template(chat_templates_answer, tokenize = False, add_generation_prompt = True, enable_thinking = False)
        outputs_answer = generate_outputs(llm, prompts_answer, sampling_params_answer, lora_request)

        for i in range(len(chat_templates_answer)):
            overall_idx, group_idx = rerun_answer_idx[i]
            group = dataset[overall_idx//RESPONSES_PER_QUESTION]["group"]
            ground_truth = dataset[overall_idx//RESPONSES_PER_QUESTION]["ground_truth"]

            response = outputs_answer[i].outputs[0].text
            answer = response[len(SECOND_CHANCE_ANSWER_STRING_PREFIX):]
            final_answers[overall_idx] = answer

            # Re-evaluate answer
            answer_is_correct = bool(verify_correctness(EVAL_DATASET_NAME, answer, ground_truth, experiment_id))
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

        # No thinking after initial token time limit to avoid cheating
        prompts_confidence = tokenizer.apply_chat_template(chat_templates_confidence, tokenize = False, add_generation_prompt = True, enable_thinking = False)
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
    
    #print(group_statistics)

    with open(Path("models/evaluate") / ("%s.json" % output_filename), "w") as json_file:
        json.dump(group_statistics, json_file, indent=4)

    if not os.path.exists("models/evaluate/final_answers"):
        os.makedirs("models/evaluate/final_answers")
    
    with open(Path("models/evaluate/final_answers") / ("%s.json" % output_filename), "w") as json_file:
        json.dump({"eval_config": config, "final_answers": final_answers}, json_file, indent=4)
    
