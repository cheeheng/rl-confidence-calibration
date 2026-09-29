# Google Search Gemini AI is used to assist in creating part of the code, with human supervision and verification.
import os
import secrets

# Attempts to help ensure reproducibility: https://discuss.vllm.ai/t/two-different-runs-give-different-answers/2025
# But still need to ensure library version consistency and same hardware
# The results remain non-deterministic when LoRA adapters are used
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.tokenizers import get_tokenizer
from vllm.sampling_params import StructuredOutputsParams
from vllm.parser.qwen3 import Qwen3Parser
from datasets import load_dataset
from pathlib import Path
from transformers import TextStreamer
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from openai_harmony import load_harmony_encoding, HarmonyEncodingName
from openai_harmony import Conversation, Role, Message, ReasoningEffort
from openai_harmony import SystemContent, DeveloperContent

import json
import argparse
import re
import yaml
import string

from utils import verify_correctness, LLM_LONG_NAME, get_system_prompt
from utils import get_temperature, get_top_p, get_top_k, get_min_p
from utils import AnswerFormat, find_token_length, is_multimodal
from utils import parse_and_grade_response, sanity_check_config
from utils import normalize_confidence, get_abstention_judge_prompt
from utils import get_chat_template_format

class VerdictFormat(BaseModel):
    response: Literal["unanswerable", "abstain", "answer"]
    model_config = ConfigDict(extra='forbid', strict=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_yaml", dest='eval_yaml', type=str)
    parser.add_argument("--judge_yaml", dest='judge_yaml', type=str)
    args = parser.parse_args()
    print(args)
    
    EVAL_EXPERIMENT_NAME = args.eval_yaml
    JUDGE_NAME = args.judge_yaml
    
    with open(Path("config/eval/") / (EVAL_EXPERIMENT_NAME + ".yaml"), 'r') as file:
        config = yaml.safe_load(file)
        print(config)

        EVAL_DATASET_NAME = config['eval_dataset_name']
        NUM_CPU_THREADS = config['num_cpu_threads']
        RESPONSES_PER_QUESTION = config['responses_per_question']

    with open(Path("config/judge/") / (JUDGE_NAME + ".yaml"), 'r') as file:
        judge_config = yaml.safe_load(file)
        print(judge_config)
        
        LLM_SHORT_NAME = judge_config['llm_judge_short_name']
        ENABLE_THINKING = judge_config['enable_thinking_judge']
        RANDOM_SEED = judge_config['random_seed_judge']
        THINKING_BUDGET = judge_config['thinking_budget_judge']
        
        if 'debug_low_vram' in judge_config:
            DEBUG_LOW_VRAM = judge_config['debug_low_vram']
        else:
            DEBUG_LOW_VRAM = False
        
        if 'debug_judge_output' in judge_config:
            DEBUG_JUDGE_OUTPUT = judge_config['debug_judge_output']
        else:
            DEBUG_JUDGE_OUTPUT = False
        
    # Note: this includes Qwen 3.5 and Qwen 3
    if LLM_SHORT_NAME.startswith("qwen3"):
        reasoning_parser = "qwen3"
    else:
        reasoning_parser = None

    answer_choices = ["unanswerable", "abstain", "answer"]
    answer_choices_schema = VerdictFormat.model_json_schema()
    if reasoning_parser == "qwen3":
        corrected_answer_choices = ['\n\n' + choice for choice in answer_choices]
    else:
        corrected_answer_choices = answer_choices.copy()
    structured_outputs_params_chatml = StructuredOutputsParams(choice=corrected_answer_choices, strict_mode=True)
    #structured_outputs_params_chatml = StructuredOutputsParams(json=answer_choices_schema, strict_mode=True)
    structured_outputs_params_harmony = StructuredOutputsParams(regex="\\A[\\s\\S]*assistantfinal(" + "|".join(answer_choices) + ")\\Z")

    model_name = LLM_LONG_NAME[LLM_SHORT_NAME]
    tokenizer = get_tokenizer(model_name)
    
    dataset_filename = "datasets/test/%s.csv" % EVAL_DATASET_NAME
    dataset = load_dataset("csv", data_files=dataset_filename)["train"]
    
    with open(Path("models/evaluate/final_answers") / ("%s.json" % EVAL_EXPERIMENT_NAME), "r") as json_file:
        answer_data = json.load(json_file)
        eval_config = answer_data['eval_config']
        final_answers = answer_data['final_answers']
    
    print(eval_config)
    # assert eval_config == config # To ensure no meddling with eval_config, but not very useful during debugging
    
    chat_templates = []
    chat_template_format = get_chat_template_format(LLM_SHORT_NAME)
    # OpenAI Harmony Chat Template format code taken and modified from https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm#using-vllm-for-direct-sampling
    if chat_template_format == "Harmony":
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        prompt_tokens = []
        reasoning_effort = ReasoningEffort.HIGH if ENABLE_THINKING else ReasoningEffort.LOW
        structured_outputs_params = structured_outputs_params_harmony
    elif chat_template_format == "ChatML":
        structured_outputs_params = structured_outputs_params_chatml
        #structured_outputs_params = None

    for i in range(len(final_answers)):
        question_index = i // RESPONSES_PER_QUESTION
        if DEBUG_JUDGE_OUTPUT:
            # This intentionally gives some abstention cases to test for correctness
            if i % 4 == 0:
                final_answers[i] = "There is not enough information to answer this question"
            elif i % 4 == 2:
                final_answers[i] = "I don't wish to answer this question"
            pass
        
        intended_user_prompt = get_abstention_judge_prompt(dataset["question"][question_index], final_answers[i])
        
        if chat_template_format == "ChatML":
            chat_template = [
                {"role": "user", "content": intended_user_prompt},
            ]
        elif chat_template_format == "Harmony":
            chat_template = Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(reasoning_effort)),
                    Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new()),
                    Message.from_role_and_content(Role.USER, intended_user_prompt),
                ]
            )

            prefill_ids = encoding.render_conversation_for_completion(chat_template, Role.ASSISTANT)
            #print(chat_template, len(prefill_ids))
            prompt_tokens.append(prefill_ids)
        else:
            assert False, "Unsupported chat template format"
        
        chat_templates.append(chat_template)
        #print("Ground truth:", dataset["ground_truth"][question_index])
        #print("LLM final answer:", final_answers[i])
        #print("Chat Template:", chat_template)
    
    assert len(final_answers) % RESPONSES_PER_QUESTION == 0
    N_test = len(final_answers) // RESPONSES_PER_QUESTION
    
    if chat_template_format == "ChatML":
        MAX_OUTPUT_TOKENS = (THINKING_BUDGET if ENABLE_THINKING else 0) + 128
    
        # Get tokenizer statistics - find longest token length (with help of Google Search Gemini AI)
        # Taken from another file: test_llm_dataset.py (originally used for internal debugging)
        prompt_tokens = tokenizer.apply_chat_template(
            chat_templates, 
            tokenize = True, 
            add_generation_prompt = True, 
            return_tensors = None,
            enable_thinking = ENABLE_THINKING)
        prompt_tokens = prompt_tokens["input_ids"] # Later versions of vLLM have added input_ids, attention_mask and other attributes
    elif chat_template_format == "Harmony":
        MAX_OUTPUT_TOKENS = THINKING_BUDGET + 128

        # Tokenization done in earlier loop
    else:
        assert False, "Unsupported chat template format"
    
    token_lengths = [len(tokens) for tokens in prompt_tokens]
    max_prompt_length = max(token_lengths)
    print("Longest prompt in abstention dataset (derived from test dataset and corresponding LLM answers) has %d token(s)." % max_prompt_length)
    max_model_length = max_prompt_length + MAX_OUTPUT_TOKENS + 512
    
    sampling_params = SamplingParams(
        temperature = get_temperature(LLM_SHORT_NAME, ENABLE_THINKING),
        top_p = get_top_p(LLM_SHORT_NAME, ENABLE_THINKING),
        top_k = get_top_k(LLM_SHORT_NAME, ENABLE_THINKING),
        min_p = get_min_p(LLM_SHORT_NAME, ENABLE_THINKING),
        # Note: buggy in OpenAI Harmony format, affects usability
        # Example reference: https://github.com/ollama/ollama/issues/11691
        # Another example reference: https://github.com/waybarrios/vllm-mlx/issues/378
        structured_outputs = structured_outputs_params if chat_template_format == "ChatML" else None, 
        # "maximum number of generated tokens per output sequence"
        # according to documentation from https://docs.vllm.ai/en/latest/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.logprobs
        max_tokens = MAX_OUTPUT_TOKENS, 
        thinking_token_budget = THINKING_BUDGET if (ENABLE_THINKING and reasoning_parser is not None) else None,
    )

    if chat_template_format == "ChatML":
        prompts = tokenizer.apply_chat_template(chat_templates, tokenize = False, add_generation_prompt = True, enable_thinking = ENABLE_THINKING)
    elif chat_template_format == "Harmony":
        prompts = [{"prompt_token_ids": tokens} for tokens in prompt_tokens]
    else:
        assert False, "Unsupported chat template format"
        
    #if chat_template_format == "Harmony":
    #    assert False, "Format enforcers currently do not work on OpenAI Harmony format."

    base_model_name = LLM_LONG_NAME[LLM_SHORT_NAME]
    
    if DEBUG_LOW_VRAM:
        gpu_memory_utilization = 0.98 if LLM_SHORT_NAME == "gpt-oss-20b" else 0.95
    else:
        gpu_memory_utilization = 0.85
    
    llm = LLM(
        model=base_model_name, 
        max_model_len=max_model_length, 
        seed = RANDOM_SEED, 
        gpu_memory_utilization=gpu_memory_utilization,
        language_model_only=True,
        cpu_offload_gb=0,
        enforce_eager=True if DEBUG_LOW_VRAM else False, # This is included only for debugging purposes
        max_num_seqs=4 if DEBUG_LOW_VRAM else 256,
        reasoning_parser=reasoning_parser,
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    print(len(chat_templates), N_test)
    assert len(chat_templates) == N_test * RESPONSES_PER_QUESTION
    
    groups = list(set(dataset["group"]))
    print("List of groups:", groups)
    
    ALL_GROUPS = "overall"
    assert ALL_GROUPS not in groups
    groups.append(ALL_GROUPS)

    assert "metadata" not in groups
    abstention_statistics = {"metadata": {}}
    abstention_statistics["metadata"]["eval_config"] = config
    abstention_statistics["metadata"]["judge_config"] = judge_config
    abstention_statistics["metadata"]["dataset_stats"] = {"num_questions": N_test}
    abstention_statistics[ALL_GROUPS] = {'counts': {choice: 0 for choice in answer_choices}, 'llm_judge_verdicts': [], 'llm_judge_format_errors': 0}
    for group in groups:
        abstention_statistics[group] = {'counts': {choice: 0 for choice in answer_choices}, 'llm_judge_verdicts': [], 'llm_judge_format_errors': 0}
    
    print_count = min(len(chat_templates), 400)
    print("Sanity check (first %d responses)" % print_count)
    format_error_count = 0
    for i in range(len(chat_templates)):
        response = outputs[i].outputs[0].text
        
        if LLM_SHORT_NAME.startswith("qwen3"):
            parser = Qwen3Parser(tokenizer)
            filtered_response = parser.extract_reasoning(response, outputs[i].outputs[0])
            response = filtered_response[1 if ENABLE_THINKING else 0].strip()
            if response not in answer_choices:
                response = None
        elif chat_template_format == "Harmony":
            final_response = None
            for choice in answer_choices:
                if response.strip().endswith("assistantfinal" + choice):
                    final_response = choice
                    break
            response = final_response
        
        if i < print_count:
            print(chat_templates[i])
            print(outputs[i].outputs[0].text)
            print("Output %d: %s\n" % (i, response))
        
        group = dataset[i//RESPONSES_PER_QUESTION]["group"]
        if response is None:
            abstention_statistics[ALL_GROUPS]['llm_judge_format_errors'] += 1
            abstention_statistics[group]['llm_judge_format_errors'] += 1
            format_error_count += 1
        else:
            abstention_statistics[ALL_GROUPS]['counts'][response] += 1
            abstention_statistics[group]['counts'][response] += 1
        
        str_response = "" if response is None else response
        abstention_statistics[ALL_GROUPS]['llm_judge_verdicts'].append(str_response)        
        abstention_statistics[group]['llm_judge_verdicts'].append(str_response)
    
    print("Total number of responses:", len(chat_templates))
    for choice in answer_choices:
        print("Total number of LLM-judged verdict '" + choice + "':", abstention_statistics[ALL_GROUPS]['counts'][choice])
    print("Total number of LLM judge format errors:", format_error_count)
    
    if not os.path.exists("models/evaluate/abstain/%s" % JUDGE_NAME):
        os.makedirs("models/evaluate/abstain/%s" % JUDGE_NAME)

    with open(Path("models/evaluate/abstain") / JUDGE_NAME / ("%s.json" % EVAL_EXPERIMENT_NAME), "w") as json_file:
        json.dump(abstention_statistics, json_file, indent=4)

