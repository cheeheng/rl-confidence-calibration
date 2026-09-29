# Google Search Gemini AI is used to create part of the code and confirm correctness of some semantics.
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
from transformers.utils import logging
from pydantic import BaseModel, ConfigDict, Field
import math
import fast_json_repair
import evaluate
import re
import json

logging.set_verbosity_warning()

LLM_LONG_NAME = {
    "qwen2.5-1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "unsloth/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-Instruct",
    "phi4": "unsloth/phi-4",
    "gemma4-e2b-it": "unsloth/gemma-4-E2B-it",
    "gemma4-e2b-it-fp8": "prithivMLmods/gemma-4-E2B-it-FP8",
    "ministral3-3b-instruct-2512-fp8": "unsloth/Ministral-3-3B-Instruct-2512-FP8",
    "ministral3-3b-instruct-2512": "unsloth/Ministral-3-3B-Instruct-2512",
    "ministral3-3b-reasoning-2512": "unsloth/Ministral-3-3B-Reasoning-2512",
    "qwen3-0.6b": "unsloth/Qwen3-0.6B",
    "qwen3-1.7b": "unsloth/Qwen3-1.7B",
    "qwen3-4b": "unsloth/Qwen3-4B",
    "qwen3-4b-fp8": "unsloth/Qwen3-4B-FP8",
    "qwen3-8b-fp8": "unsloth/Qwen3-8B-FP8",
    "qwen3-8b": "unsloth/Qwen3-8B",
    "qwen3-14b-nvfp4": "RedHatAI/Qwen3-14B-NVFP4",
    "qwen3-14b-fp8": "unsloth/Qwen3-14B-FP8",
    "qwen3-14b": "unsloth/Qwen3-14B",
    "qwen3-30b-a3b-fp8": "unsloth/Qwen3-30B-A3B-FP8", # Added at the same time as get_abstention_judge_prompt was updated, no expected impact on reinforcement learning training
    "qwen3-4b-instruct-2507": "unsloth/Qwen3-4B-Instruct-2507",
    "qwen3-4b-thinking-2507": "unsloth/Qwen3-4B-Thinking-2507",
    "qwen3-4b-thinking-2507": "unsloth/Qwen3-4B-Thinking-2507",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "nemotron-3.5-lightning-nvfp4": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "nemotron-3.5-lightning-fp8": "RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8",
}

def is_multimodal(llm_short_name: str) -> bool:
    if llm_short_name.startswith("gemma4-"):
        return True
    elif llm_short_name.startswith("qwen3-") or llm_short_name.startswith("qwen2.5-"):
        return False
    elif llm_short_name.startswith("gpt-oss-"):
        return False
    else:
        assert False, "Unable to determine if model is multimodal"

def model_supports_float16(llm_short_name: str) -> bool:
    if llm_short_name.startswith("gemma4-"):
        return False
    elif llm_short_name.startswith("qwen3-") or llm_short_name.startswith("qwen2.5-"):
        return True
    else:
        assert False, "Unable to determine if model supports float16"

# Qwen 3 best practices: https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
# Qwen 3 (2507) best practices: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#best-practices for thinking, https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507#best-practices for instruct
# Gemma 4 best practices: https://huggingface.co/google/gemma-4-E2B-it#best-practices
# GPT OSS best practices: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune

def get_temperature(llm_short_name: str, enable_thinking: bool):
    if llm_short_name.startswith("qwen3-"):
        return 0.6 if enable_thinking else 0.7
    elif llm_short_name.startswith("ministral3-"):
        return 0.7
    elif llm_short_name.startswith("gemma4-"):
        return 1
    elif llm_short_name.startswith("gpt-oss-"):
        return 1
    else:
        return 1

def get_top_p(llm_short_name: str, enable_thinking: bool):
    if llm_short_name.startswith("qwen3-"):
        return 0.95 if enable_thinking else 0.8
    elif llm_short_name.startswith("ministral3-"):
        return 0.95
    elif llm_short_name.startswith("gemma4-"):
        return 0.95
    elif llm_short_name.startswith("gpt-oss-"):
        return 1
    else:
        return 1

def get_top_k(llm_short_name: str, enable_thinking: bool):
    if llm_short_name.startswith("qwen3-"):
        return 20
    elif llm_short_name.startswith("gemma4-"):
        return 64
    elif llm_short_name.startswith("gpt-oss-"):
        return 0
    else:
        return 0

def get_min_p(llm_short_name: str, enable_thinking: bool):
    return 0

def get_rl_temperature(llm_short_name: str, enable_thinking: bool):
    return 1

def get_rl_top_p(llm_short_name: str, enable_thinking: bool):
    if llm_short_name.startswith("gemma4-") and False:
        return 0.95
    else:
        return 1

def get_rl_top_k(llm_short_name: str, enable_thinking: bool):
    if llm_short_name.startswith("gemma4-") and False:
        return 64
    else:
        return 0 # 0 considers all tokens

def get_rl_min_p(llm_short_name: str, enable_thinking: bool):
    return 0

class AnswerFormat(BaseModel):
    reasoning: str
    answer: str # Note: the 1000-character limit is removed because this is identified as a potential confounding factor in the paper review and it is not necessary.
    confidence_analysis: str
    confidence: int = Field(ge=0, le=100)
    model_config = ConfigDict(extra='forbid', strict=True)

def f_underconfidence(k, c):
    return (k*c + math.log1p(-k*c / (1+k))) / (k - math.log1p(k))

def g_underconfidence(k, c):
    return (k*c + (k+1)*math.log1p(-k*c / (1+k))) / (k - math.log1p(k))

def f_overconfidence(k, c):
    return ((k+1)*math.log(c*k+1) - c*k) / ((k+1)*math.log(k+1) - k)

def g_overconfidence(k, c):
    return (math.log(c*k+1) - c*k) / ((k+1)*math.log(k+1) - k)

def f(confidence, reward_scheme):
    if reward_scheme == "correctness-only":
        return 1
    elif reward_scheme == "log-1":
        return 1 + math.log(confidence)
    elif reward_scheme == "log-no-correctness-reward":
        return math.log(confidence)
    elif reward_scheme == "log-1-div-ln-202":
        return 1 + (math.log(confidence) / math.log(202))
    elif reward_scheme == "brier-no-correctness-reward":
        return - (1 - confidence) ** 2
    elif reward_scheme == "brier-1":
        return 1 - (1 - confidence) ** 2
    elif reward_scheme == "brier-2":
        return 1 - 2 * ((1 - confidence) ** 2)
    elif reward_scheme == "brier-log-hybrid":
        return confidence
    elif reward_scheme == "underconfidence-1":
        return f_underconfidence(1, confidence)
    elif reward_scheme == "underconfidence-4":
        return f_underconfidence(4, confidence)
    elif reward_scheme == "overconfidence-1":
        return f_overconfidence(1, confidence)
    elif reward_scheme == "overconfidence-4":
        return f_overconfidence(4, confidence)
    elif reward_scheme == "overconfidence-1000":
        return f_overconfidence(1000, confidence)
    else:
        assert False
    
def g(confidence, reward_scheme):
    if reward_scheme == "correctness-only":
        return 0
    elif reward_scheme == "log-1":
        return math.log(1-confidence)
    elif reward_scheme == "log-no-correctness-reward":
        return math.log(1-confidence)
    elif reward_scheme == "log-1-div-ln-202":
        return (math.log(1-confidence) / math.log(202))
    elif reward_scheme == "brier-no-correctness-reward":
        return - confidence ** 2
    elif reward_scheme == "brier-1":
        return - confidence ** 2
    elif reward_scheme == "brier-2":
        return - 2 * (confidence ** 2)
    elif reward_scheme == "brier-log-hybrid":
        return confidence + math.log(1-confidence)
    elif reward_scheme == "underconfidence-1":
        return g_underconfidence(1, confidence)
    elif reward_scheme == "underconfidence-4":
        return g_underconfidence(4, confidence)
    elif reward_scheme == "overconfidence-1":
        return g_overconfidence(1, confidence)
    elif reward_scheme == "overconfidence-4":
        return g_overconfidence(4, confidence)
    elif reward_scheme == "overconfidence-1000":
        return g_overconfidence(1000, confidence)
    else:
        assert False

def E(confidence, p, reward_scheme):
    return p * f(confidence, reward_scheme) + (1 - p) * g(confidence, reward_scheme)

def normalize_confidence(confidence: float) -> float:
    return (confidence + 0.5) / 101

def miscalibration_penalty(confidence, p, reward_scheme):
    return E(p, p, reward_scheme) - E(confidence, p, reward_scheme)

def get_rl_train_path(llm_short_name: str, enable_thinking: bool, dataset_name: str) -> str:
    return "datasets/rl-train/%s/%s/%s.csv" % (llm_short_name, "think" if enable_thinking else "nothink", dataset_name)

def get_system_prompt(enable_thinking: bool) -> str:
    prompt = 'Respond in the JSON format: {"reasoning": string, "answer": string, "confidence_analysis": string, "confidence": int}\n\n'
    if enable_thinking:
        prompt += 'Provide your summarized thought process in "reasoning" attribute.\n'
    else:
        prompt += 'Provide your detailed internal thought process in "reasoning" attribute.\n'
    prompt += 'Provide only your final answer in "answer" attribute.\n'
    prompt += 'For mathematical questions, provide your final answer in LaTeX format.\n'
    prompt += 'Provide your analysis on the uncertainty of your answer in "confidence_analysis" attribute, taking into account various factors that may lead to your answer being different or incorrect, such as errors in calculation and ambiguities in answer or grading.\n'
    prompt += 'In "confidence" attribute, express your confidence as an integer between 0 and 100 inclusive, representing the subjective probability that your answer is correct.\n\n'
    prompt += 'You score points based on whether your answer correctly and how well-calibrated your confidence is. Your objective is to maximize the total number of points scored over all the questions asked.'
    return prompt

# This method is edited before any evaluation takes place, but while reinforcement learning experiments are in progress.
# Qwen 3 (4B) HotpotQA-Modified experiments have already completed.
# The first run of Gemma 4 (E2B) HotpotQA-Modified experiments have already completed.
# Qwen 3 (4B) DeepMath-103K experiments are still ongoing.
# Since this method is only called during evaluation, there is no expected impact of the output by changing the function after starting the reinforcement learning experiments.
def get_abstention_judge_prompt(question: str, final_answer: str):
    prompt = "Given the question and the answer written in JSON format, determine whether the LLM opted to abstain from answering the question.\n\n"
    prompt += "Respond with the following (without the quotes):\n"
    prompt += '(i) "unanswerable" if the LLM has responded that the question is unanswerable due to having insufficient information, regardless of whether the question is actually answerable.\n'
    prompt += '(ii) "abstain" if the LLM has abstained from answering the question in a manner that does not otherwise qualify as "unanswerable". If the LLM abstains by implying that the question is unanswerable, respond with "unanswerable".\n'
    prompt += '(iii) "answer" if the LLM has answered the question, regardless of whether the question is answered correctly.\n\n'
    prompt += 'Answer with only "unanswerable", "abstain" or "answer", without the quotes. Do not attempt to answer or solve the question given in JSON because this is irrelevant in judging whether or how the LLM has abstained from answering the question.\n\n'
    prompt += json.dumps({"question": question, "answer": final_answer})
    return prompt

# Get tokenizer statistics - find longest token length (with help of Google Search Gemini AI)
def find_token_length(entries, tokenizer):
    prompt_tokens = tokenizer(entries["prompt"], return_tensors=None)
    #print(prompt_tokens['input_ids'])
    #assert False
    token_length = [len(prompt_token) for prompt_token in prompt_tokens['input_ids']]
    return {"token_length": token_length}

# Determines if confidence score is valid - should be integer between 0 and 100 inclusive
def confidence_is_valid(confidence: str) -> bool:
    confidence = str(confidence) # This part is necessary as Python allows converting float 14.5 to integer.
    try:
        # Python raises error when string 14.5 is converted to an integer, hence this is ok.
        confidence = int(confidence)
        if confidence < 0 or confidence > 100:
            raise ValueError()
        return True
    except ValueError:
        return False

# Attempts to restore the original string by replacing characters such as \t with their escaped versions.
def add_escape_characters(answer: str) -> str:
    escape_characters = ['\a', '\r', '\t', '\b', '\f', '\v']
    replacement_characters = ['\\a', '\\r', '\\t', '\\b', '\\f', '\\v']
    assert len(escape_characters) == len(replacement_characters)
    
    for i in range(len(escape_characters)):
        answer = answer.replace(escape_characters[i], replacement_characters[i])
    return answer

# Adds $$ in the string if string is not already LaTeX math - currently does a blind check to determine if $$ is present
# If $$ is not present, code simply surrounds string with $$.
def turn_into_latex_math(latex_str: str) -> str:
    if latex_str == "":
        return "$$"
    elif latex_str[0] == '$' and latex_str[-1] == '$':
        return latex_str
    else:
        return '$' + latex_str + '$'
        
def remove_latex_math(latex_str: str) -> str:
    # I allowed . to be removed since some yes/no answers resulted in LLM outputting yes. or no., resulting in correct answer marked as wrong
    if latex_str == "":
        return ""
    elif latex_str[0] == '$' and latex_str[-1] == '$':
        return latex_str[1:-1]
    elif latex_str[-1] == '.':
        return latex_str[:-1]
    else:
        return latex_str

rouge = None
def f1_score_of_word_overlap(llm_answer: str, ground_truth: str, experiment_id: str = None) -> float:
    global rouge
    if rouge is None:
        rouge = evaluate.load("rouge", experiment_id = experiment_id)
    results = rouge.compute(predictions=[llm_answer], references=[ground_truth])
    return results['rouge1']

# Argument order is important here
def verify_correctness(dataset_name: str, llm_answer: str, ground_truth: str, experiment_id: str = None) -> bool:
    #print("LLM answer:", llm_answer)
    #print("Ground truth:", ground_truth)
    #print()

    if dataset_name in ["addition", "multi-armed-bandit-22222", "multi-armed-bandit-64", "multi-armed-bandit-82", 
        "noisy-ground-truth-sequential", "noisy-ground-truth-random"]:
        return llm_answer.lower().strip() == ground_truth.lower().strip()
    elif dataset_name in ["hotpotqa", "hotpotqa-modified"]:
        return f1_score_of_word_overlap(llm_answer.lower().strip(), ground_truth.lower().strip(), experiment_id = experiment_id) > 0.7
    elif dataset_name in ["deepmath-103k", "bigmath"]:
        llm_answer = llm_answer.strip()
        ground_truth = ground_truth.strip()
        
        # This should never happen as the preprocessing filter removes questions that are multiple choice, yes/no and proof questions
        if dataset_name == "deepmath-103k":
            # If ground truth is A, then Yes and True are considered correct.
            # If ground truth is B, then No and False are considered correct. This is to account for LLM hallucination in ground truth.
            # If ground truth is C, then only C is correct.
            # If ground truth is D, then only D is correct.
            # If ground truth is Yes or True, then Yes and True are considered correct.
            # If ground truth is No or False, then No and False are considered correct.
            
            llm_answer_no_latex = remove_latex_math(llm_answer).lower().strip()
            ground_truth_no_latex = remove_latex_math(ground_truth).lower().strip()
            if ground_truth_no_latex == "a":
                assert False
                return llm_answer_no_latex in ["a", "yes", "true"]
            elif ground_truth_no_latex == "b":
                assert False
                return llm_answer_no_latex in ["b", "no", "false"]
            elif ground_truth_no_latex == "c":
                assert False
                return llm_answer_no_latex == "c"
            elif ground_truth_no_latex == "d":
                assert False
                return llm_answer_no_latex == "d"
            elif ground_truth_no_latex in ["yes", "true"]:
                assert False
                return llm_answer_no_latex in ["yes", "true"]
            elif ground_truth_no_latex in ["no", "false"]:
                assert False
                return llm_answer_no_latex in ["no", "false"]

        # https://github.com/huggingface/Math-Verify
        # We allow both $answer$ and answer to simplify marking.
        ground_truth_latex_math = turn_into_latex_math(ground_truth)
        llm_answer_latex_math = turn_into_latex_math(llm_answer)
        
        ground_truth_latex_math = parse(ground_truth_latex_math, extraction_config=[LatexExtractionConfig()])
        llm_answer_latex_math = parse(llm_answer_latex_math, extraction_config=[LatexExtractionConfig()])
        
        llm_answer = parse(llm_answer)
        
        if dataset_name == "bigmath":
            return verify(ground_truth_latex_math, llm_answer, float_rounding=2) or verify(ground_truth_latex_math, llm_answer_latex_math, float_rounding=2)
        elif dataset_name == "deepmath-103k":
            ground_truth = parse(ground_truth)
            # Accept answer if and only if answer matches either in non-LaTeX mode or in LaTeX mode
            return verify(ground_truth, llm_answer, float_rounding=2) or verify(ground_truth_latex_math, llm_answer_latex_math, float_rounding=2)
    else:
        assert False

def corrected_loads(response: str):
    # fast_json_repair.loads is useful, but sometimes, it fails to repair the string
    # This often happens when the LLM output has text other than the relevant JSON. 
    # Often, heuristics such as removing the distracting parts not relevant to the JSON will lead to more accurate detection of the JSON.
    
    # Attempt to detect JSON
    # This roughly works because re.search finds first occurrence of matched string, and * matches as many characters as possible.
    # re: https://docs.python.org/3/library/re.html
    # Note: [\s\S] is required to match any character, including whitespace
    # https://stackoverflow.com/questions/33312175/matching-any-character-including-newlines-in-a-python-regex-subexpression-not-g
    matched_json = re.search('{\\s*"[\\s\\S]*}', response)
    if matched_json is None:
        return "" # Empty string to signify format is wrong.
    
    start, excluded_end = matched_json.span()
    corrected_response = response[start:excluded_end]
    
    # The corrected_response variable still needs one round of correction.
    # This is because fast_json_repair.loads() fails to account for the unescaped ""
    # Though not perfect, a mitigating measure would be to look for the syntax markers of the intended JSON tags, and replace all other occurrences of " with '.
    
    avoid_index_range = []
    
    # This needs to be done separately because re.finditer() only finds non-overlapping matches
    lst_end = [match.span() for match in re.finditer('"\\s*,?\\s*("|})', corrected_response)] # detects intended closing " in JSON field
    
    # Detects JSON "reasoning" tags etc., including the opening " for the value of the field if applicable
    lst_start = [match.span() for match in re.finditer('"(reasoning|answer|confidence_analysis|confidence)"\\s*:\\s*"?', corrected_response)] 

    avoid_index_range = sorted(lst_start + lst_end) 

    corrected_response_chars = list(corrected_response)
    
    # matches all unescaped ": (?<!\\)"
    
    unescaped_idx = sorted([match.span() for match in re.finditer(r'(?<!\\)"', corrected_response)])
    #print(unescaped_idx)

    for i in range(len(unescaped_idx)):
        assert unescaped_idx[i][0] + 1 == unescaped_idx[i][1]
        j = unescaped_idx[i][0]
        assert corrected_response_chars[j] == '"'
        
        should_be_escaped = True
        for start, exclusive_end in avoid_index_range:
            if j >= start and j < exclusive_end:
                should_be_escaped = False
                break
            
        if should_be_escaped:
            corrected_response_chars[j] = '\\"'
    
    corrected_response = ''.join(corrected_response_chars)
   
    #print("Response")
    #print(response)
    #print("Corrected response:")
    #print(corrected_response)
    return fast_json_repair.loads(corrected_response, ensure_ascii=False)

def parse_and_grade_response(response: str, dataset_name: str, ground_truth: str, experiment_id: str = None):
    response_json = corrected_loads(response)
            
    while type(response_json) == type([]) and len(response_json) == 1:
        # fast_json_repair.loads (called in corrected_loads) may read a list of length 1, with the only element of the list following the correct format.
        # And it may accidentally occur as a nested list
        response_json = response_json[0]
    
    invalid_format = False
    if type(response_json) != type({}):
        invalid_format = True
        response_json = {}
    
    # From here onwards, we assume response_json is a Python dictionary
    desired_fields = ['reasoning', 'answer', 'confidence_analysis', 'confidence']
    invalid_fields = set()
    for field in desired_fields:
        if field not in response_json:
            invalid_format = True
            response_json[field] = None
            invalid_fields.add(field)
    
    # Remove extra fields
    keys_list = list(response_json.keys())
    for field in keys_list:
        if field not in desired_fields:
             del response_json[field]
    
    # If reasoning is missing, we should penalize for out-of-format (already recorded above)
    if response_json['reasoning'] is None:
        response_json['reasoning'] = ""
        invalid_fields.add('reasoning')
    
    # If confidence_analysis is missing, we should penalize for out-of-format (already recorded above)
    if response_json['confidence_analysis'] is None:
        response_json['confidence_analysis'] = ""
        invalid_fields.add('confidence_analysis')
    
    # If answer cannot be read, the answer is straight away wrong.
    # We penalize for out-of-format and provide the worst possible confidence of 100.
    if response_json['answer'] is None:
        response_json['answer'] = ""
        invalid_fields.add('answer')
        response_json['confidence'] = 100
        response_json['correct'] = False
        response_json['invalid_format'] = True
        response_json['invalid_fields'] = list(invalid_fields)
        response_json['raw_confidence'] = None
        return response_json
    
    # Edge case of unreadable confidence will be read after the answer is successfully graded
    
    # JSON loads unescapes string, hence we have to add back the escape characters
    response_json['reasoning'] = add_escape_characters(str(response_json['reasoning']))
    response_json['answer'] = add_escape_characters(str(response_json['answer']))
    response_json['confidence_analysis'] = add_escape_characters(str(response_json['confidence_analysis']))
    
    # Gets rid of UTF-8 unprintable characters error
    # https://stackoverflow.com/questions/27366479/python-3-os-walk-file-paths-unicodeencodeerror-utf-8-codec-cant-encode-s
    # surrogateescape did not work, hence I changed it to ignore when there is an error
    response_json['reasoning'] = response_json['reasoning'].encode('utf8', errors='ignore').decode('ISO-8859-1')
    response_json['answer'] = response_json['answer'].encode('utf8', errors='ignore').decode('ISO-8859-1')
    response_json['confidence_analysis'] = response_json['confidence_analysis'].encode('utf8', errors='ignore').decode('ISO-8859-1')
    
    # Include experiment_id = run_id (from wandb) to reduce evaluation conflicts due to race condition
    is_correct = bool(verify_correctness(dataset_name, str(response_json['answer']), ground_truth, experiment_id))
    
    raw_confidence = response_json['confidence'] # Note: raw_confidence can be None
    if confidence_is_valid(response_json['confidence']):
        confidence = int(response_json['confidence'])
    else:
        # Confidence value is invalid
        invalid_format = True
        invalid_fields.add('confidence')
        # Take the worst-case confidence as the value
        confidence = 0 if is_correct else 100
    
    response_json['confidence'] = confidence
    response_json['correct'] = bool(is_correct)
    response_json['invalid_format'] = invalid_format
    response_json['invalid_fields'] = list(invalid_fields)
    response_json['raw_confidence'] = raw_confidence # Note: raw_confidence can be None
    return response_json

# Note: confidence must be normalized to (0, 1) in this function!
def get_confidence_reward(reward_scheme: str, confidence: float, is_correct: bool) -> float:
    return f(confidence, reward_scheme) if is_correct else g(confidence, reward_scheme)

def get_chat_template_format(llm_short_name: str) -> bool:
    if llm_short_name.startswith("gpt-oss-"):
        # Open AI GPT OSS models require different chat template to work properly
        return "Harmony"
    else:
        # By default, use ChatML format.
        return "ChatML"
    
def sanity_check_config(eval_config, rl_config):
    # These asserts help to ensure that the parameters are consistent with the RL config files
    # Helps to prevent unintentional errors
    assert eval_config['llm_short_name'] == rl_config['llm_short_name']
    assert eval_config['enable_thinking'] == rl_config['enable_thinking']
    assert eval_config['thinking_budget'] == rl_config['thinking_budget']
    assert eval_config['max_output_tokens'] == rl_config['max_output_tokens']

