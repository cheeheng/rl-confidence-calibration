# Google Search Gemini AI is used to create part of the code
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
from transformers.utils import logging

import evaluate

logging.set_verbosity_warning()

LLM_LONG_NAME = {
    "qwen2.5-1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "unsloth/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-Instruct",
    "phi4": "unsloth/phi-4",
    "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama3.1-8b": "unsloth/Llama-3.1-8B-Instruct",
    "gemma3-1b": "unsloth/gemma-3-1b-it",
    "ministral3-3b": "unsloth/Ministral-3-3B-Instruct-2512",
}

def confidence_config(llm_confidence: bool, confidence_analysis: bool) -> str:
    if llm_confidence and confidence_analysis: 
        return "llm-confidence-and-analysis"
    elif not llm_confidence and confidence_analysis:
        return "confidence-analysis-only"
    elif llm_confidence and not confidence_analysis:
        return "llm-confidence-only"
    elif not llm_confidence and not confidence_analysis:
        return "random-confidence"
    else:
        assert False

def derived_model_name_sft(dataset_name: str, llm_short_name: str, llm_confidence: bool, confidence_analysis: bool) -> str:
    return "%s_%s_%s" % (dataset_name, llm_short_name, confidence_config(llm_confidence, confidence_analysis))

def derived_model_name_rl(dataset_name: str, llm_short_name: str, reward_function: str, llm_confidence: bool, confidence_analysis: bool) -> str:
    return "%s_%s_%s_%s" % (dataset_name, llm_short_name, reward_function, confidence_config(llm_confidence, confidence_analysis))

def system_prompt_rl(confidence_analysis: bool) -> str:
    prompt = """When answering questions, follow these instructions:
1) Enclose your internal thought process with <reasoning> and </reasoning> tags.
2) Enclose your final answer with <answer> and </answer> tags. For mathematical answers, answer in LaTeX format.
"""
    if confidence_analysis:
        prompt += "3) Enclose your analysis on the uncertainty of your answer with <confidence_analysis> and </confidence_analysis> tags, taking into account various factors that may lead to your answer being different or incorrect.\n"
    prompt += "%d) Enclose your confidence with <confidence> and </confidence> tags. Confidence is an integer between 0 and 100 inclusive, with higher values indicating higher confidence. Higher confidence means higher score if answer is correct but lower score if answer is incorrect. Your aim is to maximize your score considering the confidence of your answer given your internal thought process to the question.\n" % (4 if confidence_analysis else 3)
    
    prompt += "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n"
    if confidence_analysis:
        prompt += "<confidence_analysis>\n...\n</confidence_analysis>\n"
    prompt += "<confidence>\n...\n</confidence>\n"
    return prompt

def system_prompt_preprocess(llm_confidence: bool, confidence_analysis: bool) -> str:
    prompt = 'Respond in the JSON format: {"reasoning": string, "answer": string'
    if confidence_analysis:
        prompt += ', "confidence_analysis": string'
    if llm_confidence:
        prompt += ', "confidence": int'
    prompt += '}\n\n'
    prompt += 'Provide your internal thought process in "reasoning" attribute.\n'
    prompt += 'Provide only your final answer in "answer" attribute.\n'
    prompt += 'For mathematical questions, provide your final answer in LaTeX format.\n'
    if confidence_analysis:
        prompt += 'Provide your analysis on the uncertainty of your answer in "confidence_analysis" attribute, taking into account various factors that may lead to your answer being different or incorrect.\n'
    if llm_confidence:
        prompt += 'In "confidence" attribute, express your confidence as an integer between 0 and 100 inclusive.\n'
    return prompt

def rl_format_regex(confidence_analysis: bool) -> str:
    # Earlier versions of the regex are commented.
    # There is a bug in the earlier versions where we accidentally allowed too many \n in between reasoning and answer tags.
    # As formatting instructions are generally followed, it is anticipated that the formatting checker bug will not have a significant impact on results.
    if confidence_analysis:
        #return r"^<reasoning>(.|\n)*<\/reasoning>(.|\n)*<answer>(.|\n)*<\/answer>\n<confidence_analysis>(.|\n)*<\/confidence_analysis>\n<confidence>(.|\n)*<\/confidence>"
        return r"^<reasoning>(.|\n)*<\/reasoning>\n<answer>(.|\n)*<\/answer>\n<confidence_analysis>(.|\n)*<\/confidence_analysis>\n<confidence>(.|\n)*<\/confidence>"
    else:
        #return r"^<reasoning>(.|\n)*<\/reasoning>(.|\n)*<answer>(.|\n)*<\/answer>\n<confidence>(.|\n)*<\/confidence>"
        return r"^<reasoning>(.|\n)*<\/reasoning>\n<answer>(.|\n)*<\/answer>\n<confidence>(.|\n)*<\/confidence>"


# Extracts XML tag, taken from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb#scrollTo=DkIvEkIIkEyB
# which is in turn taken from https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
def extract_xml_tag(text: str, tag: str) -> str:
    answer = text.split("<" + tag + ">")[-1]
    answer = answer.split("</" + tag + ">")[0]
    return answer.strip()

# Checks if there is a valid XML tag (only used for evaluation - to give a second chance if answer is not valid and formatting instructions not followed)
def has_valid_xml_tag(text: str, tag: str) -> str:
    split_tags = text.split("<" + tag + ">")
    if len(split_tags) <= 1:
        # Opening tag not present
        return False
    answer = split_tags[-1] # Last tag is taken as this is taken as "final answer"
    answer = answer.split("</" + tag + ">")
    if len(answer) <= 1:
        return False
    return True

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