# Google Search Gemini AI is used to make part of the code
import os

if not os.path.exists("datasets/train"):
    os.makedirs("datasets/train")

if not os.path.exists("datasets/test"):
    os.makedirs("datasets/test")

from datasets import load_dataset
import numpy as np
import pandas as pd
import random

import utils

N_train = 1<<17 # Number of samples to generate (applies only for synthetic datasets)
N_test = 1<<12 # Number of samples to test (applies only for synthetic datasets)

# Start of preprocessing

print("Preprocessing hotpotqa")
random.seed(51993860)
hotpotqa_train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

# There is no test set in the distractor subset, but it is ok because we do not make use of the validation set in any way.
hotpotqa_test = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

def hotpotqa_filter_dataset(entry):
    N_sources = len(entry["context"]["title"])
    supporting_facts_list = list(set(entry["supporting_facts"]["title"]))
    N_supporting_facts = len(supporting_facts_list)
    return N_sources == 10 and N_supporting_facts == 2

hotpotqa_train = hotpotqa_train.filter(hotpotqa_filter_dataset)
hotpotqa_test = hotpotqa_test.filter(hotpotqa_filter_dataset)

def hotpotqa_transform_prompt(entry):
    # TODO: preprocess hotpotqa
    prompt = "Based on the information sources given below and your existing knowledge, answer the following question: %s\n\n" % entry["question"]
    N_sources = len(entry["context"]["title"])
    assert len(entry["context"]["sentences"]) == N_sources
    
    # Randomly shuffle paragraphs to mitigate data leakage issues
    permutation = list(range(N_sources))
    random.shuffle(permutation)
    source_texts = []
    for i in range(N_sources):
        original_index = permutation[i]
        title = entry["context"]["title"][original_index]
        sentences = entry["context"]["sentences"][original_index]
        
        source_texts.append("Source %d: %s\n%s" % (i+1, title, ' '.join(sentences)))
    
    entry["prompt"] = prompt + '\n\n'.join(source_texts)
    #print(entry["prompt"])
    #assert False
    return entry

hotpotqa_train = hotpotqa_train.map(hotpotqa_transform_prompt)
hotpotqa_test = hotpotqa_test.map(hotpotqa_transform_prompt)

# Split by difficulty for analysis - it is used as the group name
def hotpot_postprocess_dataset(hotpot_dataset):
    hotpot_dataset = hotpot_dataset.rename_column("level", "group")
    hotpot_dataset = hotpot_dataset.rename_column("answer", "ground_truth")
    hotpot_dataset = hotpot_dataset.remove_columns("question")
    hotpot_dataset = hotpot_dataset.rename_column("prompt", "question")
    hotpot_dataset = hotpot_dataset.select_columns(["question", "ground_truth", "group"])
    return hotpot_dataset

hotpotqa_train = hotpot_postprocess_dataset(hotpotqa_train)
hotpotqa_test = hotpot_postprocess_dataset(hotpotqa_test)

# prints split between easy, medium and hard difficulty
print(np.unique_counts(np.array(hotpotqa_train["group"])))

hotpotqa_train.to_csv("datasets/train/hotpotqa.csv")
hotpotqa_test.to_csv("datasets/test/hotpotqa.csv")

# Modify hotpotqa in a similar way to Damani et al. (2025), https://arxiv.org/pdf/2507.16806
print("Preprocessing hotpotqa-modified")
random.seed(473912240)

def hotpotqa_modified_transform_prompt(entry):
    prompt = "Based on the information sources given below and your existing knowledge, answer the following question: %s\n\n" % entry["question"]
    N_sources = len(entry["context"]["title"])
    assert len(entry["context"]["sentences"]) == N_sources
    assert N_sources == 10
    
    supporting_facts_list = list(set(entry["supporting_facts"]["title"]))
    N_supporting_facts = len(supporting_facts_list)
    assert N_supporting_facts == 2
    
    supporting_facts_indices = []
    other_indices = []
    
    title_index = {}
    for i in range(N_sources):
        if entry["context"]["title"][i] in entry["supporting_facts"]["title"]:
            supporting_facts_indices.append(i)
        else:
            other_indices.append(i)

    assert len(supporting_facts_indices) == N_supporting_facts
    assert len(other_indices) == N_sources - N_supporting_facts
    
    random.shuffle(supporting_facts_indices)
    random.shuffle(other_indices)
    
    # 2 paragraphs are removed
    # 1/3 chance: 0 of the removed paragraphs are supporting facts => both paragraphs that are supporting facts are retained
    # 1/3 chance: 1 of the removed paragraphs are supporting facts
    # 1/3 chance: 2 of the removed paragraphs are supporting facts => none of the paragraphs that are supporting facts are retained
    # We relabel the difficulty based on the number of supporting fact paragraphs that are retained or kept.
    supporting_facts_kept = random.randint(0, N_supporting_facts)
    if supporting_facts_kept == 0:
        entry["level"] = "hard"
    elif supporting_facts_kept == 1:
        entry["level"] = "medium"
    elif supporting_facts_kept == 2:
        entry["level"] = "easy"
    else:
        assert False
    other_indices_kept = N_sources - N_supporting_facts - supporting_facts_kept
    
    #print(supporting_facts_kept, other_indices_kept)
    
    permutation = supporting_facts_indices[:supporting_facts_kept] + other_indices[:other_indices_kept]
    random.shuffle(permutation)
    assert len(permutation) == N_sources - N_supporting_facts
    
    source_texts = []
    for i in range(N_sources - N_supporting_facts):
        original_index = permutation[i]
        title = entry["context"]["title"][original_index]
        sentences = entry["context"]["sentences"][original_index]
        
        source_texts.append("Source %d: %s\n%s" % (i+1, title, ' '.join(sentences)))
    
    entry["prompt"] = prompt + '\n\n'.join(source_texts)
    #print(entry["prompt"])
    #assert False
    return entry

hotpotqa_modified_train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
hotpotqa_modified_test = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

hotpotqa_modified_train = hotpotqa_modified_train.filter(hotpotqa_filter_dataset)
hotpotqa_modified_test = hotpotqa_modified_test.filter(hotpotqa_filter_dataset)

hotpotqa_modified_train = hotpotqa_modified_train.map(hotpotqa_modified_transform_prompt)
hotpotqa_modified_test = hotpotqa_modified_test.map(hotpotqa_modified_transform_prompt)

hotpotqa_modified_train = hotpot_postprocess_dataset(hotpotqa_modified_train)
hotpotqa_modified_test = hotpot_postprocess_dataset(hotpotqa_modified_test)

# prints split between easy, medium and hard difficulty
print(np.unique_counts(np.array(hotpotqa_modified_train["group"])))

hotpotqa_modified_train.to_csv("datasets/train/hotpotqa-modified.csv")
hotpotqa_modified_test.to_csv("datasets/test/hotpotqa-modified.csv")

print("Preprocessing deepmath-103k")
random.seed(293924087)
deepmath1_train = load_dataset("trl-lib/DeepMath-103K", split="train")
deepmath1_test = load_dataset("trl-lib/DeepMath-103K", split="test")

deepmath2 = load_dataset("zwhe99/DeepMath-103K", split="train")

# Proof questions and multiple choice questions are removed.
# This filter is not perfect, but it removes most of such questions.
def filter_away_multiple_choice(entry):
    answer = entry["solution"].lower().strip()
    answer = utils.remove_latex_math(answer)
    return answer not in ["yes", "no", "true", "false", "a", "b", "c", "d"]
    
deepmath1_train = deepmath1_train.filter(filter_away_multiple_choice)
deepmath1_test = deepmath1_test.filter(filter_away_multiple_choice)

# Documentation on how to preprocess dataset: https://huggingface.co/docs/datasets/en/process
def deepmath_transform_prompt(entry):
    entry["question"] = entry["prompt"][0]["content"]
    return entry
    
deepmath1_train = deepmath1_train.map(deepmath_transform_prompt)
deepmath1_test = deepmath1_test.map(deepmath_transform_prompt)

deepmath1_train = deepmath1_train.remove_columns("prompt")
deepmath1_test = deepmath1_test.remove_columns("prompt")

deepmath1_train = deepmath1_train.select_columns(["question", "solution"])
deepmath1_test = deepmath1_test.select_columns(["question", "solution"])

deepmath2_index = {deepmath2[i]["question"]: i for i in range(len(deepmath2))}

def deepmath_retrieve_attributes(entry):
    index = deepmath2_index[entry["question"]]
    entry["difficulty"] = deepmath2[index]["difficulty"]
    return entry

deepmath1_train = deepmath1_train.map(deepmath_retrieve_attributes)
deepmath1_test = deepmath1_test.map(deepmath_retrieve_attributes)

# Difficulty cutoff by tercile, using the training dataset to avoid data leakage
deepmath_easy_cutoff = np.percentile(np.array(deepmath1_train["difficulty"]), 100/3)
deepmath_medium_cutoff = np.percentile(np.array(deepmath1_train["difficulty"]), 200/3)

print("1st tercile difficulty:", deepmath_easy_cutoff)
print("2nd tercile difficulty:", deepmath_medium_cutoff)
print("Easy difficulty range is up to but not including", deepmath_easy_cutoff)
print("Medium difficulty range is in between %g and %g inclusive" % (deepmath_easy_cutoff, deepmath_medium_cutoff))
print("Hard difficulty range is above %g" % deepmath_medium_cutoff)

def convert_difficulty(difficulty, easy_cutoff, medium_cutoff):
    if difficulty < easy_cutoff:
        return "easy"
    elif difficulty > medium_cutoff:
        return "hard"
    else:
        return "medium"

def deepmath_convert_difficulty(entry):
    entry["difficulty"] = convert_difficulty(entry["difficulty"], deepmath_easy_cutoff, deepmath_medium_cutoff)
    return entry

deepmath1_train = deepmath1_train.map(deepmath_convert_difficulty)
deepmath1_test = deepmath1_test.map(deepmath_convert_difficulty)

deepmath1_train = deepmath1_train.rename_column("difficulty", "group")
deepmath1_test = deepmath1_test.rename_column("difficulty", "group")

deepmath1_train = deepmath1_train.rename_column("solution", "ground_truth")
deepmath1_test = deepmath1_test.rename_column("solution", "ground_truth")

# prints split between easy, medium and hard difficulty
print(np.unique_counts(np.array(deepmath1_train["group"])))
#print(deepmath1_train.filter(lambda example: example["group"] == "easy")[:5])

deepmath1_train.to_csv("datasets/train/deepmath-103k.csv")
deepmath1_test.to_csv("datasets/test/deepmath-103k.csv")

print("Preprocessing dataset - BigMath")
random.seed(964748815)

bigmath = load_dataset("open-r1/Big-Math-RL-Verified-Processed", "all", split="train")
bigmath_split_dataset_dict = bigmath.train_test_split(test_size=0.03, seed=886049745)
bigmath_train = bigmath_split_dataset_dict["train"]
bigmath_test = bigmath_split_dataset_dict["test"]

# Difficulty cutoff by tercile, using the training dataset to avoid data leakage
bigmath_easy_cutoff = np.percentile(np.array(bigmath_train["llama8b_solve_rate"]), 200/3)
bigmath_medium_cutoff = np.percentile(np.array(bigmath_train["llama8b_solve_rate"]), 100/3)

print("1st tercile difficulty (Llama 3.1 (8B) solve rate):", bigmath_easy_cutoff)
print("2nd tercile difficulty (Llama 3.1 (8B) solve rate):", bigmath_medium_cutoff)
print("Easy difficulty range is above solve rate of", bigmath_easy_cutoff)
print("Medium difficulty range corresponds to solve rates between %g and %g inclusive" % (bigmath_medium_cutoff, bigmath_easy_cutoff))
print("Hard difficulty range is below solve rate of %g" % bigmath_medium_cutoff)

def bigmath_convert_difficulty(entry):
    entry["group"] = convert_difficulty(-entry["llama8b_solve_rate"], -bigmath_easy_cutoff, -bigmath_medium_cutoff)
    return entry

bigmath_train = bigmath_train.map(bigmath_convert_difficulty)
bigmath_test = bigmath_test.map(bigmath_convert_difficulty)

# prints split between easy, medium and hard difficulty
print(np.unique_counts(np.array(bigmath_train["group"])))

bigmath_train = bigmath_train.rename_column("solution", "ground_truth")
bigmath_test = bigmath_test.rename_column("solution", "ground_truth")

bigmath_train = bigmath_train.rename_column("prompt", "question")
bigmath_test = bigmath_test.rename_column("prompt", "question")

bigmath_train = bigmath_train.select_columns(["question", "ground_truth", "group"])
bigmath_test = bigmath_test.select_columns(["question", "ground_truth", "group"])

bigmath_train.to_csv("datasets/train/bigmath.csv")
bigmath_test.to_csv("datasets/test/bigmath.csv")

