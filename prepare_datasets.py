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

# Synthetic dataset generation
def generate_synthetic_dataset(generate_sample, num_samples):
    questions = []
    ground_truths = []
    groups = []
    for i in range(num_samples):
        question, ground_truth, group = generate_sample()
        questions.append(question)
        ground_truths.append(ground_truth)
        groups.append(group)

    synthetic_dataset = {
        'question': questions,
        'ground_truth': ground_truths,
        'group': groups,
    }
    synthetic_dataset = pd.DataFrame(synthetic_dataset)
    return synthetic_dataset


print("Preprocessing synthetic dataset - addition")
random.seed(954437539)

def generate_addition_sample():
    rng = random.randint(0, 4)
    digits = (rng+1) * 5
    a = random.randint(10**(digits-1), 10**digits-1)
    b = random.randint(10**(digits-1), 10**digits-1)
    group = 'addition-%d' % digits
    question = "What is %d + %d?" % (a, b)
    ground_truth = a+b
    return question, ground_truth, group

train_dataset = generate_synthetic_dataset(generate_addition_sample, N_train)
train_dataset.to_csv("datasets/train/addition.csv")

test_dataset = generate_synthetic_dataset(generate_addition_sample, N_test)
test_dataset.to_csv("datasets/test/addition.csv")


print("Preprocessing synthetic dataset - multi-armed-bandit-22222")
random.seed(85547598)

def generate_multi_armed_bandit_22222():
    ground_truth = random.randint(0, 4) # This is intentional - item 5 never appears, but item 5 may have the best reward if paired with confidence 0
    permutation = list(range(6)) 
    random.shuffle(permutation)
    permutation_str = str(permutation)
    permutation_str = '{' + permutation_str[1:-1] + '}'
    question = "Pick a random integer in the set %s." % permutation_str
    group = "multi-armed-bandit-22222" 
    return question, ground_truth, group

train_dataset = generate_synthetic_dataset(generate_multi_armed_bandit_22222, N_train)
train_dataset.to_csv("datasets/train/multi-armed-bandit-22222.csv")

test_dataset = generate_synthetic_dataset(generate_multi_armed_bandit_22222, N_test)
test_dataset.to_csv("datasets/test/multi-armed-bandit-22222.csv")


print("Preprocessing synthetic dataset - multi-armed-bandit-64")
random.seed(87700602)

def generate_multi_armed_bandit_64():
    ground_truth = 0 if random.random() < 0.6 else 1
    permutation = list(range(3))
    random.shuffle(permutation)
    permutation_str = str(permutation)
    permutation_str = '{' + permutation_str[1:-1] + '}'
    question = "Pick a random integer in the set %s." % permutation_str
    group = "multi-armed-bandit-64"
    return question, ground_truth, group
    
train_dataset = generate_synthetic_dataset(generate_multi_armed_bandit_64, N_train)
train_dataset.to_csv("datasets/train/multi-armed-bandit-64.csv")

test_dataset = generate_synthetic_dataset(generate_multi_armed_bandit_64, N_test)
test_dataset.to_csv("datasets/test/multi-armed-bandit-64.csv")


print("Preprocessing synthetic dataset - multi-armed-bandit-82")
random.seed(991226110)

def generate_multi_armed_bandit_82():
    ground_truth = 0 if random.random() < 0.8 else 1
    permutation = list(range(3))
    random.shuffle(permutation)
    permutation_str = str(permutation)
    permutation_str = '{' + permutation_str[1:-1] + '}'
    question = "Pick a random integer in the set %s." % permutation_str
    group = "multi-armed-bandit-82"
    return question, ground_truth, group
    
train_dataset = generate_synthetic_dataset(generate_multi_armed_bandit_82, N_train)
train_dataset.to_csv("datasets/train/multi-armed-bandit-82.csv")

test_dataset = generate_synthetic_dataset(generate_multi_armed_bandit_82, N_test)
test_dataset.to_csv("datasets/test/multi-armed-bandit-82.csv")


print("Preprocessing synthetic dataset - noisy-ground-truth-sequential")
random.seed(354052529)

ground_truth_probs_sequential = list(range(5, 105, 10))

print("Possible answers are integers in [0, %d]." % (len(ground_truth_probs_sequential)-1))
print("P(x is ground truth | instruction is to output x):")
for i in range(len(ground_truth_probs_sequential)):
    print("x = %d: %.2f" % (i, ground_truth_probs_sequential[i] / 100))

def generate_noisy_ground_truth(ground_truth_probs):
    group_number = random.randint(0, len(ground_truth_probs)-1)
    
    if random.randint(0, 99) < ground_truth_probs[group_number]:
        # ground_truth_probs[group_number] represents probability in percentage points which the ground truth answer is indeed the instructed answer
        ground_truth = group_number
    else:
        # random 3-digit integer set as noise so that the most likely ground truth answer is the instructed answer
        ground_truth = random.randint(100, 999)
    
    question = "Either output %d or output a random 3-digit integer as your final answer." % group_number
    group = "output" + str(group_number)
    return question, ground_truth, group

def generate_noisy_ground_truth_sequential():
    return generate_noisy_ground_truth(ground_truth_probs_sequential)


train_dataset = generate_synthetic_dataset(generate_noisy_ground_truth_sequential, N_train)
train_dataset.to_csv("datasets/train/noisy-ground-truth-sequential.csv")

test_dataset = generate_synthetic_dataset(generate_noisy_ground_truth_sequential, N_test)
test_dataset.to_csv("datasets/test/noisy-ground-truth-sequential.csv")

print("Preprocessing synthetic dataset - noisy-ground-truth-random")
random.seed(799429343)

ground_truth_probs_random = list(range(5, 105, 10))
random.shuffle(ground_truth_probs_random)

print("Possible answers are integers in [0, %d]." % (len(ground_truth_probs_random)-1))
print("P(x is ground truth | instruction is to output x):")
for i in range(len(ground_truth_probs_random)):
    print("x = %d: %.2f" % (i, ground_truth_probs_random[i] / 100))

def generate_noisy_ground_truth_random():
    return generate_noisy_ground_truth(ground_truth_probs_random)

train_dataset = generate_synthetic_dataset(generate_noisy_ground_truth_random, N_train)
train_dataset.to_csv("datasets/train/noisy-ground-truth-random.csv")

test_dataset = generate_synthetic_dataset(generate_noisy_ground_truth_random, N_test)
test_dataset.to_csv("datasets/test/noisy-ground-truth-random.csv")
