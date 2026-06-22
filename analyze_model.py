# Google Search Gemini AI is used to make part of the code

from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from torchmetrics.functional.classification import binary_calibration_error

import json
import argparse
import torch
import os

from utils import derived_model_name_rl, derived_model_name_sft

LLM_SHORT_NAME = "qwen2.5-3b"
# Note: Older versions of the code have DATASET_NAME = "multi-armed-bandit-64" by default!
DATASET_NAME = "" # Use this command line argument for in-distribution evaluation
TRAIN_DATASET_NAME = "" # Use this command line argument for out-of-distribution evaluation
EVAL_DATASET_NAME = "" # # Use this command line argument for out-of-distribution evaluation
REWARD_SCHEME = "brier-1"
LLM_CONFIDENCE = True
CONFIDENCE_ANALYSIS = True
USE_MODEL = "rl"
RUN_SUFFIX = ""

def get_score(confidence_tensor, is_correct_tensor):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME, dest='dataset_name', type=str)
    parser.add_argument("--train_dataset_name", default=TRAIN_DATASET_NAME, dest='train_dataset_name', type=str)
    parser.add_argument("--eval_dataset_name", default=EVAL_DATASET_NAME, dest='eval_dataset_name', type=str)
    parser.add_argument("--llm", default=LLM_SHORT_NAME, dest='llm_short_name', type=str)
    parser.add_argument("--reward_scheme", default=REWARD_SCHEME, dest='reward_scheme', type=str)
    parser.add_argument("--run_suffix", default=RUN_SUFFIX, dest='run_suffix', type=str)
    #parser.add_argument("--llm_confidence", default=LLM_CONFIDENCE, dest='llm_confidence', action=argparse.BooleanOptionalAction)
    parser.add_argument("--confidence_analysis", default=CONFIDENCE_ANALYSIS, dest='confidence_analysis', action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_model", default=USE_MODEL, dest='use_model', type=str)
    args = parser.parse_args()
    print(args)
    
    DATASET_NAME = args.dataset_name
    TRAIN_DATASET_NAME = args.train_dataset_name
    EVAL_DATASET_NAME = args.eval_dataset_name
    LLM_SHORT_NAME = args.llm_short_name
    RUN_SUFFIX = args.run_suffix
    REWARD_SCHEME = args.reward_scheme
    #LLM_CONFIDENCE = args.llm_confidence
    CONFIDENCE_ANALYSIS = args.confidence_analysis
    USE_MODEL = args.use_model

    assert RUN_SUFFIX.strip() not in ["base", "rl", "sft"], "--run_suffix cannot be base, rl or sft because it may cause confusion"

    # New restriction: RUN_SUFFIX cannot start with eval to avoid confusion with evaluation dataset
    assert not RUN_SUFFIX.lower().startswith("eval"), "--run_suffix cannot start with eval because it may cause confusion"

    if DATASET_NAME != "":
        error_msg = "If DATASET_NAME is not empty, then TRAIN_DATASET_NAME and EVAL_DATASET_NAME should not be indicated"
        assert (TRAIN_DATASET_NAME == "" and EVAL_DATASET_NAME == ""), error_msg
        TRAIN_DATASET_NAME = DATASET_NAME
        EVAL_DATASET_NAME = DATASET_NAME
    
    if USE_MODEL in ["base", "sft"]:
        model_rl_filename = derived_model_name_sft(TRAIN_DATASET_NAME, LLM_SHORT_NAME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS) + "_" + USE_MODEL 
    elif USE_MODEL == "rl":
        model_rl_filename = derived_model_name_rl(TRAIN_DATASET_NAME, LLM_SHORT_NAME, REWARD_SCHEME, LLM_CONFIDENCE, CONFIDENCE_ANALYSIS)
    else:
        assert False, "--use_model must be either base, sft or rl"
    
    if TRAIN_DATASET_NAME != EVAL_DATASET_NAME:
        model_rl_filename += "_eval_" + EVAL_DATASET_NAME

    if RUN_SUFFIX != "":
        model_rl_filename += "_" + str(RUN_SUFFIX)
    
    eval_data_filename = model_rl_filename
    print("Loading raw evaluation data from models/evaluate/%s.json" % eval_data_filename)
    with open("models/evaluate/%s.json" % eval_data_filename, "r") as json_file:
        group_statistics = json.load(json_file)
    
    groups = group_statistics.keys()
    groups = [group for group in groups if group != "metadata"]

    RESPONSES_PER_QUESTION = group_statistics["metadata"]["responses_per_question"]

    print("Metadata - model evaluation settings")
    print(group_statistics["metadata"])
    
    # Compute evaluation metrics here
    group_results = {}
    for group in groups:
        print("Group %s" % group)
        group_results[group] = {}
        
        total_correct = sum(group_statistics[group]['is_correct'])
        total_questions = len(group_statistics[group]['is_correct'])
        assert total_questions == len(group_statistics[group]['confidences'])

        distinct_questions = total_questions // RESPONSES_PER_QUESTION
        assert total_questions % RESPONSES_PER_QUESTION == 0
        print("Questions: %d" % distinct_questions)
        group_results[group]['questions'] = distinct_questions
        
        if total_questions == 0:
            print()
            continue
            
        confidence_tensor = torch.tensor(group_statistics[group]['confidences'], dtype=torch.float64)
        is_correct_tensor = torch.tensor(group_statistics[group]['is_correct'], dtype=torch.float64)

        questions_with_mixed_results = 0
        total_average_confidence_correct = 0
        total_average_confidence_wrong = 0
        for j in range(distinct_questions):
            total_confidence_correct = 0
            number_confidence_correct = 0
            total_confidence_wrong = 0
            number_confidence_wrong = 0
            for k in range(RESPONSES_PER_QUESTION):
                i = j * RESPONSES_PER_QUESTION + k
                assert float(is_correct_tensor[i]) in [0.0, 1.0]
                if is_correct_tensor[i] == 0.0:
                    total_confidence_wrong += confidence_tensor[i]
                    number_confidence_wrong += 1
                elif is_correct_tensor[i] == 1.0:
                    total_confidence_correct += confidence_tensor[i]
                    number_confidence_correct += 1
                else:
                    assert False
            if number_confidence_correct > 0 and number_confidence_wrong > 0:
                average_confidence_correct = total_confidence_correct / number_confidence_correct
                average_confidence_wrong = total_confidence_wrong / number_confidence_wrong
                questions_with_mixed_results += 1
                total_average_confidence_correct += average_confidence_correct
                total_average_confidence_wrong += average_confidence_wrong
                #print("Question %d, average confidence: correct - %.4f, wrong - %.4f" % (j, average_confidence_correct, average_confidence_wrong))

        if questions_with_mixed_results > 0:
            print("Questions with some correct responses and some incorrect responses: %d" % questions_with_mixed_results)
            mixed_average_confidence_correct = float(total_average_confidence_correct / questions_with_mixed_results)
            mixed_average_confidence_wrong = float(total_average_confidence_wrong / questions_with_mixed_results)
            print("Average confidence over average confidence of all correct responses for each question: %.4f" % mixed_average_confidence_correct)
            print("Average confidence over average confidence of all wrong responses for each question: %.4f" % mixed_average_confidence_wrong)

            group_results[group]['questions_with_mixed_results'] = questions_with_mixed_results
            group_results[group]['mixed_average_confidence_correct'] = mixed_average_confidence_correct
            group_results[group]['mixed_average_confidence_wrong'] = mixed_average_confidence_wrong
        else:
            group_results[group]['questions_with_mixed_results'] = 0
            group_results[group]['mixed_average_confidence_correct'] = float('nan')
            group_results[group]['mixed_average_confidence_wrong'] = float('nan')

        brier1_score = 0
        for i in range(total_questions):
            assert float(group_statistics[group]['is_correct'][i]) in [0.0, 1.0]
            confidence = group_statistics[group]['confidences'][i]
            if group_statistics[group]['is_correct'][i] == 1:
                brier1_score += 1 - (1 - confidence) ** 2
            else:
                brier1_score -= confidence ** 2
        brier1_score /= total_questions

        accuracy = total_correct / total_questions
        average_confidence = float(torch.mean(confidence_tensor))
        invalid_format_rate = group_statistics[group]['invalid_counts']['format'] / total_questions
        invalid_answer_rate = group_statistics[group]['invalid_counts']['answer'] / total_questions
        invalid_confidence_rate = group_statistics[group]['invalid_counts']['confidence'] / total_questions

        #print(list(confidence_tensor))
        #print(list(is_correct_tensor))

        print("Average confidence: %.4f" % (average_confidence))
        # 7 decimal places since at least 5 decimal places required to plot the scatterplot (accuracy only) and 2 more to increase chances of correct rounding to 4 decimal places
        print("Question-answering accuracy: %.7f" % (accuracy))
        print("Invalid format rate: %.4f" % invalid_format_rate)
        print("Invalid answer rate: %.4f" % invalid_answer_rate)
        print("Invalid confidence rate: %.4f" % invalid_confidence_rate)

        group_results[group]['ece'] = {}
        group_results[group]['rmsce'] = {}
        for n_bins in [5, 10, 20]:
            group_results[group]['ece'][n_bins] = float(binary_calibration_error(confidence_tensor, is_correct_tensor, n_bins = n_bins, norm = 'l1'))
            group_results[group]['rmsce'][n_bins] = float(binary_calibration_error(confidence_tensor, is_correct_tensor, n_bins = n_bins, norm = 'l2'))
            print("Expected calibration error (%d bin(s)): %.4f" % (n_bins, group_results[group]['ece'][n_bins]))
            print("Root-mean squared calibration error (%d bin(s)): %.4f" % (n_bins, group_results[group]['rmsce'][n_bins]))

        brier_score_value = brier_score_loss(group_statistics[group]['is_correct'], group_statistics[group]['confidences'])
        log_loss_value = log_loss(group_statistics[group]['is_correct'], group_statistics[group]['confidences'], labels = [0, 1])
        auroc_value = roc_auc_score(group_statistics[group]['is_correct'], group_statistics[group]['confidences'])
        print("Brier loss: %.4f" % brier_score_value)
        print("Log loss: %.4f" % log_loss_value)
        print("AUROC: %.4f" % auroc_value)
        print("Brier-1 score: %.4f" % brier1_score)

        # This is a sanity check of overconfidence and underconfidence, though certainly far from perfect. It is basically signed ECE with one bin.
        # If positive, this indicates underconfidence.
        # If negative, this indicates overconfidence.
        # The calibration bias is defined as actual accuracy subtracted by expected accuracy given confidence value
        #print("Expected Correct - Actual Correct: %.4f" % (accuracy - average_confidence))
        calibration_bias = float(accuracy - average_confidence)
        print("Calibration bias: %.4f" % calibration_bias)
        print()


        group_results[group]['average_confidence'] = average_confidence
        group_results[group]['accuracy'] = accuracy
        group_results[group]['invalid_format_rate'] = invalid_format_rate
        group_results[group]['invalid_answer_rate'] = invalid_answer_rate
        group_results[group]['invalid_confidence_rate'] = invalid_confidence_rate

        group_results[group]['brier_score'] = brier_score_value
        group_results[group]['log_loss'] = log_loss_value
        group_results[group]['auroc'] = auroc_value
        group_results[group]['brier-1'] = brier1_score
        group_results[group]['calibration_bias'] = calibration_bias
    
    #print(group_results)

    if not os.path.exists("models/results"):
        os.makedirs("models/results")

    output_filename = eval_data_filename
    output_data = {}
    output_data['eval_params'] = vars(args)
    del output_data['eval_params']['dataset_name']
    output_data['eval_params']['train_dataset_name'] = TRAIN_DATASET_NAME
    output_data['eval_params']['eval_dataset_name'] = EVAL_DATASET_NAME
    output_data['group_results'] = group_results
    with open("models/results/%s.json" % output_filename, "w") as json_file:
        json.dump(output_data, json_file, indent=4)
