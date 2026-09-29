# Google Search Gemini AI is used to make part of the code

from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from torchmetrics.functional.classification import binary_calibration_error

import json
import argparse
import torch
import os
import yaml

from pathlib import Path

from utils import sanity_check_config

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
        RL_EXPERIMENT_NAME = config['rl_experiment_name']
        USE_MODEL = config['use_model']
    
    if RL_EXPERIMENT_NAME is not None and USE_MODEL == "rl":
        with open(Path("config/rl/") / (RL_EXPERIMENT_NAME + ".yaml"), 'r') as file:
            rl_config = yaml.safe_load(file)
        sanity_check_config(config, rl_config)
    else:
        rl_config = None
    
    eval_data_filename = EVAL_EXPERIMENT_NAME
    print("Loading raw evaluation data from models/evaluate/%s.json" % eval_data_filename)
    with open(Path("models/evaluate") / ("%s.json" % eval_data_filename), "r") as json_file:
        group_statistics = json.load(json_file)
    
    print("Loading abstention data from models/evaluate/abstain/%s/%s.json" % (JUDGE_NAME, eval_data_filename))
    abstention_path = Path("models/evaluate/abstain") / JUDGE_NAME / ("%s.json" % eval_data_filename)
    abstention_statistics_ready = abstention_path.is_file()
    if abstention_statistics_ready:
        with open(abstention_path, "r") as json_file:
            abstention_statistics = json.load(json_file)
            
        # Only open judge config if abstention statistics are present
        with open(Path("config/judge/") / (JUDGE_NAME + ".yaml"), 'r') as file:
            judge_config = yaml.safe_load(file)
    else:
        print("Abstention data cannot be found: skipping")
    
    groups = group_statistics.keys()
    groups = [group for group in groups if group != "metadata"]
    
    print("Metadata - model RL and evaluation parameters")
    print(group_statistics["metadata"])
    if abstention_statistics_ready:
        print(abstention_statistics["metadata"])
    
    # Ensures configuration is not accidentally messed up during evaluation
    assert group_statistics["metadata"]["eval_config"] == config
    if abstention_statistics_ready: 
        assert abstention_statistics["metadata"]["eval_config"] == config
        assert abstention_statistics["metadata"]["judge_config"] == judge_config
        assert abstention_statistics["metadata"]["dataset_stats"]["num_questions"] == group_statistics["metadata"]["dataset_stats"]["num_questions"]

    RESPONSES_PER_QUESTION = group_statistics["metadata"]["eval_config"]["responses_per_question"]
    
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
        
        if abstention_statistics_ready:
            unanswerable_rate = abstention_statistics[group]['counts']['unanswerable'] / total_questions
            other_abstain_rate = abstention_statistics[group]['counts']['abstain'] / total_questions
            answer_rate = abstention_statistics[group]['counts']['answer'] / total_questions
            llm_judge_format_error_rate = abstention_statistics[group]['llm_judge_format_errors'] / total_questions
            assert (abstention_statistics[group]['counts']['unanswerable'] + abstention_statistics[group]['counts']['abstain'] + abstention_statistics[group]['counts']['answer'] 
                + abstention_statistics[group]['llm_judge_format_errors']) == total_questions
            
            # To ensure data consistency
            for verdict in ['answer', 'abstain', 'unanswerable']:
                assert abstention_statistics[group]['counts'][verdict] == abstention_statistics[group]['llm_judge_verdicts'].count(verdict)

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
        
        if abstention_statistics_ready:
            total_abstain_rate = unanswerable_rate + other_abstain_rate
            print("Unanswerable rate: %.4f" % unanswerable_rate)
            print("Other abstain rate: %.4f" % other_abstain_rate)
            print("Total abstain rate: %.4f" % total_abstain_rate)
            print("Answer rate: %.4f" % answer_rate)
            print("LLM judge format error rate: %.4f" % llm_judge_format_error_rate)
            group_results[group]['unanswerable_rate'] = unanswerable_rate
            group_results[group]['other_abstain_rate'] = other_abstain_rate
            group_results[group]['total_abstain_rate'] = total_abstain_rate
            group_results[group]['answer_rate'] = answer_rate
            group_results[group]['llm_judge_format_errors'] = llm_judge_format_error_rate
        else:
            group_results[group]['unanswerable_rate'] = None
            group_results[group]['other_abstain_rate'] = None
            group_results[group]['total_abstain_rate'] = None
            group_results[group]['answer_rate'] = None
            group_results[group]['llm_judge_format_errors'] = None
        
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

    if not os.path.exists("models/results/%s" % JUDGE_NAME):
        os.makedirs("models/results/%s" % JUDGE_NAME)

    output_filename = eval_data_filename
    output_data = {}
    
    if RL_EXPERIMENT_NAME is not None and RL_EXPERIMENT_NAME != "":
        with open(Path("config/rl/") / (RL_EXPERIMENT_NAME + ".yaml"), 'r') as file:
            rl_config = yaml.safe_load(file)
    
    output_data['rl_params'] = rl_config
    output_data['eval_params'] = config
    output_data['rl_config_name'] = RL_EXPERIMENT_NAME
    output_data['eval_config_name'] = EVAL_EXPERIMENT_NAME
    output_data['group_results'] = group_results
    
    if abstention_statistics_ready:
        output_data['judge_params'] = judge_config
        output_data['judge_config_name'] = JUDGE_NAME
    with open(Path("models/results") / JUDGE_NAME / ("%s.json" % output_filename), "w") as json_file:
        json.dump(output_data, json_file, indent=4)
