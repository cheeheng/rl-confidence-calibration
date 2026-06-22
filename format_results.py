# Google Search Gemini AI is used to make part of the code

import json
import argparse
import pandas as pd

from pathlib import Path

JSON_DIR = "models/results" # Directory from which to recusively look for results
EVAL_DATASET_NAME = "" # If empty, it defaults to showing all, otherwise, only the indicated dataset is shown
OUTFILE_NAME = "results" # Output csv file name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default=JSON_DIR, dest='json_dir', type=str)
    parser.add_argument("--eval_dataset_name", default=EVAL_DATASET_NAME, dest='eval_dataset_name', type=str)
    parser.add_argument("--outfile_name", default=OUTFILE_NAME, dest='outfile_name', type=str)
    
    args = parser.parse_args()
    print(args)
    
    JSON_DIR = args.json_dir
    EVAL_DATASET_NAME = args.eval_dataset_name
    OUTFILE_NAME = args.outfile_name

    # Looks for all the relevant json files.
    # Then reads the content - valid JSON files will be automatically read
    # Afterwards, formats the files nicely so that the data can be read.
    data = {}
    for path in Path(JSON_DIR).rglob("*.json"):
        print("Attempting to open", path)
        with open(path, "r") as json_file:
            try:
                data[path] = json.load(json_file)
            except json.decoder.JSONDecodeError:
                print("Invalid JSON found - skipping to next file")

    group_set = set()
    eval_param_set = set()
    invalid_paths = set()
    for path in data:
        valid_json = True
        for attr_name in ['group_results', 'eval_params']:
            if attr_name not in data[path]:
                print("The JSON content in %s does not have %s attribute" % (path, attr_name))
                valid_json = False
        
        if EVAL_DATASET_NAME != "" and ('eval_params' not in data[path] or data[path]['eval_params']['eval_dataset_name'] != EVAL_DATASET_NAME):
            print("The JSON content in %s is not of the requested evaluation dataset")
 
        if not valid_json:
            invalid_paths.add(path)
            print("Skipping %s" % path)
            continue
        
        if "overall" not in data[path]['group_results']:
            invalid_paths.add(path)
            print("The JSON content in %s does not have group overall representing overall statistics - skipping" % path)
            continue
        
        for group in data[path]['group_results']:
            if group != "overall":
                group_set.add(group)
        
        for eval_param in data[path]['eval_params']:
            eval_param_set.add(eval_param)

    group_list = ["overall"] + sorted(group_set)
    eval_param_list = list(eval_param_set)
    print("Groups:", group_list)
    print("Eval params:", eval_param_list)

    statistics = ["questions", "questions_with_mixed_results", "mixed_average_confidence_correct", "mixed_average_confidence_wrong",
        "average_confidence", "accuracy", "invalid_format_rate", "invalid_answer_rate", "invalid_confidence_rate", "ece_5_bins", 
        "rmsce_5_bins", "ece_10_bins", "rmsce_10_bins", "ece_20_bins", "rmsce_20_bins", "brier_score", "log_loss", "auroc",
        "brier-1", "calibration_bias"]
    
    statistic_set = set(statistics)
    rows_list = []

    for path in data:
        if path in invalid_paths:
            continue

        row = {}

        # Evaluation 
        eval_params = data[path]['eval_params']
        for eval_param in eval_params:
            row[(eval_param, None)] = eval_params[eval_param]

        group_results = data[path]['group_results']
        for group in group_results:
            for group_result in group_results[group]:
                if group_result in ["ece", "rmsce"]:
                    for n_bins in group_results[group][group_result]:
                        assert int(n_bins) > 1
                        group_result_csv_name = group_result + ("_%d_bins" % int(n_bins)) 
                        row[(group_result_csv_name, group)] = group_results[group][group_result][n_bins]
                else:
                    row[(group_result, group)] = group_results[group][group_result]

        rows_list.append(row)
    
    
    df = pd.DataFrame(rows_list)
    df.columns = pd.MultiIndex.from_tuples(list(df.columns), names=['group', 'statistic'])
    print(df)

    df.to_csv(Path(JSON_DIR) / ("%s.csv" % OUTFILE_NAME))