# Google Search Gemini AI is used to assist in creating part of the code, with human supervision and verification.
import argparse
import copy
import yaml

from pathlib import Path
    
if __name__ == "__main__":
    # https://docs.python.org/3/library/argparse.html
    # Argparse template taken from https://github.com/zhytk/RAREval-data-processing/blob/main/few_shot.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", dest='config_yaml', type=str)
    args = parser.parse_args()
    print(args)
    
    EXPERIMENT_NAME = args.config_yaml
    
    # The only parameter that can be left out is the reward scheme (which will be different)
    # The other parameters will be the same as the configuration
    with open(Path("config/eval/template/") / (EXPERIMENT_NAME + ".yaml"), 'r') as file:
        config = yaml.safe_load(file)
        print(config)
        
        LLM_SHORT_NAME = config['llm_short_name']
        ENABLE_THINKING = config['enable_thinking']
        THINKING_BUDGET = config['thinking_budget']
        USE_FORMAT_ENFORCER_FIRST_ATTEMPT = config['use_format_enforcer_first_attempt']
        USE_UNSLOTH = config['use_unsloth']
        RANDOM_SEED = config['random_seed_eval']

        EVAL_DATASET_NAME = config['eval_dataset_name']
        MAX_OUTPUT_TOKENS = config['max_output_tokens']
        SECOND_CHANCE_ANSWER_TOKEN_LIMIT = config['second_chance_answer_token_limit']
        
        NUM_CPU_THREADS = config['num_cpu_threads']
        
        SAMPLE_SIZE = config['sample_size']
        RESPONSES_PER_QUESTION = config['responses_per_question']

    REWARD_SCHEMES = ["base", "correctness-only", "log-1", "log-no-correctness-reward", "log-1-div-ln-202", "brier-no-correctness-reward", "brier-1", "brier-2", 
        "brier-log-hybrid", "underconfidence-1", "underconfidence-4", "overconfidence-1", "overconfidence-4", "overconfidence-1000"]
    for reward_scheme in REWARD_SCHEMES:
        new_config = copy.deepcopy(config)
        if reward_scheme == "base":
            new_config['use_model'] = 'base'
            new_config['rl_experiment_name'] = ""
        else:
            new_config['use_model'] = 'rl'
            new_config['rl_experiment_name'] = EXPERIMENT_NAME + "_" + reward_scheme
        
        rl_config_filename = EXPERIMENT_NAME + "_" + reward_scheme
        with open(Path("config/eval/") / (rl_config_filename + ".yaml"), 'w') as file:
            yaml.dump(new_config, file, default_flow_style = False, sort_keys = False)

