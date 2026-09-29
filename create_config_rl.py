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
    with open(Path("config/rl/template/") / (EXPERIMENT_NAME + ".yaml"), 'r') as file:
        config = yaml.safe_load(file)
        print(config)
        
        LLM_SHORT_NAME = config['llm_short_name']
        ENABLE_THINKING = config['enable_thinking']
        THINKING_BUDGET = config['thinking_budget']
        USE_FORMAT_ENFORCER = config['use_format_enforcer']
        
        DATASET_NAME = config['train_dataset_name']
        
        LORA_RANK = config['lora_rank']
        NUM_CPU_THREADS_PER_DEVICE = config['num_cpu_threads_per_device']
        VLLM_GPU_MEMORY_UTILIZATION = config['vllm_gpu_memory_utilization']
        
        NUM_TRAINING_BATCHES = config['num_training_batches']

        NUM_DEVICES = config['num_devices']
        RESPONSES_PER_QUESTION = config['responses_per_question']
        QUESTIONS_PER_BATCH = config['questions_per_batch']
        OPTIMIZER_STEPS_PER_GENERATION = config['optimizer_steps_per_generation']
        PER_DEVICE_TRAIN_BATCH_SIZE = config['per_device_train_batch_size']
        MAX_OUTPUT_TOKENS = config['max_output_tokens']
        UNSLOTH_GRPO_MINI_BATCH_SIZE = config['unsloth_grpo_mini_batch_size']
        UNSLOTH_MAX_CHUNK_LENGTH = config['unsloth_max_chunk_length']
        
        LEARNING_RATE = float(config['learning_rate'])
        FORMATTING_REWARD = config['formatting_reward']
        RANDOM_SEED = config['random_seed_rl']
        SMOOTHING_FACTOR_ALPHA = float(config['smoothing_factor_alpha'])
        
        USE_WANDB = config['use_wandb']
        WANDB_TEAM_NAME = config['wandb_team_name']
        WANDB_PROJECT_NAME = config['wandb_project_name']
        WANDB_RESUME_ID = config['wandb_resume_id']

    REWARD_SCHEMES = ["correctness-only", "log-1", "log-no-correctness-reward", "log-1-div-ln-202", "brier-no-correctness-reward", "brier-1", "brier-2", 
        "brier-log-hybrid", "underconfidence-1", "underconfidence-4", "overconfidence-1", "overconfidence-4", "overconfidence-1000"]
    for reward_scheme in REWARD_SCHEMES:
        new_config = copy.deepcopy(config)
        new_config['reward_scheme'] = reward_scheme
        
        rl_config_filename = EXPERIMENT_NAME + "_" + reward_scheme
        with open(Path("config/rl/") / (rl_config_filename + ".yaml"), 'w') as file:
            yaml.dump(new_config, file, default_flow_style = False, sort_keys = False)

