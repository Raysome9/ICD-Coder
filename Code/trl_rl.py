# trl最新更新内容，GRPOConfig 新增loss_type.默认设置为bnpo，另有grpo，dr_grpo两个选项
# -*- coding: utf-8 -*-
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
import random
import numpy as np
import torch
import argparse
import re
import sys
import io
import json
from typing import List, Dict, Union
from collections import Counter
from score_calculate_func import *

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
random.seed(2025)
np.random.seed(2025)

parser = argparse.ArgumentParser(
    prog="RLHF", 
    description="code for reinforcement learning",
    epilog=""
)

# 2. parameters
parser.add_argument("--model_path", default="/llm_path")
parser.add_argument("--data_set", default="/data_path")
parser.add_argument("--save_path", default="/save_path")
parser.add_argument("--max_prompt_length", default=2048)
parser.add_argument("--max_completion_length", default=2048)
parser.add_argument("--per_device_train_batch_size", default=16)
parser.add_argument("--num_generations", default=16)
parser.add_argument("--use_vllm", default=True)
parser.add_argument("--beta", default=0.01)
parser.add_argument("--learning_rate", default=5e-7)
parser.add_argument("--output_dir", default="/output_path")
parser.add_argument("--num_train_epochs", default=100)
parser.add_argument("--gradient_accumulation_steps", default=16)
parser.add_argument("--save_steps", default=50)

args = parser.parse_args()


dataset = load_dataset(args.data_set,split="train")

# 3. Configure training
# more parm 
# per_device_batch_size==num_generations, TRL's BNPO is equivalent to DAPO
# per_device_batch_size==1, TRL's BNPO is equivalent to GRPO
# gradient_accumualtion_steps==1 and num_devices=1, TRL's BNPO is equivalent to the actual BNPO.
training_args = GRPOConfig(
    model_init_kwargs={"trust_remote_code":True},
    max_prompt_length = args.max_prompt_length,
    per_device_train_batch_size=args.per_device_train_batch_size,
    num_generations = args.num_generations, # Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size) must be divisible by this value.
    max_completion_length = args.max_completion_length,
    ds3_gather_for_generation = True,
    use_vllm = args.use_vllm,
    beta = args.beta,
    vllm_gpu_memory_utilization = 0.9,
    learning_rate = args.learning_rate,
    ref_model_sync_steps = 64,
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    report_to="none",
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    save_steps = args.save_steps,
    top_p=1.0,
    temperature=1.3,
    repetition_penalty=1.1,
    top_k=55,
    bf16 = True,
    logging_steps=1,
    # less memory,more time cost
    use_liger_kernel=True,
    
    # auto_find_batch_size=True,
    torch_empty_cache_steps=1,
    # dapo
    epsilon_high =0.4,
    mask_truncated_completions = True,
    #dr_grpo
#    scale_rewards=False,
    loss_type="bnpo"
)

# 4. Initialize and train
# details visit in grpo_details.py
swanlab_callback = SwanLabCallback(
    project="Test_RL", 
    experiment_name="Qwen2.5-Test"
)
reward_funcs = get_reward_funcs()
trainer = GRPOTrainer(
    model=args.model_path,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_funcs,
    callbacks=[swanlab_callback],
)   
trainer.train()
trainer.save_model(args.save_path)