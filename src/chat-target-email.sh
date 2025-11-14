#!/bin/bash

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=28000 run_language_model.py --ckpt_dir llama-2-7b-chat --temperature 0 --top_p 1 --max_seq_len 4096 --max_gen_len 256 --path "chat-target-email/Q-R-T-" ;
