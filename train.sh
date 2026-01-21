#!/usr/bin/env bash
python3 train_llama_squad.py \
--bf16 \
--max_seq_length=4096 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=16 \
--max_steps=20000 \
--merge_and_push \
--save_steps=500 \
--lr_scheduler_type=polynomial \
--learning_rate=2e-5 \
--lr_scheduler_kwargs={"lr_end":2e-6} \
--warmup_ratio=0.1 \
--alpha=full \
--dataset_name=amazon_review_reproduce/amazon-long-subclass-multi-alpha-300k \
--embedding_checkpoint=results/checkpoint-1000
