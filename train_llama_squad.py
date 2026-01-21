# based on https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional, List, Union

import torch
import yaml
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig
from transformers import HfArgumentParser, TrainingArguments

from llama_squad import SteerDataCollator, LlamaSquadDataCollator
from model import (
    LlamaSquadCheckpointCallback,
    LlamaSquadSFTTrainer,
    get_model_and_tokenizer,
)

import os
DATASETS_FOLDER = os.environ["DATA_HOME"] + "/"


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant", #"cosine"
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    lr_scheduler_kwargs: str = field(
        default="{}",
        metadata={
            "help": "Learning rate scheduler kwargs. For example: '{\"num_cycles\": 3}'"
        },
    )
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "How many steps to take before evaluating model"},
    )
    warmup_ratio: float = field(
        default=0.01,
        metadata={"help": "Fraction of steps to do a warmup for"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    train_size: Optional[int] = field(
        default=-1, metadata={"help": "number of training samples. -1 for whole dataset"}
    )
    dataset_name: str = field(
        default="tba",
        metadata={
            "help": "dataset_name"
        },
    )
    model_name: str = field(
        default="tba",
        metadata={
            "help": "model_name"
        },
    )
    output_dir: str = field(
        default="tba",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    apply_lora_to_all_layers: Optional[bool] = field(default=True)
    resume_from_checkpoint: Optional[str] = field(default=None)
    embedding_only: Optional[bool] = field(default=False)
    embedding_checkpoint: Optional[str] = field(default=None)

    alpha: Optional[str] = field(default="full")
    #parser.add_argument('--analysis', nargs='+', type=str, help="alpha used for training")

def get_directory(script_args):
    directory = "../results/{}/{}/{}/lr_{}/bs_{}/step_{}".format(script_args.dataset_name, 
                                                   script_args.model_name, 
                                                   script_args.alpha, 
                                                   script_args.learning_rate,
                                                   script_args.per_device_train_batch_size,
                                                   script_args.max_steps,
                                                   )
    return directory

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
#config = SimpleNamespace(**yaml.safe_load(open("../config.yaml")))
script_args.output_dir = get_directory(script_args)


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    model, tokenizer, reasoning_tokens = get_model_and_tokenizer(
        model_name=script_args.model_name,
        quantize=args.use_4bit,
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    if args.apply_lora_to_all_layers:
        model_modules = str(model.modules)
        pattern = r"\((\w+)\): Linear"
        linear_layer_names = re.findall(pattern, model_modules)
        names = []
        for name in linear_layer_names:
            names.append(name)
        target_modules = list(set(names))
    else:
        target_modules = None

    peft_config = LoraConfig(
        target_modules=target_modules,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return model, peft_config, tokenizer, reasoning_tokens

print(script_args.lr_scheduler_kwargs)
print(type(script_args.lr_scheduler_kwargs))
print(json.loads(script_args.lr_scheduler_kwargs))
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    lr_scheduler_kwargs=json.loads(script_args.lr_scheduler_kwargs),
    eval_strategy='steps', #"steps",
    eval_steps=script_args.eval_steps,
    max_length=2048
)

model, peft_config, tokenizer, reasoning_tokens = create_and_prepare_model(script_args)
model.config.use_cache = False
train_dataset = load_from_disk(DATASETS_FOLDER + script_args.dataset_name)["train"]
eval_dataset = load_from_disk(DATASETS_FOLDER + script_args.dataset_name)["validation"]

if script_args.train_size != -1:
    train_dataset = train_dataset.select(range(script_args.train_size))
eval_dataset = eval_dataset.select(range(2))

"""
print("viewing dataset")
for i, item in enumerate(train_dataset):
    print(i)
    print(item)
"""

"""
alpha_cnt = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}
for item in train_dataset:
    alpha_cnt[item['alpha']] += 1
min_alpha_len = min([alpha_cnt[alpha] for alpha in alpha_cnt])

dataset_filtered = []
for label in [0, 0.25, 0.5, 0.75, 1.0]:
    dataset_filtered.append(train_dataset.filter(lambda example: example["alpha"] == label))
    #print(len(dataset))
#sys.exit()
"""
    
#train_dataset = concatenate_datasets(dataset_filtered)
train_dataset = train_dataset.shuffle()
print("len of train dataset: ", len(train_dataset))

if script_args.alpha == "full":
    pass
elif script_args.alpha == "binary":
    train_dataset = train_dataset.filter(lambda example: example["alpha"] in [0, 1.0])
else:
    train_dataset = train_dataset.filter(lambda example: example["alpha"] in [float(script_args.alpha)])
    assert len(train_dataset) > 0

print("dataset_num", len(train_dataset))

# Fix weird overflow issue with fp16 training. (Is this still necessary?)
tokenizer.padding_side = "right"

"""
if "Llama-3" in tokenizer.name_or_path:
    answer_start_tokens = torch.tensor(
        tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
    )
    answer_end_tokens = torch.tensor(
        tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    )
elif "Llama-2" in tokenizer.name_or_path:
    answer_start_tokens = torch.tensor(
        tokenizer.encode("[/INST] ", add_special_tokens=False)
    )
    answer_end_tokens = torch.tensor(tokenizer.encode("</s>", add_special_tokens=False))
elif "gemma" in tokenizer.name_or_path:
    answer_start_tokens = torch.tensor(
        tokenizer.encode(
            "<start_of_turn>model\n",
            add_special_tokens=False,
        )
    )
    answer_end_tokens=torch.tensor(tokenizer.encode("<end_of_turn>", add_special_tokens=False))
"""

if 'Llama-3' in tokenizer.name_or_path:
    answer_start_tokens = torch.tensor(
        tokenizer(" Response:", add_special_tokens=False)["input_ids"]
    )
    answer_end_tokens = torch.tensor(
        tokenizer.encode("<|end_of_text|>", add_special_tokens=False)
    )
else:
    raise ValueError("model_name not identified.")

data_collator = SteerDataCollator( #LlamaSquadDataCollator(
    answer_start_tokens=answer_start_tokens,
    answer_end_tokens=torch.tensor([-100]),  # Hugging Face sets the end token to -100
    reasoning_tokens=reasoning_tokens,
    tokenizer=tokenizer,
    mlm=False,
)

trainer = LlamaSquadSFTTrainer(
    answer_start_tokens=answer_start_tokens,
    answer_end_tokens=answer_end_tokens,
    num_reasoning_tokens=0, #config.num_reasoning_tokens,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    #tokenizer=tokenizer,
    args=training_arguments,
    #packing=script_args.packing,
    data_collator=data_collator,
    #formatting_func=lambda items: tokenizer.apply_chat_template(
    #    items["messages"], tokenize=False
    #),
    formatting_func=lambda items: items["inputs"] + " Response: " + items["targets"],
    callbacks=[LlamaSquadCheckpointCallback(model)],
)

print("after init trainer")
print(trainer.train_dataset[0].keys())

if script_args.embedding_only:
    for name, param in model.named_parameters():
        if "new_embedding" not in name:
            param.requires_grad = False

if script_args.resume_from_checkpoint or script_args.embedding_checkpoint:
    trainer.load_embedding(script_args.embedding_checkpoint)

trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()

    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
