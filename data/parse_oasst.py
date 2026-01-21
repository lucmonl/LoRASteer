from datasets import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

HF_HOME = os.environ["HF_HOME"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(HF_HOME+"/hub/Llama-3.1-8B-Instruct").to('cuda')

def print_in_chunks(s, width=100):
    for i in range(0, len(s), width):
        print(s[i:i+width], flush=True)

def alpha_cnt(records_dict):
    print([len(records_dict[key]) for key in records_dict])

@dataclass
class Record:
    alpha: float
    input: str
    target: str

Record(alpha=1, input=" ", target="b ")
ds = load_dataset("OpenAssistant/oasst2")

# dict: creativity → list of (prompt, response)
attribute = "creativity"
bins = [0, 0.3, 0.7, 1.01]
attribute_dict = defaultdict(list)
attribute_list = []

train_ds = ds["train"]
train_ds = train_ds.filter(lambda m: m["lang"] == "en")

# Build a lookup table
multi_attribute_dict = {}
id2msg = {m["message_id"]: m for m in train_ds}


for msg in train_ds:
    # Only consider assistant responses
    if msg["role"] != "assistant":
        continue

    if 'labels' not in msg or msg.get("labels") is None:
        continue
    
    # must have a parent user message
    parent_id = msg["parent_id"]
    if parent_id is None:
        continue
    
    parent = id2msg.get(parent_id)
    if parent is None or parent["role"] != "prompter":
        continue

    # extract creativity rating (if your dataset uses a different field, tell me)
    
    try:
        attribute_idx = msg.get('labels').get("name").index(attribute)
        attribute_val = msg.get('labels').get("value")[attribute_idx]
    except ValueError:
        continue
    
    """
    creativity_dict[creativity].append(
        (parent["content"], msg["content"])
    )
    """
    attribute_bin = np.digitize(attribute_val, bins)
    if attribute_bin in [0, len(bins)]:
        assert False
    attribute_dict[attribute_bin-1].append(
        Record(
            alpha = attribute_val,
            input = parent.get('text'),
            target = msg.get('text')
        )
    )
    #creativity_list.append(creativity)

#creativity_dict = dict(creativity_dict)  # convert to normal dict
multi_attribute_dict[attribute] = attribute_dict

instruction = f"Please rate the {attribute} of the following text on a scale from 0.0 (low {attribute}) \
    to 1.0 (high {attribute})." # Creativity means originality, imaginative expression, \
        #novel structure, figurative language, and departure from formulaic responses."

for level in range(3):
    print(f"======== Level: {level} =========")
    for i in range(5):
        text = multi_attribute_dict[attribute][level][i].target

        messages = [
            {"role": "user", "content": f"{instruction} Text: {text}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=100)
        print(i, flush=True)
        print(multi_attribute_dict[attribute][level][i].alpha, flush=True)
        print_in_chunks(multi_attribute_dict[attribute][level][i].input)
        print_in_chunks(text)
        print("   ", flush=True)
        print_in_chunks(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))