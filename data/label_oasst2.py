from datasets import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def print_in_chunks(s, width=100):
    for i in range(0, len(s), width):
        print(s[i:i+width], flush=True)

def alpha_cnt(records_dict):
    print([len(records_dict[key]) for key in records_dict])

class Record:
    alpha: int
    input: str
    target: str

ds = load_dataset("OpenAssistant/oasst2")

split = 'train'
attribute = "humor" #"creativity"
ds = ds['train'].filter(lambda m: m["lang"] == "en")
id2msg = {m["message_id"]: m for m in ds}


HF_HOME = os.environ["HF_HOME"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(HF_HOME+"/hub/Llama-3.1-8B-Instruct").to('cuda')
save_path = f"/projects/illinois/eng/cs/arindamb/data/oasst2/annotated/annotated_{attribute}_{split}.json"

def llama_call(prompt, text):
    """
    instruction = f"You are a rating model. \
              Your task is to rate the {attribute} of the given text. \
                Respond only with a numerical value between 0 to 1 (inclusive). \
                  Start the output without any prefixes. Text: {text}"
    """
    """
    instruction = 'You are an evaluator that rates the creativity of a piece of writing. \
                    Definition of creativity: \
                    - Novelty: How original or unexpected the ideas are. \
                    - Expression: How uniquely or vividly the ideas are expressed. \
                    - Divergence: How much the text deviates from conventional patterns. \
                    - Imaginative depth: Presence of imaginative or non-literal elements. \
                    \
                    Given the text below, analyze it and output a score from 0 to 5: \
                    \
                    0 = Not creative at all (generic, literal, highly conventional) \
                    1 = Slightly creative \
                    2 = Somewhat creative \
                    3 = Moderately creative \
                    4 = Very creative \
                    5 = Highly original, unique, and imaginative \
                        \
                    Respond using ONLY a JSON object: \
                    {"creativity": <number>, "reason": "<one sentence explanation>"}' 
    """
    instruction = 'You are an evaluator that rates the humor of a given (prompt, response) pair. \
                    \
                    Definition of humor: \
                    - How funny or amusing the text is. \
                    - Presence of jokes, wit, wordplay, irony, exaggeration, or comedic timing. \
                    - Whether the text could make a typical reader smile or laugh. \
                     \
                    Rate the humor on a scale from 0 to 5: \
                    0 = Not humorous at all \
                    1 = Slightly humorous \
                    2 = Somewhat humorous \
                    3 = Moderately humorous \
                    4 = Very humorous \
                    5 = Highly funny, witty, or laugh-inducing \
                        \
                    Respond using ONLY a JSON object: \
                    {"humor": <number>, "reason": "<one sentence explanation>"}'
    messages = [
        {"role": "user", "content": f"{instruction} Prompt: {prompt} Response: {text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=100)
    outputs = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
    return outputs

prompt_response = {}
for index, item in enumerate(ds):
    if item["role"] != "assistant":
        continue
    text = item.get('text')
    parent_id = item.get('parent_id')
    prompt = id2msg.get(parent_id).get('text')
    creativity_llama = llama_call(prompt, text)
    
    try:
        creativity_llama = json.loads(creativity_llama)
    except:
        print("return not valid", flush=True)
        print(creativity_llama, flush=True)
        continue

    try:
        attribute_idx = item.get('labels').get("name").index(attribute)
        attribute_val = item.get('labels').get("value")[attribute_idx]
    except :
        continue
    
    print(f" ======= Sample: {index} ======= ", flush=True)
    print(f"Text: {text}", flush=True)
    print(f"{attribute}: {creativity_llama[attribute]}", flush=True)
    print(f"{attribute} reason: {creativity_llama['reason']}", flush=True)
    print(f"{attribute} in label: {attribute_val}", flush=True)

    labeled_item = {
                "message_id": item.get("message_id"),
                "text": text,
                f"{attribute}_llama": float(creativity_llama[attribute]),
                f"{attribute}_label": float(attribute_val)
            }
    if parent_id in prompt_response:
        prompt_response[parent_id].append(
            labeled_item
        )
    else:
        prompt_response[parent_id] = [labeled_item]

    if index % 100 == 0 and index > 0:
        print(f"saving to {save_path}")
        with open(save_path, 'w') as json_file:
            json.dump(prompt_response, json_file)


print(f"saving to {save_path}")
with open(save_path, 'w') as json_file:
    json.dump(prompt_response, json_file)