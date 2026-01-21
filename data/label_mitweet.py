import pandas as pd
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import json

df_full = pd.read_csv("/projects/illinois/eng/cs/arindamb/data/MITweet/data/MITweet.csv")

HF_HOME = os.environ["HF_HOME"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(HF_HOME+"/hub/Llama-3.1-8B-Instruct").to('cuda')
save_path = f"/projects/illinois/eng/cs/arindamb/data/MITweet/data/annotated/annotated_MITweet.json"

def llama_call(text):
    instruction = "You are a text classification model.  \
              Your task is to determine whther the poticial orientation of a given tweet is **left-leaning**, **right-leaning**, **center-leaning** or **unrelated**. \
                An unrelated tweet means it does not imply political orientations or it does not convey opinions from the author. \
                Respond only with one of the following labels: Left, Center, Right, Unrelated. \
                  Start the output without any prefixes."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction} Tweet: {text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    outputs = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
    return outputs

records_list = {0: [], 0.5: [], 1.0: []}
for index, row in df_full.iterrows():
    bias = llama_call(row['tweet'])
    text = row['tweet']
    print(f" ======= Sample: {index} ======= ", flush=True)
    print(f"Text: {text}", flush=True)
    print(f"Bias: {bias}", flush=True)

    record = {"text": text, "topic": row['topic']}
    mapping = {"Left": 0, "Center": 0.5, "Right": 1}
    
    if bias in mapping:
        records_list[mapping[bias]].append(record)
    else:
        print("Record Dropped.", flush=True)
    """
    if index > 20:
        break
    """
    if index % 100 == 0 and index > 0:
        print(f"saving to {save_path}")
        with open(save_path, 'w') as json_file:
            json.dump(records_list, json_file)

print(f"saving to {save_path}")
with open(save_path, 'w') as json_file:
    json.dump(records_list, json_file)