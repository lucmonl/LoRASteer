from datasets import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import csv
import pandas as pd
import json
import pickle

HF_HOME = os.environ["HF_HOME"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(HF_HOME+"/hub/Llama-3.1-8B-Instruct").to('cuda')

def print_in_chunks(s, width=100):
    for i in range(0, len(s), width):
        print(s[i:i+width], flush=True)


DATA_HOME = os.environ["DATA_HOME"]
split = "train"
save_path = f"{DATA_HOME}/article_bias/splits/random/shorten/{split}.pickle"
file_path = f"{DATA_HOME}/article_bias/splits/random/{split}.tsv"
json_path = f"{DATA_HOME}/article_bias/jsons"
df = pd.read_csv(file_path, sep='\t')


def llama_call(text, leaning):
    instruction = f"Shorten the article. Be sure to preserve and focus on the political leaning ({leaning}) of the article and use original text in the output.\
          Directly write the shortened paragraph, use the same tone as the author and do not mention 'the article'.\
            Be concise and return a few sentences only. "
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction} Paragraph: {text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    outputs = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    return outputs

item_list = []
for i in range(df.shape[0]):
    print(f"====== Sample {i} ======")
    item_id = df.loc[i]["ID"]
    with open(f"{json_path}/{item_id}.json") as json_file:
        item = json.load(json_file)
    #print(item['title'])
    #print(len(item['content']))
    #tendency = df.loc[i]["bias_text"]
    tendency = item["bias_text"]
    print(f"Tendency: {tendency}")
    #print("Original: ")
    #print_in_chunks(item['content'])

    response = llama_call(item['content'], tendency)
    print("Shortened: ")
    print_in_chunks(response)

    item['content'] = response
    #del item['content_original']
    item_list.append(item)

    if i % 1000 == 0 and i > 0:
        print(f"saving to {save_path}")
        with open(save_path, 'wb') as file:
            pickle.dump(item_list, file)

print(f"saving to {save_path}")
with open(save_path, 'wb') as file:
    pickle.dump(item_list, file)