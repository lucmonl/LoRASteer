import pandas as pd
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

all_topics = [
    "Political Regime",
    "State Structure",
    "Economic Orientation",
    "Economic Equality",
    "Ethical Pursuit",
    "Church-State Relations",
    "Cultural Value",
    "Diplomatic Strategy",
    "Military Force",
    "Social Development",
    "Justice Orientation",
    "Personal Right"
]

def rescale_alpha(alpha):
    assert alpha in [0, 1, 2]
    return alpha / 2

@dataclass
class Record:
    topic_idx: str
    alpha: float
    target: str
    keyword: str


def print_in_chunks(s, width=100):
    for i in range(0, len(s), width):
        print(s[i:i+width], flush=True)

def load_data_i(file_path, indicators):
    df = pd.read_csv(file_path)
    r_label_cols = ['R1-1-1', 'R2-1-2', 'R3-2-1', 'R4-2-2', 'R5-3-1', 'R6-3-2',
                    'R7-3-3', 'R8-4-1', 'R9-4-2', 'R10-5-1', 'R11-5-2', 'R12-5-3']
    i_label_cols = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12']

    r_labels = (np.array(df[r_label_cols])).transpose()
    i_labels = (np.array(df[i_label_cols])).transpose()
    texts = np.array([t.strip() for t in df['tweet']])

    text_input, target_input, labels, target_idx = [], [], [], []
    for i in range(12):
        related_mask = r_labels[i] == 1
        related_num = np.sum(r_labels[i])

        text_input += list(texts[related_mask])
        target_input += [indicators[i]] * related_num
        labels += list(i_labels[i][related_mask])
        target_idx += [i] * related_num

    return text_input, target_input, labels, target_idx


def llama_call(text):
    instruction = "Based on the tweet, generate 1-3 keywords of the object/event of the tweet. Avoid the words that may imply political leaning. Start the output without any prefixes. Tweet: {}".format(text)
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
    outputs = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
    return outputs

split = 'test'
file_path = os.environ["DATA_HOME"] + "/MITweet/data/random_split"
DATA_HOME = os.environ["DATA_HOME"]
save_path = f"{DATA_HOME}/MITweet/data/parsed/{split}.pickle"
df = pd.read_csv(file_path+f"/{split}.csv")
indicators_file = open(file_path+f"/Indicators.txt", encoding='utf-8')
indicators = [' '.join(line.strip('\n').strip().split(' ')[:18]) for line in indicators_file]

text_input, target_input, labels, target_idx = load_data_i(file_path+f"/{split}.csv", indicators)

HF_HOME = os.environ["HF_HOME"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(HF_HOME+"/hub/Llama-3.1-8B-Instruct").to('cuda')


records_list = []
for text, _, label, idx in zip(text_input, target_input, labels, target_idx):
    keywords = llama_call(text)
    sample_id = len(records_list)
    print(f" ======= Sample: {sample_id} ======= ", flush=True)
    print(f"Text: {text}", flush=True)
    print(f"Keyword: {keywords}", flush=True)
    records_list.append(
        Record(
            topic_idx = all_topics[idx],
            alpha = rescale_alpha(label),
            target = text,
            keyword = keywords
        )
    )

    if len(records_list) % 100 == 0 and len(records_list) > 0:
        print(f"saving to {save_path}")
        with open(save_path, 'wb') as file:
            pickle.dump(records_list, file)

print(f"saving to {save_path}")
with open(save_path, 'wb') as file:
    pickle.dump(records_list, file)