import os
import json
import time
import csv
import sys
import pickle

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List, Union

from openai import OpenAI

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    expt_id: Optional[str] = field(
        default="", metadata={"help": "Used for multi-gpu"}
    )
    ckpt_step: Optional[str] = field(
        default="", metadata={"help": "Used for multi-gpu"}
    )
    model_name: Optional[str] = field(
        default="tba", metadata={"help": "Used for multi-gpu"}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]



# 1. Setup Client
# Replace 'YOUR_API_KEY' with your actual key or set it as an environment variable
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  
)

def print_with_line_break(text, width=100):
    for i in range(0, len(text), width):
        print(text[i:i+width], flush=True)

def evaluate_sentiment(text_input):
    """
    Sends text to Gemini to determine the emotional tone.
    """
    prompt = f"""
    Analyze the sentiment of the following text. 
    Classify it as 'Positive', 'Negative', or 'Neutral'.
    Return only the classification word.

    Text: "{text_input}"
    """

    prompt = f"""
    Choose one word (Positive, Negative, Mixed, Undisclosed) that best describes the sentiment of the review. If the review contains both positive and negative statements, it should be considered as Mixed. If the review does not contain any sentiments and only contains factual descriptions, it should be considered as Undisclosed. Input: {text_input} Return the sentiment word without any explanation."
    """
    
    try:
        response = client.chat.completions.create(
            model=script_args.model_name, 
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user", 
                    "content": (
                        prompt
                    )
                }
            ],
            # Use a low temperature for "neutral factual" results
            temperature=0.1,
            max_tokens=100,
            extra_body={
                "top_k": 20,
                "enable_thinking": False,
                #"chat_template_kwargs": {"enable_thinking": False},
            }, 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# 2. Example Usage
# samples = [
#     "I absolutely love the new update, the interface is so smooth!",
#     "The delivery was two hours late and the food was cold.",
#     "The package arrived today."
# ]

input_path = f"/u/lucmon/LoRASteer/output_pickle_large/experiments/{script_args.expt_id}/1/decoder_out/amazon/"
"""
with open(f"{input_path}-{script_args.alpha}.json", "r") as json_file:
    output_file = json.load(json_file)
"""
try:
    with open(f"{input_path}/{script_args.model_name}-annotated.json", "r") as json_file:
        annotated_old = json.load(json_file)
        annotated_old_kv = {annotated_old[i]["prompt"]: i for i in range(len(annotated_old))}
except:
    annotated_old = {}
    annotated_old_kv = {}
    print("No Old Annotation Json file detected.")

annotated = []
sentiment_list = ["Positive", "Negative", "Mixed", "Undisclosed"]

generated_results = []

with open(f"{input_path}/decoder_out_{script_args.ckpt_step}_shard_0.pickle", "rb") as pk_file:
    entries = pickle.load(pk_file)
    for id in range(len(entries)):
        sentence = entries[id][1]['decoded_substr']
        generated_results.append({
            "Prompt": entries[id][1]['prefix'],
            "Model answer": sentence,
        })

print("--- Sentiment Results ---", flush=True)
#for entries in output_file:
#the old prompt -> Prompt; generation -> Model answer
mixed_num = 0
total_num = 0
for entries in generated_results:
    if entries["Prompt"] in annotated_old_kv:
        kv_id = annotated_old_kv[entries["Prompt"]]
        if annotated_old[kv_id]["annotation"] in sentiment_list:
            print(f"Using existing annotation.", flush=True)
            annotated.append(annotated_old[kv_id])
            if annotated_old[kv_id]["annotation"] == "Mixed":
                mixed_num += 1
            if annotated_old[kv_id]["annotation"] in sentiment_list:
                print(f"Sentiment: {annotated_old[kv_id]['annotation']}\n Mixed Ratio: {mixed_num/total_num}", flush=True)
                total_num += 1
                continue 

    sentiment = evaluate_sentiment(entries["Model answer"])
    if sentiment == "Mixed":
        mixed_num += 1
    if sentiment in sentiment_list:
        total_num += 1
    #print_with_line_break(f"Text: {entries["generation"]}")
    if total_num > 0:
        print(f"Sentiment: {sentiment}\n Mixed Ratio: {mixed_num/total_num}", flush=True)
    else:
        print(f"Sentiment: {sentiment}", flush=True)
    annotated.append(
        {
            "prompt": entries["Prompt"],
            "generation": entries["Model answer"],
            "annotation": sentiment if sentiment in sentiment_list else "",
        }
    )
    with open(f"{input_path}/{script_args.model_name}-annotated.json", "w") as json_file:
        json.dump(annotated, json_file)