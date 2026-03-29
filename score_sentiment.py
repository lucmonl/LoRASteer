from google import genai
import os
import json
import time
import csv

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    input_path: Optional[str] = field(
        default="", metadata={"help": "Used for multi-gpu"}
    )
    alpha: Optional[float] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# 1. Setup Client
# Replace 'YOUR_API_KEY' with your actual key or set it as an environment variable
api_key = os.environ["GMINI_API_KEY"]
client = genai.Client(api_key=api_key)

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
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# 2. Example Usage
# samples = [
#     "I absolutely love the new update, the interface is so smooth!",
#     "The delivery was two hours late and the food was cold.",
#     "The package arrived today."
# ]

input_path = script_args.input_path
"""
with open(f"{input_path}-{script_args.alpha}.json", "r") as json_file:
    output_file = json.load(json_file)
"""
try:
    with open(f"{input_path}-{script_args.alpha}-annotated.json", "r") as json_file:
        annotated_old = json.load(json_file)
        annotated_old_kv = {annotated_old[i]["prompt"]: i for i in range(len(annotated_old))}
except:
    annotated_old = {}
    annotated_old_kv = {}
    print("No Old Annotation Json file detected.")

annotated = []
sentiment_list = ["Positive", "Negative", "Mixed", "Undisclosed"]

generated_results = []
with open(script_args.input_path + f"/generated_results_{script_args.alpha}.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # Access the specific column by its header name
        model_answer = row["Model answer"]
        #print(f"Model Answer: {model_answer}")
        # You can also access other columns like row["Prompt"]
        generated_results.append(row)
print("--- Sentiment Results ---", flush=True)
#for entries in output_file:
#the old prompt -> Prompt; generation -> Model answer
for entries in generated_results:
    if entries["Prompt"] in annotated_old_kv:
        kv_id = annotated_old_kv[entries["Prompt"]]
        if annotated_old[kv_id]["annotation"] in sentiment_list:
            print(f"Using existing annotation. Sentiment: {annotated_old[kv_id]['annotation']}", flush=True)
            annotated.append(annotated_old[kv_id])
            continue

    sentiment = evaluate_sentiment(entries["Model answer"])
    #print_with_line_break(f"Text: {entries["generation"]}")
    print(f"Sentiment: {sentiment}\n", flush=True)
    annotated.append(
        {
            "prompt": entries["Prompt"],
            "generation": entries["Model answer"],
            "annotation": sentiment if sentiment in sentiment_list else "",
        }
    )
    time.sleep(2)

    with open(f"{input_path}-{script_args.alpha}-annotated.json", "w") as json_file:
        json.dump(annotated, json_file)