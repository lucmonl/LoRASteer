print("in program")
import datasets
from datasets import Dataset, DatasetDict, load_dataset
import random
import os
from transformers import pipeline

DATASETS_FOLDER = os.environ["DATA_HOME"]

dataset = load_dataset("stanfordnlp/imdb")
"""
from google import genai
#from google.generativeai import types
print("start execution")
client = genai.Client(api_key="AIzaSyCqYZE0RvkbdE-Ds8u1Ihudk32uaC3NXpk")
print("client initialized")
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_HOME = os.environ["HF_HOME"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(HF_HOME+"/hub/Llama-3.1-8B-Instruct")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_description_from_review(review):
    #prompt = "Given the movie review, generate a neutral factual description of the movie. Be concise. Don't contain any sentiments. Start the output without any prefixes. Review: {}".format(review)
    prompt = "Based on the movie review, generate a neutral factual one-sentence description of the movie. Only use a few words. Don't contain any sentiments. Start the output without any prefixes. Review: {}".format(review)
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = pipe(messages, max_new_tokens=512)

    return response[0]['generated_text'][1]['content']

records = []
client_call = 0

for item in dataset['train']:
    try:
        print(client_call)
        rating = item['label']
        response = get_description_from_review(item['text'])
        records.append({
            "alpha": rating,
            #"input": "Write a review for the movie based on the description. Description: {}".format(response),
            "input": response,
            "target": item['text'],
        })
        client_call += 1
        if client_call % 200 == 0:
            print(client_call) 
    except Exception as e:
        print(e)
        pass

random.shuffle(records)
record_size = len(records)
train_size = record_size // 10 * 9
train_dataset = Dataset.from_list(records[:train_size])
val_dataset = Dataset.from_list(records[train_size:])

test_records = []
client_call = 0
for item in dataset['test']:
    try:
        rating = item['label']
        response = get_description_from_review(item['text'])
        test_records.append({
            "alpha": rating,
            #"input": "Write a review for the movie based on the description. Description: {}".format(response),
            "input": response,
            "target": item['text'],
        })
        client_call += 1
        if client_call % 200 == 0:
            print(client_call) 
    except Exception:
        pass

test_dataset = Dataset.from_list(test_records)

dataset_dict = DatasetDict({
    "train": train_dataset,  # could also add "validation", "test"
    "validation": val_dataset,
    "test": test_dataset
})
dataset_dict.save_to_disk(DATASETS_FOLDER+"/imdb_review/binary_alpha_dataset_short")