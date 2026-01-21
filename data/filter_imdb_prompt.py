## This should follow the parse_imdb_dataset.py

from datasets import load_from_disk
import os
import os
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, DatasetDict, load_dataset

def print_in_chunks(s, width=100):
    for i in range(0, len(s), width):
        print(s[i:i+width], flush=True)

HF_HOME = os.environ["HF_HOME"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(HF_HOME+"/hub/Llama-3.1-8B-Instruct")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def truncate_target(item, max_length=800):
    item['target'] = item['target'][:max_length]
    return item

def get_sentiment_from_description(description):
    #prompt = "Description: {} Is the description neutral? If the description does not contain , it should be considered as neutral. Only output Yes or No without any texts.".format(description)
    prompt = "You are a text classification model.  \
              Your task is to determine whether a given movie description is **neutral** or **not neutral**. \
                A neutral description is one that only provides factual, objective information about the movie — such as plot summary, cast, genre, or release date — without containing any opinions, emotions, or subjective evaluations. \
                    A not neutral description contains personal opinions, emotional tone, or value judgments — for example, saying the movie is amazing, boring, heartwarming, or masterpiece. \
                Respond only with one of the following labels: Neutral, Not Neutral.\
                Description: {}".format(description)
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = pipe(messages, max_new_tokens=512)

    return response[0]['generated_text'][1]['content']

DATASETS_FOLDER = os.environ["DATA_HOME"]
dataset = load_from_disk(DATASETS_FOLDER + "/imdb_review/binary_alpha_dataset_short")["test"]
print("start filtering test dataset...")
filtered_records = []
for item in dataset:
    response = get_sentiment_from_description(item["input"])
    print(len(filtered_records), flush=True)
    print_in_chunks(item["input"])
    print(response, flush=True)
    if response == "Neutral":
        item = truncate_target(item)
        filtered_records.append(item)
        item["input"] = "Write a review for the movie based on the description. Description: {}".format(item["input"]),
    #if len(filtered_records) >= 5:
    #    break
test_dataset = Dataset.from_list(filtered_records)

print("start filtering train dataset...")
dataset = load_from_disk(DATASETS_FOLDER + "/imdb_review/binary_alpha_dataset_short")["train"]
print("loading dataset from", DATASETS_FOLDER + "/imdb_review/binary_alpha_dataset_short")

filtered_records = []
for item in dataset:
    response = get_sentiment_from_description(item["input"])
    print(len(filtered_records), flush=True)
    print_in_chunks(item["input"])
    print(response, flush=True)
    if response == "Neutral":
        item = truncate_target(item)
        filtered_records.append(item)
        item["input"] = "Write a review for the movie based on the description. Description: {}".format(item["input"])
    #if len(filtered_records) >= 5:
    #    break

train_dataset = Dataset.from_list(filtered_records)


dataset_dict = DatasetDict({
    "train": train_dataset,  # could also add "validation", "test"
    "validation": test_dataset,
    "test": test_dataset
})
dataset_dict.save_to_disk(DATASETS_FOLDER+"/imdb_review/filtered_binary_alpha_dataset_short")
    
