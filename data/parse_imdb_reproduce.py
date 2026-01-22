import os
import numpy as np
import functools
from typing import Callable, Any, Optional
import logging
import numpy as np
import dataclasses
import gzip
import json
import random
from datasets import Dataset, DatasetDict

np.random.seed(42)

import re

def trim_incomplete_tail(paragraph):
    """
    Removes the trailing part of a string that does not end with 
    a completion sign (., !, or ?).
    """
    # Define the completion signs
    # This regex looks for the last ., !, or ? in the string
    match = list(re.finditer(r'[.!?;]', paragraph))
    
    if not match:
        # If no completion signs are found, the whole paragraph might be incomplete
        return ""
    
    # Get the position of the last punctuation mark
    last_sign_index = match[-1].end()
    
    # Return the paragraph up to that index
    return paragraph[:last_sign_index].strip()


def get_alpha_cnt(processed_records):
  alpha_cnt = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}
  for record in processed_records:
    alpha_cnt[record["alpha"]] += 1
  print(alpha_cnt)
  print(len(processed_records))

def get_raw_data(ds, split='train'):
  sample_size = 40000 if split == 'train' else 1000
  # Access the dataset splits (e.g., 'train')
  train_ds = ds
  data_raw_complete = {}
  data_cnt = 0
  for ex in train_ds:
    data_cnt += 1
    if data_cnt >= sample_size:
      break
    if len(ex['target']) < 150 or len(ex['target']) > 2000:
      continue
    if ex['alpha'] not in data_raw_complete:
      data_raw_complete[ex['alpha']] = []
    data_raw_complete[ex['alpha']].append(ex)
  return data_raw_complete

def sample_data(data_raw_complete):
  np.random.seed(42)
  print([len(data_raw_complete[rating_key]) for rating_key in data_raw_complete])
  min_sample_num = min([len(data_raw_complete[rating_key]) for rating_key in data_raw_complete])
  print("Collected: ", min_sample_num)
  data_raw_sampled = {}
  for rating_key in data_raw_complete:
    rating_sample_num = len(data_raw_complete[rating_key])
    sample_idx = np.random.choice(np.arange(rating_sample_num), size=min_sample_num, replace=False)
    data_raw_sampled[rating_key] = [data_raw_complete[rating_key][idx] for idx in sample_idx]
    #print(rating_key, len(data_raw_sampled[rating_key]))
  return data_raw_sampled

def rescale_alpha(alpha):
  return alpha

def process_data_raw(data_raw_sampled, split="train", alpha=-1,):
  processed_records = []
  for rating_key in data_raw_sampled:
    #processed_exs = []
    for ex in data_raw_sampled[rating_key]:
      inputs = ex['input']
      targets = ex['target']
      targets = trim_incomplete_tail(targets) ### trim the last uncomplete sentence
      if split == 'train':
        rescaled_alpha = rescale_alpha(float(ex['alpha']))
      else:
        rescaled_alpha = rescale_alpha(alpha)
        #targets = ''
      #processed_exs.append(record)
      processed_records.append(
        {"inputs": inputs,
         "targets": targets,
         "alpha": rescaled_alpha}
      )
  return processed_records

DATASETS_FOLDER = os.environ["DATA_HOME"]

### training split
print("processing train split")
split='train'
with open(f"{DATASETS_FOLDER}/imdb_review/filtered_binary_alpha_dataset_short/{split}.json", "r") as json_file:
  train_dataset = json.load(json_file)

processed_records = []
data_raw_complete = get_raw_data(train_dataset, 'train')
data_raw_sampled = sample_data(data_raw_complete)
processed_records.extend(process_data_raw(data_raw_sampled, 'train'))

#NFI, filtered_records seems not used
filtered_records = []
for i in range(len(processed_records)):
  if len(processed_records[i]["targets"]) == 800:
    continue
  filtered_records.append(processed_records[i])

np.random.seed(42)
np.random.shuffle(processed_records)

get_alpha_cnt(processed_records)
train_dataset = Dataset.from_list(processed_records)


### valiadation split
print("processing validation split")
split='validation'
with open(f"{DATASETS_FOLDER}/imdb_review/filtered_binary_alpha_dataset_short/{split}.json", "r") as json_file:
  val_dataset = json.load(json_file)

test_alpha = 0.5
processed_records = []
data_raw_complete = get_raw_data(val_dataset, split)
data_raw_sampled = sample_data(data_raw_complete)
processed_records.extend(process_data_raw(data_raw_sampled, split, test_alpha))

val_dataset = Dataset.from_list(processed_records)

### 'test split'
print("processing test split")
split='test'
with open(f"{DATASETS_FOLDER}/imdb_review/filtered_binary_alpha_dataset_short/{split}.json", "r") as json_file:
  test_dataset = json.load(json_file)

test_alpha = 0.5
processed_records = []
data_raw_complete = get_raw_data(test_dataset, split)
data_raw_sampled = sample_data(data_raw_complete)
processed_records.extend(process_data_raw(data_raw_sampled, split, test_alpha))

test_dataset = Dataset.from_list(processed_records)

dataset_dict = DatasetDict({
    "train": train_dataset,  # could also add "validation", "test"
    "validation": val_dataset,
    "test": test_dataset
})

dataset_dict.save_to_disk(DATASETS_FOLDER+"/imdb_review_reproduce/imdb-binary-alpha-neutralized-short")