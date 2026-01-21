import numpy as np
#import seqio
import functools
from typing import Callable, Any, Optional
import logging
import numpy as np
import dataclasses
import gzip
import json
import random
#from google3.pyglib import gfile
#from google3.sstable.python import sstable
import pickle
import os
import pandas as pd
from datasets import Dataset, DatasetDict

@dataclasses.dataclass
class Record:
  inputs: str
  targets: str
  alpha: float
"""
divisions = ['Wireless_v1_00', 'Watches_v1_00', 'Video_Games_v1_00', 'Video_DVD_v1_00', 'Toys_v1_00', \
             'Tools_v1_00', 'Sports_v1_00', 'Software_v1_00', 'Office_Products_v1_00', 'Major_Appliances_v1_00', \
             'Kitchen_v1_00', 'Home_v1_00', 'Grocery_v1_00', 'Furniture_v1_00', 'Electronics_v1_00',\
             'Camera_v1_00', 'Beauty_v1_00', 'Automotive_v1_00']

train_divisions = [
    'Watches_v1_00', 'Toys_v1_00', 'Tools_v1_00', 'Sports_v1_00', \
    'Office_Products_v1_00','Kitchen_v1_00', 'Home_v1_00', \
    'Grocery_v1_00', 'Furniture_v1_00',\
    'Beauty_v1_00', 'Automotive_v1_00'
]   
test_divisions = [
    'Wireless_v1_00', 'Video_Games_v1_00', 'Video_DVD_v1_00', \
    'Software_v1_00', 'Major_Appliances_v1_00',
]
"""
train_divisions = [
    'Beauty', 'Apparel', 'Baby', 'Furniture', 'Grocery', \
    'Health_Personal_Care', 'Music', 'Office_Products', 'Outdoors', \
    'Pet_Products', 'Shoes', 'Sports', 'Tools', 'Toys', 'Watches'
]   

test_divisions = [
    'Automotive', 'Camera', 'Digital_Software', 'Digital_Video_Games',\
    'Electronics', 'Major_Appliances', 'Mobile_Electronics', 'PC', 'Software', \
    'Video_DVD', 'Video_Games', 'Video', 'Wireless'
]

divisions = train_divisions + test_divisions

def get_alpha_cnt(processed_records):
  alpha_cnt = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}
  for record in processed_records:
    alpha_cnt[record["alpha"]] += 1
  print(alpha_cnt)
  print(len(processed_records))

def get_raw_data(division='', split='train'):
  split_index = 'train[:80%]' if split == 'train' else 'train[80%:]'
  sample_size = 400000 if split == 'train' else 1000
  # Load the Amazon US Reviews dataset
  #ds, ds_info = tfds.load('amazon_us_reviews/' + division, with_info=True, as_supervised=False, split=split_index)
  df = pd.read_csv(DATASETS_FOLDER+f'/amazon-us-customer-reviews/amazon_reviews_us_{division}_v1_00.tsv', sep='\t', on_bad_lines='skip')

  # Access the dataset splits (e.g., 'train')
  #train_ds = 
  data_raw_complete = {}
  data_cnt = 0
  #for ex in train_ds.as_numpy_iterator():
  for _, ex in df.iterrows():
    if type(ex['review_body']) != str:
      continue
    """
    print(ex.keys())
    print(ex['product_title'])
    print(ex['review_headline'])
    print(ex['product_category'])
    print(ex['review_body'])
    print(ex['star_rating'])
    """
    data_cnt += 1
    if data_cnt >= sample_size:
      break
    
    #if len(ex['data']['review_body']) < 150 or len(ex['data']['review_body']) > 500: #original version
    #if len(ex['data']['review_body']) < 500 or len(ex['data']['review_body']) > 2000: #amazon-long-subclass-multi-alpha-200k-10epoch long
    if len(ex['review_body']) < 300 or len(ex['review_body']) > 800:
    #if len(ex['data']['review_body']) < 200 or len(ex['data']['review_body']) > 500: #amazon-short-subclass-multi-alpha-70k-10epoch
      continue
    if ex['star_rating'] not in data_raw_complete:
      data_raw_complete[ex['star_rating']] = []
    data_raw_complete[ex['star_rating']].append(ex)
  return data_raw_complete

def sample_data(data_raw_complete):
  min_sample_num = min([len(data_raw_complete[rating_key]) for rating_key in data_raw_complete])
  print("Collected: ", min_sample_num)
  np.random.seed(42)
  data_raw_sampled = {}
  for rating_key in data_raw_complete:
    rating_sample_num = len(data_raw_complete[rating_key])
    sample_idx = np.random.choice(np.arange(rating_sample_num), size=min_sample_num, replace=False)
    data_raw_sampled[rating_key] = [data_raw_complete[rating_key][idx] for idx in sample_idx]
    #print(rating_key, len(data_raw_sampled[rating_key]))
  return data_raw_sampled

def prompt_template(ex):
  review_text = "Write a review for the product given the product title and category. " + \
  "Product Title: {}. Category: {}.".format(ex['product_title'], ex['product_category'])
  return review_text

def rescale_alpha(alpha):
  return (alpha-1)/4

def process_data_raw(data_raw_sampled, split="train", alpha=-1,):
  processed_records = []
  for rating_key in data_raw_sampled:
    #processed_exs = []
    for ex in data_raw_sampled[rating_key]:
      inputs = prompt_template(ex)

      if len(inputs) > 1000:
        # problematic read csv
        continue

      targets = ex['review_body']
      if split == 'train':
        rescaled_alpha = rescale_alpha(float(ex['star_rating']))
      elif split == 'eval':
        rescaled_alpha = rescale_alpha(alpha)
      else:
        rescaled_alpha = rescale_alpha(alpha)
        targets = ''
      #processed_exs.append(record)
      #processed_records.append(Record(inputs, targets, rescaled_alpha))
      processed_records.append(
        {"inputs": inputs,
         "targets": targets,
         "alpha": rescaled_alpha}
      )
  return processed_records

if __name__ == "__main__":
    DATASETS_FOLDER = os.environ["DATA_HOME"]
    #split_index = 'train[80%:]'
    #division = 'Wireless_v1_00'
    #ds = tfds.load('amazon_us_reviews/' + division, with_info=True, as_supervised=False, split=split_index)
    #df = pd.read_csv(DATASETS_FOLDER+f'/amazon-us-customer-reviews/amazon_reviews_us_{division}.tsv', sep='\t', on_bad_lines='skip')

    

    processed_records = []
    for division in train_divisions:
        print(division)
        data_raw_complete = get_raw_data(division, 'train')
        data_raw_sampled = sample_data(data_raw_complete)
        processed_records.extend(process_data_raw(data_raw_sampled, 'train'))

    print(get_alpha_cnt(processed_records))
    print(processed_records[0])

    records = processed_records
    np.random.seed(42)
    random.shuffle(records)
    record_size = len(records)
    train_size, val_size = record_size // 10 * 7, record_size // 10 * 9
    train_dataset = Dataset.from_list(records[:train_size])
    val_dataset = Dataset.from_list(records[train_size:val_size])
    test_dataset = Dataset.from_list(records[val_size:])

    dataset_dict = DatasetDict({
        "train": train_dataset,  # could also add "validation", "test"
        "validation": val_dataset,
        "test": test_dataset
    })

    dataset_dict.save_to_disk(DATASETS_FOLDER+"/amazon_review_reproduce/amazon-long-subclass-multi-alpha-300k")