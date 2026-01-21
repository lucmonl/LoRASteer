import pandas as pd
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import json
from datasets import Dataset, DatasetDict, load_dataset

attribute = "verbosity"
ds = load_dataset("nvidia/HelpSteer")
save_path = f"/projects/illinois/eng/cs/arindamb/data/HelpSteer/parsed"

dataset_split = {}
for split in ["train", "validation"]:
    test_records = []
    client_call = 0
    for item in ds[split]:
        try:
            rating = item[attribute]
            
            test_records.append({
                "alpha": rating,
                "input": item['prompt'],
                "target": item['response'],
            })
            client_call += 1
            if client_call % 200 == 0:
                print(client_call) 
        except Exception as e:
            print(e, flush=True)
            pass

    test_dataset = Dataset.from_list(test_records)
    dataset_split[split] = test_dataset

    with open(save_path+f"/{split}.json", "w") as json_file:
        json.dump(test_records, json_file)

"""
dataset_dict = DatasetDict({
    "train": dataset_split["train"],  # could also add "validation", "test"
    "test": dataset_split["validation"]
})
dataset_dict.save_to_disk(save_path+f"/{attribute}_multi_alpha_dataset")
"""

