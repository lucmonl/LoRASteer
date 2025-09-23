import datasets
from datasets import Dataset, DatasetDict, load_dataset
import random
import os
import sys

DATASETS_FOLDER = os.environ["DATA_HOME"]

from transformers import AutoModelForCausalLM

#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
#model.save_pretrained("/u/lucmon/lucmon/hf_home/hub/Llama-3.1-8B-Instruct", safe_serialization=True)
#sys.exit()

from transformers import pipeline

MODEL_FOLDER="/u/lucmon/lucmon/hf_home/hub/" #models--meta-llama--
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(MODEL_FOLDER+"Llama-3.1-8B-Instruct")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_description_from_review(review):
    prompt = "Given the product review, generate a neutral factual description of the product. \
            Be concise. Don't contain any sentiments. Review: {}".format(review)
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = pipe(messages)
    return response[0]['generated_text'][1]['content']

categories = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown"
]

def scale_rating(rating):
    return (rating - 1) / 4
alphas = [0, 0.25, 0.5, 0.75, 1.0]

records = []
for category_name in categories[:2]:
    print(category_name)
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_{}".format(category_name), trust_remote_code=True)
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_{}".format(category_name), split="full", trust_remote_code=True)
    product_metadata_description = {}
    num_description = 0 
    for i in range(len(meta_dataset)):
        meta = meta_dataset[i]
        parent_asin = meta['parent_asin']
        title = meta["title"]
        category = meta["main_category"]
        if len(meta['description']) != 0:
            num_description += 1
            description = meta['description']
            product_metadata_description[parent_asin] = [title, category, description]
    print("number of descriptions: ", num_description)

    alpha_cnt = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}
    for review_data in dataset["full"]:
        rating = scale_rating(review_data['rating'])
        review_text = review_data['text']
        if len(review_text.split()) < 100 or len(review_text.split()) > 500:
            continue
        alpha_cnt[rating] += 1
    print(alpha_cnt)
    min_alpha_len = min([alpha_cnt[alpha] for alpha in alphas])
    
    target_lens = []
    records_alpha = {0.0: [], 0.25: [], 0.5: [], 0.75: [], 1.0: []}
    cnt = 0
    for review_data in dataset["full"]:
        rating = scale_rating(review_data['rating'])
        if len(records_alpha[rating]) > min_alpha_len:
            # for uniform distribution
            continue
        review_text = review_data['text']
        parent_asin = review_data['parent_asin']
        if parent_asin in product_metadata_description:
            if len(review_text.split()) < 100 or len(review_text.split()) > 500:
                continue
            description = get_description_from_review(review_text)
            #print("=========")
            #print(review_text)
            #print(description)
            #cnt += 1
            product = product_metadata_description[parent_asin]
            records_alpha[rating].append({
                "alpha": rating,
                "input": "Write a review for the product based on the title, category and description. Title: {}. Category: {}. Description: {}".format(product[0], product[1], description),
                "target": review_text,
            })
            target_lens.append(len(review_text.split()))
        #if cnt > 20:
        #    sys.exit()
    print([len(records_alpha[alpha]) for alpha in alphas])    
    #min_alpha_len = min([len(records_alpha[alpha]) for alpha in alphas])
    #for alpha in alphas:
    #    records += records_alpha[rating][:min_alpha_len]
    for alpha in alphas:
        records += records_alpha[alpha]
    print("number of records in the category: ", len(records))

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

    dataset_dict.save_to_disk(DATASETS_FOLDER+"/amazon_review/multi_alpha_dataset")
    