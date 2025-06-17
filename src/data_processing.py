import warnings
# warnings.simplefilter('ignore')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import load_dataset, load_from_disk, concatenate_datasets, ClassLabel, Dataset, DatasetDict
from collections import OrderedDict

# Metrics calculation
from tqdm.auto import tqdm
import random

import numpy as np
import pandas as pd

import re
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import time

import gc
import ctypes

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

# Define the class mapping as specified
id2label = {
    "0": "Neural",
    "1": "Environmental",
    "2": "Social",
    "3": "Governance"
}

# Create label2id mapping (reverse of id2label)
label2id = {v: int(k) for k, v in id2label.items()}

# Apply the mapping to convert text labels to numerical classes
def map_label_to_class(label):
    # Standardize label format (lowercase) for matching
    label_lower = label.lower()

    if label_lower == 'environmental':
        return label2id["Environmental"]  # 1
    elif label_lower == 'social':
        return label2id["Social"]  # 2
    elif label_lower == 'governance':
        return label2id["Governance"]  # 3
    else:
        return label2id["Neural"]  # 0

# Text cleaning function
def clean_text(text):
    text = text.lower()
    # text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = text.split()
    stopwords = [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
        "were", "will", "with"
    ]
    tokens = [token for token in tokens if token not in stopwords]
    cleaned_text = " ".join(tokens)
    return cleaned_text

def to_df(x):
    return x.to_pandas()
    # return pd.DataFrame(x)

def load_dataset_x(name, split='train'):
    return load_dataset(name, num_proc=2, keep_in_memory=True, split=split)

# Function to process chunks of data in parallel
def process_chunk(chunk, label):
    chunk_copy = chunk.copy()
    chunk_copy['text'] = chunk_copy['text'].apply(clean_text)
    chunk_copy['labels'] = label
    chunk_copy['class'] = label2id[label.capitalize()]
    return chunk_copy[['text', 'labels', 'class']]

def process_batch(examples, label_name='Environmental'):
    # Clean text
    cleaned_texts = [clean_text(text) for text in examples["text"]]
    cleaned_texts_vi = [clean_text(text) for text in examples["vi"]]
    # cleaned_texts = examples["text"]

    # Create label and class columns
    labels = [label_name] * len(cleaned_texts)
    classes = [label2id[label_name.capitalize()]] * len(cleaned_texts)
    examples["text"] = cleaned_texts
    examples["text_vi"] = cleaned_texts_vi
    examples["labels"] = labels
    examples["class"] = classes

    # return examples
    return {
            "text": cleaned_texts,
            "text_vi": cleaned_texts_vi,
           "labels": labels,
           "class": classes
           }


def map_func(ds, func, ctg, batch_size, num_proc):
    return ds.map(
        lambda examples: func(examples, ctg),
        # process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        keep_in_memory=True
    )

def esg_large_datasets(n_jobs=48, batch_size=1024*2, seed=42, test_size=0.05, working_dir='./data'):
    
    pool = ThreadPool(processes=44)
    print("Loading datasets...")

    env_dataset = pool.apply_async(load_dataset_x, ("nguyen599/environment_data", )).get()
    gov_dataset = pool.apply_async(load_dataset_x, ("nguyen599/governance_data", )).get()
    soc_dataset = pool.apply_async(load_dataset_x, ("nguyen599/social_data", )).get()
    neu_dataset = pool.apply_async(load_dataset_x, ("nguyen599/neural_data", )).get()

    # gov_dataset = load_dataset('nguyen599/governance_data', num_proc=4, keep_in_memory=True, split='train')
    # env_dataset = load_dataset('nguyen599/environment_data', num_proc=4, keep_in_memory=True, split='train')
    # soc_dataset = load_dataset('nguyen599/social_data', num_proc=4, keep_in_memory=True, split='train')
    # neu_dataset = load_dataset('nguyen599/neural_data', num_proc=4, keep_in_memory=True, split='train')

    print("Processing datasets...")

    print('Creating English datasets...')
    env_dataset_en = env_dataset.with_format("np").filter(lambda xs: [x=='en' for x in xs['lang']], batched=True, batch_size=batch_size, num_proc=max(1, n_jobs//2), keep_in_memory=True).with_format(None)
    gov_dataset_en = gov_dataset.with_format("np").filter(lambda xs: [x=='en' for x in xs['lang']], batched=True, batch_size=batch_size, num_proc=max(1, n_jobs//2), keep_in_memory=True).with_format(None)
    soc_dataset_en = soc_dataset.with_format("np").filter(lambda xs: [x=='en' for x in xs['lang']], batched=True, batch_size=batch_size, num_proc=max(1, n_jobs//2), keep_in_memory=True).with_format(None)
    neu_dataset_en = neu_dataset.with_format("np").filter(lambda xs: [x=='en' for x in xs['lang']], batched=True, batch_size=batch_size, num_proc=max(1, n_jobs//2), keep_in_memory=True).with_format(None)

    print('Creating Vietnamese datasets...')
    env_dataset_vi = env_dataset.with_format("np").filter(lambda xs: [x == 'vi' for x in xs['lang']], batched=True, batch_size=1024, num_proc=max(1, n_jobs//2)).with_format(None)
    gov_dataset_vi = gov_dataset.with_format("np").filter(lambda xs: [x == 'vi' for x in xs['lang']], batched=True, batch_size=1024, num_proc=max(1, n_jobs//2)).with_format(None)
    soc_dataset_vi = soc_dataset.with_format("np").filter(lambda xs: [x == 'vi' for x in xs['lang']], batched=True, batch_size=1024, num_proc=max(1, n_jobs//2)).with_format(None)
    neu_dataset_vi = neu_dataset.with_format("np").filter(lambda xs: [x == 'vi' for x in xs['lang']], batched=True, batch_size=1024, num_proc=max(1, n_jobs//2)).with_format(None)

    print('env_en:', len(env_dataset_en))
    print('gov_en:', len(gov_dataset_en))
    print('soc_en:', len(soc_dataset_en))
    print('neu_en:', len(neu_dataset_en))

    print('env_vi:', len(env_dataset_vi))
    print('gov_vi:', len(gov_dataset_vi))
    print('soc_vi:', len(soc_dataset_vi))
    print('neu_vi:', len(neu_dataset_vi))


    # Select only the columns we need
    env_processed_en = env_dataset_en.select_columns(["text", "labels", "class"])
    gov_processed_en = gov_dataset_en.select_columns(["text", "labels", "class"])
    soc_processed_en = soc_dataset_en.select_columns(["text", "labels", "class"])
    neu_processed_en = neu_dataset_en.select_columns(["text", "labels", "class"])

    env_processed_vi = env_dataset_vi.select_columns(["text", "labels", "class"])
    gov_processed_vi = gov_dataset_vi.select_columns(["text", "labels", "class"])
    soc_processed_vi = soc_dataset_vi.select_columns(["text", "labels", "class"])
    neu_processed_vi = neu_dataset_vi.select_columns(["text", "labels", "class"])

    # Combine the processed datasets
    print("Combining processed datasets...")
    combined_df = concatenate_datasets([env_processed_en, gov_processed_en, soc_processed_en, neu_processed_en])
    combined_df = combined_df.shuffle(seed=seed)

    combined_df_vi = concatenate_datasets([env_processed_vi, gov_processed_vi, soc_processed_vi, neu_processed_vi])
    combined_df_vi = combined_df_vi.shuffle(seed=seed)

    unique_classes = [0,1,2,3]
    # Convert the column to ClassLabel type
    combined_df = combined_df.cast_column("class", ClassLabel(names=unique_classes))
    combined_df_vi = combined_df_vi.cast_column("class", ClassLabel(names=unique_classes))

    print('Splitting')
    split_dataset = combined_df.train_test_split(test_size=test_size, stratify_by_column="class", seed=seed)
    dataset_train, dataset_test = split_dataset["train"], split_dataset["test"]

    split_dataset = combined_df_vi.train_test_split(test_size=test_size, stratify_by_column="class", seed=seed)
    dataset_vi_train, dataset_vi_test = split_dataset["train"], split_dataset["test"]


    combined_df = concatenate_datasets([dataset_train, dataset_vi_train])
    combined_df = combined_df.shuffle(seed=seed)

    print('Saving datasets...')
    combined_df.save_to_disk(f"{working_dir}/train")
    dataset_test.save_to_disk(f"{working_dir}/val_en")
    dataset_vi_test.save_to_disk(f"{working_dir}/val_vi")
    print('Data processing done.')
    
    return combined_df, dataset_test, dataset_vi_test

if __name__=='__main__':
    # Usage example
    train_dataset_raw, valid_dataset_raw, valid_dataset_vi_raw = esg_large_datasets(n_jobs=48, batch_size=1024*3, seed=seed, test_size=test_size, working_dir='./data')  # Use 48 cores for parallel processing

    print(train_dataset_raw[0])
    print(train_dataset_raw)
    print(valid_dataset_raw)
    print(valid_dataset_vi_raw)
    
     
