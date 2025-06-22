%%writefile train.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FSDP_VERSION"] = "2"
# os.environ["WANDB_DISABLED"] = "true"
from datasets import load_dataset, load_from_disk, concatenate_datasets, ClassLabel, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    EvalPrediction
)

from trl import SFTTrainer
from huggingface_hub import ModelCard, create_repo, upload_folder

# Metrics calculation
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix, average_precision_score, precision_recall_curve, recall_score, precision_score

import torch
import torch.nn as nn
import accelerate
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import re
from collections import defaultdict


torch.backends.cuda.matmul.allow_tf32 = True

accelerator = Accelerator()

def master_print(*args, **kwargs):
    if accelerator.is_main_process:
        print(*args, **kwargs)

# Function for one-hot encoding of labels
num_classes = 4
def one_hot_encode(label):
    encoded_label = np.zeros(num_classes)
    # print(label)
    encoded_label[label] = 1
    return encoded_label

# Initialize tokenizer
num_classes = 4
# model_name = "yiyanghkust/finbert-esg" # 109
model_name = "yiyanghkust/finbert-esg-9-categories" # 109
# model_name = "TankuVie/bert-base-multilingual-uncased-vietnamese_sentiment_analysis" # 167
model_name = "distilbert/distilbert-base-multilingual-cased" # 135
# model_name = "vinai/phobert-base" # 135
model_name = "FacebookAI/xlm-roberta-base" # 278
model_name = "FacebookAI/roberta-base" # 124
model_name = "microsoft/deberta-v3-base" # 184
# model_name = "microsoft/deberta-v3-small" # 141
# model_name = "microsoft/deberta-v3-large" # 435
# model_name = "google-bert/bert-base-multilingual-cased" # 167
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
master_print(len(tokenizer))

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    # device_maps='auto',
    problem_type="multi_label_classification",
    # problem_type="single_label_classification",
    num_labels=num_classes
)

class_mapping = {
            0: "Neural",
            1: "Environmental",
            2: "Social",
            3: "Governance"
        }

model.config.id2label = class_mapping
model.config.label2id = {v:k for k,v in class_mapping.items()}

num_classes = 4

master_print(model)
master_print(model._no_split_modules)

if model_name in ["vinai/phobert-base", "FacebookAI/xlm-roberta-base"] or 'roberta' in model.config.model_type:
    # model._no_split_modules = ['RobertaEmbeddings', 'RobertaSdpaSelfAttention']
    master_print('roberta model')
    master_print(model.classifier.out_proj.in_features)
    model.classifier.out_proj = nn.Linear(in_features=model.classifier.out_proj.in_features, out_features=num_classes, bias=True)
else:
    print('NOO')
    master_print(model.classifier.in_features)
    model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes, bias=True)

if 'deberta' in model.config.model_type:
    model._no_split_modules = ['DebertaV2Layer']
#     master_print(model._no_split_modules)
# model = BetterTransformer.transform(model, keep_original_model=True)

# Tokenization function
def tokenize_fn(examples):
    encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    # encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    encoding['labels'] = [one_hot_encode(x) for x in examples['class']]
    return encoding

def per_class_accuracy(y_true, y_pred):
    correct = defaultdict(int)
    total = defaultdict(int)
    acc = {0:-1, 1:-1, 2:-1, 3:-1}
    for yt, yp in zip(y_true, y_pred):
        total[yt] += 1
        if yt == yp:
            correct[yt] += 1

    for cls in total:
        acc[cls] = correct[cls] / total[cls]
    return acc

def multi_label_metrics(predictions, labels, threshold=0.5, class_mapping=None):

    # Class mapping
    if not class_mapping:
        class_mapping = {
            0: "Neural",
            1: "Environmental",
            2: "Social",
            3: "Governance"
        }

    probs_np = predictions

    # Initialize y_pred with zeros
    y_pred = np.zeros(probs_np.shape)

    # Set argmax index in each row to 1
    y_pred[np.arange(probs_np.shape[0]), np.argmax(probs_np, axis=1)] = 1
    # print(y_pred)
    # Finally, compute metrics
    y_true = labels
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=True)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    # print('true', y_true[:10])
    # print(y_pred[:10])
    # assert 1==0
    # Calculate ROC AUC - handling potential errors
    # roc_auc = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovo')

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=True)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=True)
    metrics = {
        'f1': f1_macro_average,
        'precision': precision,
        'recall': recall,
        # 'roc_auc': roc_auc,
        'accuracy': accuracy,
        # 'per_class_accuracy': per_class_accuracy
    }
    acc_per_class = per_class_accuracy(y_true, y_pred)
    for k in range(num_classes):
        class_name = class_mapping[k]
        metrics[f'{class_name}_acc'] = acc_per_class[k]
        overall_metrics[f'{class_name}_f1'] = f1_per_class[k]

    conf_matrix = confusion_matrix(y_true, y_pred)
    if accelerator.is_main_process:
        print()
        print(conf_matrix)
        labels = [class_mapping[i] for i in range(num_classes)]
        labels = [x[0] for x in list(class_mapping.values())[:num_classes]]
        plt.figure(figsize=(5, 3.5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=labels,
                    yticklabels=labels, cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    return metrics

def comprehensive_multi_label_metrics(predictions, labels, threshold=0.5, class_mapping=None, num_classes=4):
    # Class mapping
    if not class_mapping:
        class_mapping = {
            0: "Neural",
            1: "Environmental",
            2: "Social",
            3: "Governance"
        }

    probs_np = predictions

    # Initialize y_pred with zeros
    y_pred = np.zeros(probs_np.shape)

    # Set argmax index in each row to 1
    y_pred[np.arange(probs_np.shape[0]), np.argmax(probs_np, axis=1)] = 1

    # Convert to class indices
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    # print(k)

    # Overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        # 'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        # 'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # Per-class metrics using sklearn's built-in functions
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Custom accuracy per class
    acc_per_class = per_class_accuracy(y_true, y_pred)

    # Organize per-class metrics
    per_class_metrics = {}
    for k in range(min(num_classes, len(f1_per_class))):
        class_name = class_mapping[k]
        overall_metrics[f'{class_name}_acc'] = acc_per_class.get(k, 0.0)
        overall_metrics[f'{class_name}_f1'] = f1_per_class[k]
        # per_class_metrics[class_name] = {
        #     'accuracy': acc_per_class.get(k, 0.0),
        #     'f1': f1_per_class[k],
        # }
    if accelerator.is_main_process:
        print()
        conf_matrix = confusion_matrix(y_true, y_pred)
        print(conf_matrix)
        labels = [class_mapping[i] for i in range(num_classes)]
        labels = [x[0] for x in list(class_mapping.values())[:num_classes]]
        plt.figure(figsize=(5, 3.5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=labels,
                    yticklabels=labels, cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    return overall_metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    result = comprehensive_multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

from transformers.utils import logging
logging.set_verbosity_debug()

if accelerator.is_main_process or 1:
    train_dataset_raw = load_from_disk("./train", keep_in_memory=True)
    valid_dataset_raw = load_from_disk("./val", keep_in_memory=True)
    valid_dataset_vi_raw = load_from_disk("./val_vi", keep_in_memory=True)
    train_dataset = train_dataset_raw.map(tokenize_fn, batched=True, batch_size=1024*3, num_proc=44, keep_in_memory=True)
    valid_dataset = valid_dataset_raw.map(tokenize_fn, batched=True, batch_size=1024*3, num_proc=24, keep_in_memory=True)
    valid_dataset_vi = valid_dataset_vi_raw.map(tokenize_fn, batched=True, batch_size=1024*3, num_proc=24, keep_in_memory=True)
# Training hyperparameters
batch_size = 48
metric_name = "vi_f1_macro"
# num_train_epochs = 3

max_steps = 2000
def main():

    args = TrainingArguments(
        f"./{model_name}-esg-classification",
        eval_strategy="steps",
        save_strategy="epoch",
        save_steps=2000,
        logging_steps=1,
        eval_steps=80,
        # tp_size=4,
        learning_rate=2e-5,
        lr_scheduler_type='cosine',
        # lr_scheduler_type = "cosine_with_restarts",
        # lr_scheduler_kwargs = { "num_cycles": 5 },
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1,
        optim="adamw_torch",  # Optimizer,
        group_by_length=True,
        gradient_accumulation_steps=1,
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs = {"use_reentrant": True},
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # num_train_epochs=5,
        max_steps=max_steps,
        data_seed=42,
        dataloader_num_workers=8,
        dataloader_drop_last=True,
        load_best_model_at_end=False,
        metric_for_best_model=metric_name,
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        # fp16=True,
        bf16=True,
        # tf32=True,
        bf16_full_eval=True,
        torch_compile_backend='inductor',
        torch_compile=True,
        # fsdp="full_shard auto_wrap",
        # fsdp_config={"backward_prefetch": "backward_pre",
        #              "forward_prefetch": True,
        #              "sync_module_states": True,
        #             "cpu_ram_efficient_loading": True,
        #              # "transformer_layer_cls_to_wrap": ["RobertaSdpaSelfAttention"] if model_name == "vinai/phobert-base" else None,
        #              # "xla": True,
        #              # "xla_fsdp_v2": True,
        #             },
        # auto_find_batch_size=True,
        use_liger_kernel=True,
        torch_empty_cache_steps=50,
        run_name=f'{model_name}',
        report_to='all',
        # push_to_hub=True,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset={'en':valid_dataset,  'vi': valid_dataset_vi},
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,

        # data_collator=data_collator,
    )
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()
    master_print("Evaluation results:", eval_result)

    log = pd.DataFrame(trainer.state.log_history)
    log.to_csv('log.csv')
    log.to_csv(f'log_{model_name.split("/")[-1]}.csv')
    log

if __name__ == "__main__":
    main()

# repo_name = 'test'
accelerator.wait_for_everyone()