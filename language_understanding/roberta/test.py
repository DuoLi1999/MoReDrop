from transformers import RobertaForSequenceClassification,RobertaTokenizer,RobertaConfig
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
import json
import optim
from typing import NamedTuple
import train
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import numpy as np
from scipy.stats import pearsonr
import os
import itertools
import csv
import fire
from datetime import datetime
import torch
import torch.nn as nn
from utils import set_seeds, get_device
from tqdm import tqdm


tasks = ["cola", "mnli", "mrpc", "qnli", "qqp","rte","sst2","stsb"]
task_to_keys = {
    "cola": ("sentence", None),#mcc-8,551
    "mnli": ("premise", "hypothesis"),#392,702
    "mrpc": ("sentence1", "sentence2"),#3,668
    "qnli": ("question", "sentence"),#104,743
    "qqp": ("question1", "question2"),#363,870
    "rte": ("sentence1", "sentence2"),#2,491
    "sst2": ("sentence", None),#67350
    "stsb": ("sentence1", "sentence2"),#Corr-5,749
}
task = tasks[7]
print(task)
sentence1_key, sentence2_key = task_to_keys[task]
dataset = load_dataset("glue", task)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# def preprocess_function(examples):
#     if sentence2_key is None:
#         return tokenizer(examples[sentence1_key], truncation=True)
#     return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,return_tensors='pt')

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding="max_length", max_length=128)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding="max_length", max_length=128, return_tensors='pt')
def format_dataset(dataset):
    # format the dataset to be used with DataLoader
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    columns = ['input_ids', 'attention_mask', 'labels']
    if 'token_type_ids' in dataset.features:
        columns.append('token_type_ids')
    dataset.set_format(type='torch', columns=columns)
    return dataset
    
class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels).to('cuda:0')
encoded_dataset = dataset.map(preprocess_function, batched=True)

train_dataset = format_dataset(encoded_dataset["train"])
valid_dataset = format_dataset(encoded_dataset["validation"])
test_dataset = format_dataset(encoded_dataset["test"])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
valid_dataloader = DataLoader(valid_dataset, batch_size=32)

with open('config/train_mrpc.json', "r") as file:
    cfg_tem = json.load(file)[0]
cfg = Config(**cfg_tem)
# trainer = train.Trainer(cfg,
#                         model,
#                         train_dataloader,
#                         optim.optim4GPU(cfg, model),
#                         'save', get_device())
iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')

for i,batch in enumerate(iter_bar):
    batch = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}    
    outputs = model(**batch)
    loss = outputs.loss
    logits = outputs.logits    
    print(batch)
print(1)