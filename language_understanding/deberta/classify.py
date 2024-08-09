# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """
import os
import transformers
import itertools
import csv
import fire
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
import optim
import train
import json
from datasets import load_dataset,load_metric,load_from_disk
from utils import set_seeds, get_device, truncate_tokens_pair,compute_metrics,get_metrics
from transformers import DebertaV2ForSequenceClassification,DebertaV2Tokenizer,DebertaV2Config

from typing import NamedTuple
# from transformers import cached_path

# cached_path('https://huggingface.co')

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



def main(idx=7,
         train_cfg='/home/hoo/Projects/MoReDrop/MoReDrop/language_understanding/deberta/config/train.json',
         model_file=None,
        #  pretrain_file='uncased_L-12_H-768_A-12/bert_model.ckpt',
         data_parallel=True,
        #  vocab='roberta.base/dict.txt',
         save_dir='save_rd',
         max_len=128,
         mode='train',
         p=0.1,
         seed=0,
         a=0,
         t=1,
         ):
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
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding="max_length", max_length=max_len)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding="max_length", max_length=max_len, return_tensors='pt')

    transformers.logging.set_verbosity_error()
    task = tasks[idx]
    print(task)
    sentence1_key, sentence2_key = task_to_keys[task]
    with open(train_cfg, "r") as file:
        cfg_tem = json.load(file)[idx]

    cfg_tem['total_steps']  /= 3
    cfg_tem['n_epochs'] = t
    cfg_tem['total_steps']  = int(cfg_tem['total_steps']) * t
    cfg = Config(**cfg_tem)
    # cfg = train.Config.from_json(train_cfg)
    # model_cfg = models.Config.from_json(model_cfg)
    set_seeds(seed)

    # dataset = load_dataset('glue', task)
    dir_data = '/home/hoo/data/glue/'+task

    dataset = load_from_disk(dir_data)

    # dataset.save_to_disk(task)
    # model_path = './model/'

    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-xsmall',cache_dir='ckp')
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-xsmall',cache_dir='ckp', num_labels=num_labels).to('cuda:0')
    encoded_dataset = dataset.map(preprocess_function, batched=True)


    train_dataset = format_dataset(encoded_dataset["train"])
    if task == 'mnli':
        valid_dataset_m = format_dataset(encoded_dataset["validation_matched"])
        valid_dataset_mm = format_dataset(encoded_dataset["validation_mismatched"])

    else:
        valid_dataset = format_dataset(encoded_dataset["validation"])

    # dataset1 = load_glue(train_data, pipeline)
    if sentence2_key is None:
        print(f"Sentence: {dataset['train'][0][sentence1_key]}")
    else:
        print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
        print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

    

    # dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    trainer = train.Trainer(cfg,
                            model,
                            data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    if mode == 'train':
        def set_rate(model,p):
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = p
        trainer.train(set_rate, p, a, data_parallel)

        # trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    # elif mode == 'eval':
        def evaluate(model, batch):
            label_id = batch['labels']
            logits = model(**batch).logits
            # metric,metric1=get_metrics(task)
            # score= compute_metrics(predictions=logits, references=label_id,metric=metric1)
            if task != 'stsb':
                _, label_pred = logits.max(1)
            # label_pred = 1
            if task == 'cola':
                mcc = matthews_corrcoef(label_id.cpu(), label_pred.cpu())
                mcc = torch.tensor(mcc)
                return mcc,mcc 
            elif task =='stsb':
                ac = pearsonr(logits.squeeze().cpu().numpy(),label_id.cpu().numpy())[0]
                ac = torch.tensor(ac)
                return ac,ac

            elif task == 'qqp' or task == 'mrpc':
                result = (label_pred == label_id).float() #.cpu().numpy()
                accuracy = result.mean()
                f1 = f1_score(label_id.cpu().numpy(),label_pred.cpu().numpy())
                f1 = torch.tensor(f1)
                return accuracy,f1
            else:
                result = (label_pred == label_id).float() #.cpu().numpy()
                accuracy = result.mean()
                return accuracy,accuracy

        #     return metric
            # mcc = matthews_corrcoef(label_id.cpu(), label_pred.cpu())

            # return mcc,mcc
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_id = task + '_'+str(seed)+ '_' +str(p) + '_' + str(a) + '_' + str(t)
        model_file_save = f"save/{file_id}_{current_timestamp}"
        # val_data = gluedata(load_glue(data['validation'], pipeline,sentence1_key,sentence2_key))
        if task == 'mnli':
            data_iter_val_m = DataLoader(valid_dataset_m, batch_size=cfg.batch_size, shuffle=True)
            trainer.data_iter=data_iter_val_m
            results_m,_ = trainer.eval(evaluate, model_file, data_parallel)
            total_accuracy_m = torch.stack(results_m)[:].mean().item()
            data_iter_val_mm = DataLoader(valid_dataset_mm, batch_size=cfg.batch_size, shuffle=True)
            trainer.data_iter=data_iter_val_mm
            results_mm,_ = trainer.eval(evaluate, model_file, data_parallel)  
            total_accuracy_mm = torch.stack(results_mm)[:].mean().item()
            print('m:', total_accuracy_m,'mm:',total_accuracy_mm,seed,p,a)
            checkpoint = {
                        'accuracy_m': total_accuracy_m,
                        'accuracy_mm': total_accuracy_mm
                        
                    }
        elif task == 'qqp' or task == 'mrpc':
            data_iter_val = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)
            trainer.data_iter=data_iter_val
            results,f_1 = trainer.eval(evaluate, model_file, data_parallel)
    
            total_accuracy = torch.stack(results)[:].mean().item()
            total_f1 =  torch.stack(f_1)[:].mean().item()
            print(task,'Accu:' ,total_accuracy, 'F-1:', total_f1 , seed,p,a)
            checkpoint = {
                        'accuracy': total_accuracy,
                        'f1':total_f1
                    }            
        else:
            data_iter_val = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)
            trainer.data_iter=data_iter_val
            results,_ = trainer.eval(evaluate, model_file, data_parallel)
    
            total_accuracy = torch.stack(results)[:].mean().item()
        
            print('Accuracy of '+ task, total_accuracy,seed,p,a)
            checkpoint = {
                        'accuracy': total_accuracy
                    }
        os.makedirs(model_file_save)
        path = os.path.join(model_file_save, "checkpoint.pth")
        torch.save(checkpoint, path)

if __name__ == '__main__':
    fire.Fire(main)
