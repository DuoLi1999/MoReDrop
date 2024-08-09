# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """
import os
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
import tokenization
import models
import optim
import train
import json
from datasets import load_dataset,load_metric,load_from_disk
from utils import set_seeds, get_device, truncate_tokens_pair,compute_metrics,get_metrics
# def load_glue(train_data,pipeline ,sentence1_key, sentence2_key ):
#     if sentence2_key==None:
#         data=[]
#         for line in train_data:
#             instance = tuple([str(line['label']),line[sentence1_key]])
#             for proc in pipeline: # a bunch of pre-processing
#                 instance = proc(instance)
#             data.append(instance) 
#     else:
#         data=[]
#         for line in train_data:
#             instance = tuple([str(line['label']),line[sentence1_key],line[sentence2_key]])
#             for proc in pipeline: # a bunch of pre-processing
#                 instance = proc(instance)
#             data.append(instance)
#     tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
#     return tensors
def load_glue(train_data, pipeline, sentence1_key, sentence2_key):
    data = []
    for line in train_data:
        if sentence2_key is None:
            instance = tuple([str(line['label']), line[sentence1_key],None])
        else:
            instance = tuple([str(line['label']), line[sentence1_key], line[sentence2_key]])
        for proc in pipeline:
            instance = proc(instance)
        data.append(instance)
    
    tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
    return tensors
class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, file, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r") as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance in self.get_instances(lines): # instance : tuple of fields
                for proc in pipeline: # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class gluedata():
    def __init__(self, data):
        self.tensors =data
    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) \
                   if text_b else []

        return (label, tokens_a, tokens_b)
from typing import NamedTuple
    
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


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)
        return (input_ids, segment_ids, input_mask, label_id)

class TokenIndexing_stsb(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = float(label)

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)
        return (input_ids, segment_ids, input_mask, label_id)

class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels,task=None):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.task = task
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)
        if task == "stsb":
            self.loss = torch.nn.MSELoss()
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.loss = torch.nn.CrossEntropyLoss()
    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        # only use the first h in the sequence
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        if self.task == "stsb":
            logits = torch.clip((self.sigmoid(logits) * 5.5), min=0.0, max=5.0)
        return logits



def main(idx=3,
         train_cfg='./config/train_mrpc.json',
         model_cfg='config/bert_base.json',
         data_file='glue_data/MRPC/train.tsv',
         model_file=None,
         pretrain_file='ckp/bert_model.ckpt',
         data_parallel=True,
         vocab='ckp/vocab.txt',
         save_dir='../exp/bert/mrpc',
         max_len=128,
         mode='train',
         p=0.1,
         seed=0,
         a=1,
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
    task = tasks[idx]
    print(task)
    sentence1_key, sentence2_key = task_to_keys[task]
    with open(train_cfg, "r") as file:
        cfg_tem = json.load(file)[idx]
    # cfg_tem['n_epochs'] = 1
    cfg_tem['total_steps']  /= 3
    cfg_tem['n_epochs'] = t
    cfg_tem['total_steps']  = int(cfg_tem['total_steps']) * t
    cfg = Config(**cfg_tem)
    # cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)
    set_seeds(seed)
    # data = load_dataset('glue', task)
    dir_data = '/home/hoo/data/glue/'+task
    data = load_from_disk(dir_data)
    train_data = data['train']
    labels = [str(label) for label in set(train_data['label'])]

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    if task == 'stsb':
        pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(max_len),
                TokenIndexing_stsb(tokenizer.convert_tokens_to_ids,
                              labels, max_len)]
    else:
        pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                    AddSpecialTokensWithTruncation(max_len),
                    TokenIndexing(tokenizer.convert_tokens_to_ids,
                                labels, max_len)]
    if task == 'mnli':
        val_data_m = gluedata(load_glue(data['validation_matched'], pipeline,sentence1_key,sentence2_key))
        val_data_mm = gluedata(load_glue(data['validation_mismatched'], pipeline,sentence1_key,sentence2_key))

    else:
        val_data = gluedata(load_glue(data['validation'], pipeline,sentence1_key,sentence2_key))


    # dataset1 = load_glue(train_data, pipeline)
    d = gluedata(load_glue(train_data, pipeline, sentence1_key, sentence2_key ))
    if sentence2_key is None:
        print(f"Sentence: {data['train'][0][sentence1_key]}")
    else:
        print(f"Sentence 1: {data['train'][0][sentence1_key]}")
        print(f"Sentence 2: {data['train'][0][sentence2_key]}")

    

    # dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(d, batch_size=cfg.batch_size, shuffle=True)
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    model = Classifier(model_cfg, num_labels,task)
    criterion = model.loss

    trainer = train.Trainer(cfg,
                            model,
                            data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    if mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            if task == 'stsb':
                label_id=label_id.float()
                loss = criterion(logits.squeeze() , label_id)
                return loss, logits
            else:
                loss = criterion(logits , label_id)
                return loss, logits
        def set_rate(model,p):
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = p
        trainer.train(get_loss, set_rate,p,a,model_file, pretrain_file, data_parallel)

        # trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    # elif mode == 'eval':
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            # metric,metric1=get_metrics(task)
            # score= compute_metrics(predictions=logits, references=label_id,metric=metric1)
            _, label_pred = logits.max(1)
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
        file_id = task + '_'+str(seed)+ '_' +str(p) + '_' + str(a)+ '_' + str(t)
        model_file_save = f"save/{file_id}_{current_timestamp}"
        # val_data = gluedata(load_glue(data['validation'], pipeline,sentence1_key,sentence2_key))
        if task == 'mnli':
            data_iter_val_m = DataLoader(val_data_m, batch_size=cfg.batch_size, shuffle=True)
            trainer.data_iter=data_iter_val_m
            results_m,_ = trainer.eval(evaluate, model_file, data_parallel)
            total_accuracy_m = torch.stack(results_m)[:].mean().item()
            data_iter_val_mm = DataLoader(val_data_mm, batch_size=cfg.batch_size, shuffle=True)
            trainer.data_iter=data_iter_val_mm
            results_mm,_ = trainer.eval(evaluate, model_file, data_parallel)  
            total_accuracy_mm = torch.stack(results_mm)[:].mean().item()
            print('m:', total_accuracy_m,'mm:',total_accuracy_mm,seed,p,a)
            checkpoint = {
                        'accuracy_m': total_accuracy_m,
                        'accuracy_mm': total_accuracy_mm
                        
                    }
        elif task == 'qqp' or task == 'mrpc':
            data_iter_val = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)
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
            data_iter_val = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)
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
