# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple
from tqdm import tqdm
import math
import torch
import torch.nn as nn

# import checkpoint


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

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, set_rate, p, a, data_parallel=True):
        """ Train Loop """
        self.model.train() # train mode
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)
        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}    

                self.optimizer.zero_grad()
                if p == 0:
                    loss = model(**batch).loss.mean() # mean() for Data Parallelism
                    # loss = outputs.loss
                    # logits = outputs.logits  
                else:
                    set_rate(model,p)
                    loss1 = model(**batch).loss
                    set_rate(model,0)
                    loss2 = model(**batch).loss

                    l = (loss1-loss2).mean()
                    gap = a*l/2
                    loss = loss2.mean() + torch.tanh(gap)
                # loss = get_loss(model, batch, global_step).mean() # mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())

                # if global_step % self.cfg.save_steps == 0: # save
                #     self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    # self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        # self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        # self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)
        f1 = []
        results = [] # prediction results
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}    
            with torch.no_grad(): # evaluation without gradient calculation
                result, accuracy = evaluate(model, batch) # accuracy to print
            results.append(result)
            f1.append(accuracy)
            iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        return results,f1

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

