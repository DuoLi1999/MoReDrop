# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta, datetime

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils import clip_grad

from torch.optim.lr_scheduler import CyclicLR, OneCycleLR

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

import csv


class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):

    num_classes = 10 if args.dataset == "cifar10" else 100
    if args.dataset == "imagenet":
        num_classes=1000

    if args.mode == "moredrop":
        from models.vit_moredrop import VisionTransformer, CONFIGS
        config = CONFIGS[args.model_type]
        config.transformer.dropout_rate=args.p
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,alpha=0.5)

    else:
        from models.vit import VisionTransformer, CONFIGS
        config = CONFIGS[args.model_type]
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)

    pre_trained_dir = 'checkpoint/'+args.model_type+'.npz'
    model.load_from(np.load(pre_trained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    
    # output the current model parameters' number per million
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    # all_preds, all_label = [], []
    num_samples = len(test_loader.dataset)
    all_preds = torch.zeros(num_samples, dtype=torch.long)
    all_labels = torch.zeros(num_samples, dtype=torch.long)

    epoch_iterator = tqdm(test_loader)
    loss_fct = torch.nn.CrossEntropyLoss()

    processed_samples = 0
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)

        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        # Another implementation of 
        batch_size = x.size(0)
        # processed_samples = step*batch_size
        all_preds[processed_samples:processed_samples+batch_size] = preds.detach().cpu()
        all_labels[processed_samples:processed_samples+batch_size] = y.detach().cpu()
        processed_samples += batch_size

        
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    # all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_labels)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy, eval_losses.avg



def train_per_epoch(args, model, train_loader, step, losses, optimizer, scheduler, scaler):
    
    num_samples = len(train_loader.dataset)

    batch_size = train_loader.batch_size
    all_preds = torch.zeros(num_samples, dtype=torch.long)
    all_labels = torch.zeros(num_samples, dtype=torch.long)

    model.train()
    epoch_iterator = tqdm(train_loader)

 

    for i, batch in enumerate(epoch_iterator):
        epoch_iterator.set_description(f"Epoch {step+1} ")

        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with autocast():
            loss, pred = model(x, y, step)

        probs = torch.argmax(pred, dim=-1)
        train_acc = simple_accuracy(probs.view(-1), y.view(-1))

        epoch_iterator.set_postfix(acc='%.4f'%train_acc, loss='%.3f' %loss)


        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # loss.backward()
        scaler.scale(loss).backward()

        # print(grad_dict.values())

        if (i+1) % args.gradient_accumulation_steps == 0:
            losses.update(loss.item()*args.gradient_accumulation_steps)
            scaler.unscale_(optimizer)
            clip_grad.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            # optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        ind_sample = i*batch_size
        samples_size = x.size(0)

        # if x.size(0) != batch_size:
        #     print('test')

        all_preds[ind_sample:ind_sample+samples_size] = probs.detach().cpu()
        all_labels[ind_sample:ind_sample+samples_size] = y.detach().cpu()
    # accuracy = 0
    accuracy = simple_accuracy(all_preds, all_labels)



    return accuracy, losses.avg

    # return losses.avg


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        # writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    ## Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    

    # t_total = args.num_steps
    t_total = args.num_epochs*len(train_loader) # type: ignore

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    file_id = args.dataset + '_' + args.model_type + '_' + str(args.seed) + '_' +str(args.p) + '_' + str(args.a) 
    model_file = f"{args.mode}/ckp/{file_id}_{current_timestamp}"
    log_file = args.mode+'/logs/' + file_id +'_'+current_timestamp+'.csv'
    os.makedirs(model_file)
    if not os.path.exists(args.mode+'/logs/'):
        os.makedirs(args.mode+'/logs/')

    csv_logger = CSVLogger(args=args, 
                           fieldnames=['epoch', 'train_acc', 'test_acc','train_loss','test_loss'], 
                           filename=log_file)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)) # type: ignore
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0

    # register_hooks(model)

    scaler = GradScaler()
    for epoch in range(args.num_epochs):

        train_accu, train_loss = train_per_epoch(args, model, train_loader, epoch, losses, optimizer, scheduler, scaler)
        losses.reset()
        
        test_accu, test_loss = valid(args, model, test_loader, global_step)  # type: ignore
        tqdm.write(f"accuracy: {test_accu}")

        if best_acc < test_accu:
            best_acc = test_accu
            checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'accuracy': best_acc
                    }
            path = os.path.join(model_file, "checkpoint.pth")
            torch.save(checkpoint, path)

        row = {'epoch': str(epoch), 'train_acc': str(train_accu), 'test_acc': str(test_accu),'train_loss':str(train_loss),'test_loss':str(test_loss)}
        csv_logger.writerow(row)


    # if args.local_rank in [-1, 0]:
    #     writer.close()  # type: ignore
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("Best Accuracy:",best_acc,args.p,args.a)


def main():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default='first_trail',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--mode",  choices=["vanilla","moredrop"], default='moredrop',
                        help="MoReDrop or not")

    parser.add_argument("--dataset", choices=["cifar10", "cifar100","imagenet"], default="cifar100",
                        help="Which downstream task.")

    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")

    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument('--device',default='cuda:0')
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=1e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=200000, type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--num_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    parser.add_argument("--p", default=0.1, type=float,
                        help="alpha for kl loss")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="changeble alpha ")
    

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',    # type: ignore
                                             timeout=timedelta(minutes=60))  
        args.n_gpu = 1

    # If level is `INFO`, then the logs output on terminal
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    train_loader, test_loader = get_loader(args)
    # Training
    train(args, model)


if __name__ == "__main__":
    main()
