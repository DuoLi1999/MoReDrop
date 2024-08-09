# MoReDrop: Dropout without Dropping
This repo contains the code of [MoReDrop: Dropout without Dropping]()

##  Usage:


```python
import torch.nn.functional as F

# define your task model, which outputs the classifier logits
model = TaskModel()
p = 0.5
alpha = 1

def set_rate(model, p):
    # set dropout rate
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p

def compute_loss(logits, logits, alpha):
    dense_loss = cross_entropy_loss(logits, label)
    sub_loss = cross_entropy_loss(logits2, label)
    loss_gap = alpha * (sub_loss - dense_loss) / 2
    loss = dense_loss + torch.tanh(loss_gap)
    return loss

# forward twice

# set p = 0, deactivate dropout, get the logits of dense model
set_rate(0) 
logits = model(x)

# set p = 0, activate dropout, get the logits of sub model
set_rate(p) 
logits2 = model(x)

loss = compute_loss(logits, logits, alpha)

```





