######## 1. download the pre-trained vit-b and move to ./checkpoint ########
######## 2. train with vanilla dropout ########
python train.py --mode vanilla --dataset cifar10
######## 3. train with moredrop ########
python train.py --mode moredrop --dataset cifar10 --p 0.1 --alpha 1
