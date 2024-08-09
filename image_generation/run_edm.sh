######## 1. data perparation ########
# CIFAR-10: Download the CIFAR-10 python version and convert to ZIP archive:
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10-32x32.zip

######## 2.EDM training (baseline) ########
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py\
--outdir=training-runs  --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --moredrop False

######## 3.EDM training with MoReDrop (ours) ########
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py\
--outdir=training-runs-moredrop  --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --moredrop True
######## 4.Sample and Evalutaion ########
torchrun --standalone --nproc_per_node=4 generate.py --outdir=out --steps=18

torchrun --standalone --nproc_per_node=4 fid.py calc --images=out\
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
