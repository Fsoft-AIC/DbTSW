# Source code for Diffusion Experiments
Based on the code from [RPSW](https://github.com/khainb/RPSW).

## Installation
Python 3.9.12 is used for the experiments. The code is tested on Ubuntu 20.04.1 LTS.

Install the required packages using the following command:
```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install `power_spherical` package:
```bash
cd power_spherical
pip install .
```

No need to setup data for CIFAR-10, as the code will download the dataset automatically.

## CIFAR-10 Training
**Note**: the code only tested on a **single GPU**. Some modifications may be needed for multi-GPU training.

For DDGAN
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 \
    --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
    --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 \
    --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 \
    --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss gan \
    --ch_mult 1 2 2 2 --save_content \
    --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For SW
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss sw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```
For EBSW
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss maxsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For RPSW
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss rpsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For DSW:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss dsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For EBSW
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss ebsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For IWRPSW:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss ebrpsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For TSW-SL:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_delta 0 --twd_gen_mode gaussian_raw --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For Db-TWD:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_delta 10 --twd_gen_mode gaussian_raw --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For Db-TWD-perp:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_delta 10 --twd_gen_mode gaussian_orthogonal --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```


#### CIFAR-10 Testing ####
For testing the trained model, use the name of the experiment in the `--exp` argument. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python3 test_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --max_epoch_id 1800 --compute_fid \
--wandb_project_name "twd" --wandb_entity "user-id"
```
