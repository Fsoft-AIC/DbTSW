# Gradient Flow
This is a source code for Gradient Flow task in the paper.

## Requirements
We need python 3.7 or above and these following packages:
```
torch>=1.13.0
matplotlib
scikit-image
scikit-learn
tqdm
numpy
wandb
```

## Usage

The source code is in the `GradientFlow.py` file with these following command line arguments:
```bash
usage: GradientFlow.py [-h] [--num_iter NUM_ITER] [--L L] [--n_lines N_LINES] [--lr_sw LR_SW] [--lr_tsw_sl LR_TSW_SL] [--delta DELTA] [--p P]
                       [--dataset_name DATASET_NAME] [--std STD] [--num_seeds NUM_SEEDS]

optional arguments:
  -h, --help            show this help message and exit
  --num_iter NUM_ITER   number of epochs of training
  --L L                 L
  --n_lines N_LINES     Number of lines in each tree
  --lr_sw LR_SW         learning rate of SW
  --lr_tsw_sl LR_TSW_SL
                        learning rate of TSW-SL
  --delta DELTA         delta to tune distance-based
  --p P                 p
  --dataset_name DATASET_NAME
                        Name of the dataset
  --std STD             std to generate root of trees
  --num_seeds NUM_SEEDS
                        std to generate root of trees
```

Example command:
```bash
python GradientFlow.py --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 50.0 --delta 1.0 --p 2 --dataset_name "gaussian_20d_small_v" --std 0.001 --num_seeds 10
```
