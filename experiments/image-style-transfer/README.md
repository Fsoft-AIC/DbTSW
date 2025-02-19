# Image Color Transfer
This is a source code for Color Transfer task in the paper.

## Requirements
We need python 3.7 or above and these following packages:
```
torch>=1.13.0
matplotlib
scikit-image
scikit-learn
tqdm
pot
wandb
```

## Usage
The source code is in the `main.py` file with these following command line arguments:
```
usage: main.py [-h] [--L N] [--delta DELTA] [--std STD] [--n_lines_tw N_LINES_TW] [--lr_tw LR_TW] [--num_iter N] [--num_iter_tw N] [--source N] [--target N]
               [--cluster] [--load] [--palette]

optional arguments:
  -h, --help            show this help message and exit
  --L N                 input batch size for training (default: 100)
  --delta DELTA
  --std STD
  --n_lines_tw N_LINES_TW
  --lr_tw LR_TW
  --num_iter N          Num Interations
  --num_iter_tw N       Num Interations of TW
  --source N            Source image path
  --target N            Target image path
  --cluster             Use clustering
  --load                Load precomputed
  --palette             Show color palette
```

This code will output the color transfer result from the source image to the target image for various methods and save it to the result folder named `final`.

Sample command:
```bash
python main.py --cluster --std=1 --delta=1 --lr_tw=11 --n_lines_tw=3 --num_iter=2000 --num_iter_tw=2000 --source=images/imageB.jpg --target=images/img1.jpg
```
