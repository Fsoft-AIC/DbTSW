import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_iter',
        type=int,
        default=2500,
        help='Mumber of training epochs')
    parser.add_argument(
        '--L',
        type=int,
        default=1000,
        help='Total number of lines')
    parser.add_argument(
        '--n_lines',
        type=int,
        default=4,
        help='TWD: Number of lines in each tree')
    parser.add_argument(
        '--lr_sw',
        type=float,
        default=1e-3,
        help='learning rate of SW methods')
    parser.add_argument(
        '--lr_tsw_sl',
        type=float,
        default=1e-3,
        help='TWD: learning rate of TWD methods')
    parser.add_argument(
        '--delta',
        type=float,
        default=50.,
        help='TWD: delta to tune distance-based')
    parser.add_argument(
        '--p',
        type=int,
        default=1,
        help='p')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='25gaussians',
        help='Name of the dataset, could be []')   
    parser.add_argument(
        '--std',
        type=float,
        default=0.1,
        help='TWD: std to generate root of trees')
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=0.1,
        help='TWD: std to generate root of trees')
    parser.add_argument(
        '--plot',
        action='store_true',
        default=False,
        help='turn on to plot diagram')


    opt = parser.parse_args()
    return opt
