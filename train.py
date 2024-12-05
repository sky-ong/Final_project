import argparse
import logging
import numpy as np
import torch
from experiment import STPP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser('Neural flows')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--experiment', type=str, default='stpp', help='Which experiment to run', choices=['stpp'])
parser.add_argument('--model', type=str, help='Whether to use ODE or flow based model', choices=['ode', 'flow'])
parser.add_argument('--data',  type=str, help='Dataset name', choices=['pinwheel', 'earthquake', 'covid', 'bike'])

# Training loop args
parser.add_argument('--epochs', type=int, default=1000, help='Max training epochs')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay (regularization)')
parser.add_argument('--batch-size', type=int, default=50, help='Batch size')
parser.add_argument('--clip', type=float, default=1, help='Gradient clipping')

# STPP-specific args
parser.add_argument('--density-model', type=str, help='Type of density model', choices=['independent', 'attention', 'jump'])
parser.add_argument('--hidden-layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--hidden-dim', type=int, default=1, help='Size of hidden layer')
parser.add_argument('--activation', type=str, default='Tanh', help='Hidden layer activation')
parser.add_argument('--final-activation', type=str, default='Identity', help='Last layer activation')

args = parser.parse_args()

def get_experiment(args, logger):
    if args.experiment == 'stpp':
        return STPP(args, logger)
    else:
        raise ValueError(f'Experiment {args.experiment} not supported')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment = get_experiment(args, logger)
    experiment.train()
    experiment.finish()

