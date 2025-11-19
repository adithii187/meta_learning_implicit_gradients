
import sys
import os
import numpy as np
import torch
import random
import pickle
import argparse
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- FIX THE IMPORT PATH ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from implicit_maml.dataset import OmniglotTask, OmniglotFewShotDataset
from implicit_maml.learner_model import Learner, make_conv_network
from implicit_maml.utils import DataLog, smooth_vector, measure_accuracy

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)
logger = DataLog()

parser = argparse.ArgumentParser(description='Bilevel Solvers for MAML on Omniglot')
parser.add_argument('--data_dir', type=str, required=True, help='Location of the Omniglot dataset')
parser.add_argument('--N_way', type=int, default=5)
parser.add_argument('--K_shot', type=int, default=1)
parser.add_argument('--inner_lr', type=float, default=1e-2)
parser.add_argument('--outer_lr', type=float, default=1e-3)
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--meta_steps', type=int, default=2000)
parser.add_argument('--task_mb_size', type=int, default=16)
parser.add_argument('--use_gpu', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--num_tasks', type=int, default=20000)
parser.add_argument('--save_dir', type=str, default='results/default')
parser.add_argument('--method', type=str, default='fomaml',
                    choices=['maml', 'fomaml', 'reptile_new', 'neumann', 'penalty']) # <-- Add 'penalty'
parser.add_argument('--neumann_steps', type=int, default=3)
parser.add_argument('--penalty_lambda', type=float, default=1.0, 
                    help='Weight for the penalty term in the penalty method')
args = parser.parse_args()
logger.log_exp_args(args)

print("Generating tasks...")
train_val_permutation = list(range(1623))
random.shuffle(train_val_permutation)
task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, num_cls=args.N_way, num_inst=args.K_shot) for _ in tqdm(range(args.num_tasks))]
dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)

device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

meta_learner = Learner(model=make_conv_network(in_channels=1, out_dim=args.N_way).to(device),
                       loss_function=torch.nn.CrossEntropyLoss(),
                       inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)
fast_learner = Learner(model=make_conv_network(in_channels=1, out_dim=args.N_way).to(device),
                       loss_function=torch.nn.CrossEntropyLoss(),
                       inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)

meta_learner.inner_steps = args.n_steps
fast_learner.inner_steps = args.n_steps
pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

print(f"Training model with bilevel solver: {args.method.upper()}")
losses = np.zeros((args.meta_steps, 4))
accuracy = np.zeros((args.meta_steps, 2))

for outstep in tqdm(range(args.meta_steps)):
    w_k = meta_learner.get_params()
    meta_grad = torch.zeros_like(w_k)
    task_batch_indices = np.random.choice(len(dataset), size=args.task_mb_size, replace=False)

    for idx in task_batch_indices:
        fast_learner.set_params(w_k.clone())
        task = dataset[idx]
        vl_before = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
        tl = fast_learner.learn_task(task, num_steps=args.n_steps, w_0=w_k)
        
        if args.method == 'maml':
            task_outer_grad = meta_learner.compute_maml_grad(task)
        elif args.method == 'fomaml':
            task_outer_grad = fast_learner.compute_foml_grad(task)
        elif args.method == 'reptile_new':
            adapted_params = fast_learner.get_params()
            task_outer_grad = w_k - adapted_params
        elif args.method == 'neumann':
            vloss = fast_learner.get_loss(task['x_val'], task['y_val'])
            grad_val = torch.autograd.grad(vloss, fast_learner.model.parameters())
            flat_grad_val = torch.cat([g.contiguous().view(-1) for g in grad_val])
            v = flat_grad_val.clone()
            g_approx = flat_grad_val.clone()
        
            for _ in range(args.neumann_steps):
                hvp = fast_learner.hessian_vector_product(task, v)
                v = -args.inner_lr * hvp
                g_approx += v
            task_outer_grad = g_approx
        else:
            raise ValueError(f"Unknown method: {args.method}")

        meta_grad += (task_outer_grad / args.task_mb_size)
        vl_after = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
        tacc = measure_accuracy(task, fast_learner, train=True)
        vacc = measure_accuracy(task, fast_learner, train=False)
        losses[outstep] += np.array([tl[0], vl_before, tl[-1], vl_after]) / args.task_mb_size
        accuracy[outstep] += np.array([tacc, vacc]) / args.task_mb_size
              
    meta_learner.outer_step_with_grad(meta_grad, flat_grad=True)
    
    logger.log_kv('train_pre', losses[outstep,0])
    logger.log_kv('test_pre', losses[outstep,1])
    logger.log_kv('train_post', losses[outstep,2])
    logger.log_kv('test_post', losses[outstep,3])
    logger.log_kv('train_acc', accuracy[outstep, 0])
    logger.log_kv('val_acc', accuracy[outstep, 1])
    
    if (outstep % 100 == 0 and outstep > 0) or outstep == args.meta_steps - 1:
        # Plot Accuracy
        plt.figure(figsize=(10,6))
        smoothed_acc = smooth_vector(accuracy[:outstep+1], window_size=25)
        plt.plot(smoothed_acc[:,0], label="Train Accuracy")
        plt.plot(smoothed_acc[:,1], label="Validation Accuracy")
        plt.ylim([0.0, 100.0]); plt.xlim([0, args.meta_steps]); plt.grid(True)
        plt.legend(loc=4); plt.title(f'Accuracy Curve for {args.method.upper()}')
        plt.savefig(os.path.join(args.save_dir, 'accuracy_curve.png'), dpi=100)
        plt.close()
        
        # Plot Loss
        plt.figure(figsize=(10,6))
        smoothed_losses = smooth_vector(losses[:outstep+1], window_size=25)
        plt.plot(smoothed_losses[:, 0], label="Train Pre-Update")
        plt.plot(smoothed_losses[:, 1], label="Validation Pre-Update")
        plt.plot(smoothed_losses[:, 2], label="Train Post-Update")
        plt.plot(smoothed_losses[:, 3], label="Validation Post-Update")
        plt.ylim([0.0, 2.0]); plt.xlim([0, args.meta_steps]); plt.grid(True)
        plt.legend(loc=1); plt.title(f'Loss Curves for {args.method.upper()}')
        plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'), dpi=100)
        plt.close()
        
        pickle.dump(meta_learner, open(os.path.join(args.save_dir, 'agent.pickle'), 'wb'))
        logger.save_log(args.save_dir)

print("Training finished.")