
import sys
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from glob import glob
from PIL import Image
from torchvision.transforms import transforms

# --- FIX THE IMPORT PATH ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- END OF FIX ---

from implicit_maml.learner_model import Learner, make_conv_network
from implicit_maml.dataset import OmniglotTask, OmniglotFewShotDataset
from implicit_maml.utils import measure_accuracy

# --- Seeding for reproducibility ---
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ==============================================================================
#      Dataset for Standard Supervised Pre-training
# ==============================================================================
class OmniglotFlatDataset(Dataset):
    """
    Creates a flat dataset from the Omniglot meta-training set for standard
    supervised classification training.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = os.path.join(data_dir, 'images_background')
        self.transform = transform
        self.image_paths = sorted(glob(os.path.join(self.data_dir, '*/*/*.png')))
        
        self.class_dirs = sorted(glob(os.path.join(self.data_dir, '*/*')))
        self.class_to_idx = {os.path.basename(c): i for i, c in enumerate(self.class_dirs)}
        self.num_classes = len(self.class_dirs)
        
        print(f"Found {len(self.image_paths)} images from {self.num_classes} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Class name is the parent directory (e.g., 'character01')
        class_name = os.path.basename(os.path.dirname(image_path))
        label = self.class_to_idx[class_name]
        
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==============================================================================
#      Baseline 1: Train From Scratch (Weak Baseline)
# ==============================================================================
def run_from_scratch_baseline(args):
    print("\n--- Running WEAK BASELINE: Training from Scratch ---")
    
    # Use the test set characters for evaluation
    train_val_permutation = list(range(1623))
    task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, num_cls=args.N_way, num_inst=args.K_shot, train=False) for _ in tqdm(range(args.num_tasks_test), desc="Generating Test Tasks")]
    dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)

    accuracies = []
    for i in tqdm(range(len(dataset)), desc="Evaluating Tasks"):
        # 1. Create a new model with random weights for each task
        model = make_conv_network(in_channels=1, out_dim=args.N_way).to(args.device)
        learner = Learner(model, nn.CrossEntropyLoss(), inner_lr=args.inner_lr, GPU=args.use_gpu)
        
        task = dataset[i]
        
        # 2. Train this random model on the support set
        learner.learn_task(task, num_steps=args.n_steps)
        
        # 3. Evaluate on the query set
        acc = measure_accuracy(task, learner, train=False)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_dev = np.std(accuracies)
    conf_interval = 1.96 * std_dev / np.sqrt(len(accuracies))
    
    print("\n--- From Scratch Results ---")
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"95% Confidence Interval: +/- {conf_interval:.2f}%")

# ==============================================================================
#      Baseline 2, Stage 1: Pre-training
# ==============================================================================
def run_pretraining(args):
    print("\n--- Running STRONG BASELINE (Stage 1): Pre-training ---")
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    dataset = OmniglotFlatDataset(args.data_dir, transform=transform)
    
    # 🔍 Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = make_conv_network(in_channels=1, out_dim=dataset.num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Starting pre-training for {args.pretrain_epochs} epochs...")
    print(f"Total classes: {dataset.num_classes}")
    
    for epoch in range(args.pretrain_epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.pretrain_epochs}"):
            images, labels = images.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                pred = outputs.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    torch.save(model.state_dict(), args.pretrained_model_path)
    print(f"\nPre-trained model saved to {args.pretrained_model_path}")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    
def sanity_check_finetuning(args):
    """Quick test to see if fine-tuning can overfit one task"""
    print("\n=== SANITY CHECK: Can we overfit a single task? ===")
    
    if not os.path.exists(args.pretrained_model_path):
        print(f"ERROR: Pre-trained model not found")
        return
    
    pretrained_state_dict = torch.load(args.pretrained_model_path, map_location=args.device)
    
    # Create ONE task
    train_val_permutation = list(range(964))
    task_def = OmniglotTask(train_val_permutation, root=args.data_dir, 
                           num_cls=args.N_way, num_inst=args.K_shot, train=True)
    dataset = OmniglotFewShotDataset(task_defs=[task_def], GPU=args.use_gpu)
    task = dataset[0]
    
    # Print all keys to see what's available
    print(f"Task keys: {task.keys()}")
    
    # Fix: Use the correct key names
    xt = task['x_train']  # or whatever the support set key is
    yt = task['y_train']  # or whatever the support label key is
    
    # Find the query set keys (try common variations)
    if 'x_test' in task:
        xq, yq = task['x_test'], task['y_test']
    elif 'x_query' in task:
        xq, yq = task['x_query'], task['y_query']
    elif 'x_val' in task:
        xq, yq = task['x_val'], task['y_val']
    else:
        print(f"Available keys: {list(task.keys())}")
        print("ERROR: Cannot find query set keys!")
        return
    
    print(f"Task info:")
    print(f"  Support set: {xt.shape}")
    print(f"  Support labels: {yt}")
    print(f"  Query set: {xq.shape}")
    print(f"  Query labels: {yq}")
    
    # Load model
    model = make_conv_network(in_channels=1, out_dim=args.N_way).to(args.device)
    filtered_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('fc1.')}
    model.load_state_dict(filtered_dict, strict=False)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print("\n=== Training Progress ===")
    model.train()
    for step in range(200):
        optimizer.zero_grad()
        outputs = model(xt)
        loss = nn.CrossEntropyLoss()(outputs, yt)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            with torch.no_grad():
                # Support set accuracy
                train_pred = outputs.argmax(dim=1)
                train_acc = (train_pred == yt).float().mean().item() * 100
                
                # Query set accuracy
                model.eval()
                query_outputs = model(xq)
                query_pred = query_outputs.argmax(dim=1)
                query_acc = (query_pred == yq).float().mean().item() * 100
                model.train()
                
                print(f"Step {step:3d}: Loss={loss.item():.4f}, "
                      f"Support Acc={train_acc:5.1f}%, Query Acc={query_acc:5.1f}%")
    
    print("\n=== Final Predictions ===")
    model.eval()
    with torch.no_grad():
        # Support set
        train_outputs = model(xt)
        train_pred = train_outputs.argmax(dim=1)
        print(f"Support - True: {yt.cpu().numpy()}")
        print(f"Support - Pred: {train_pred.cpu().numpy()}")
        print(f"Support - Match: {(train_pred == yt).cpu().numpy()}")
        
        # Query set
        query_outputs = model(xq)
        query_pred = query_outputs.argmax(dim=1)
        print(f"\nQuery - True: {yq.cpu().numpy()[:20]}")
        print(f"Query - Pred: {query_pred.cpu().numpy()[:20]}")
        print(f"Query - Match: {(query_pred == yq).cpu().numpy()[:20]}")
        
        final_acc = (query_pred == yq).float().mean().item() * 100
        print(f"\n=== FINAL QUERY ACCURACY: {final_acc:.2f}% ===")        
    # ==============================================================================
#      Baseline 2, Stage 2: Fine-tuning (Strong Baseline)
# ==============================================================================
# Replace the old function in examples/baselines.py with this one

# def run_finetune_baseline(args):
#     print("\n--- Running STRONG BASELINE (Stage 2): Fine-tuning ---")
    
#     if not os.path.exists(args.pretrained_model_path):
#         print(f"ERROR: Pre-trained model not found at {args.pretrained_model_path}")
#         print("Please run pre-training first with: python examples/baselines.py --mode pretrain ...")
#         return

#     # Load the state dict from the pre-trained model
#     pretrained_state_dict = torch.load(args.pretrained_model_path, map_location=args.device)
    
#     # Generate test tasks
#     train_val_permutation = list(range(1623))
#     task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, num_cls=args.N_way, num_inst=args.K_shot, train=False) for _ in tqdm(range(args.num_tasks_test), desc="Generating Test Tasks")]
#     dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)

#     accuracies = []
#     for i in tqdm(range(len(dataset)), desc="Evaluating Tasks"):
#         # 1. Create a new N-way model for the few-shot task
#         model = make_conv_network(in_channels=1, out_dim=args.N_way).to(args.device)
        
#         # --- THIS IS THE CRITICAL FIX ---
#         # 2. Filter out the final layer ('fc1') from the pre-trained dictionary
#         #    so that we only load the convolutional feature extractor weights.
#         filtered_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('fc1.')}
        
#         # 3. Load the filtered weights into the new model.
#         model.load_state_dict(filtered_pretrained_dict, strict=False)
#         # The `fc1` layer of the new model will keep its random initialization.
#         # --- END OF FIX ---

#         # 4. Create an optimizer that ONLY trains the final layer
#         optimizer = torch.optim.SGD(model.fc1.parameters(), lr=args.inner_lr)
        
#         task = dataset[i]
#         xt, yt = task['x_train'], task['y_train']
        
#         # 5. Fine-tune the final layer on the support set
#         for _ in range(args.n_steps * 2): # Fine-tune for a bit longer
#             optimizer.zero_grad()
#             loss = nn.CrossEntropyLoss()(model(xt), yt)
#             loss.backward()
#             optimizer.step()
            
#         # 6. Evaluate on the query set
#         eval_learner = Learner(model, nn.CrossEntropyLoss(), GPU=args.use_gpu)
#         acc = measure_accuracy(task, eval_learner, train=False)
#         accuracies.append(acc)

#     mean_acc = np.mean(accuracies)
#     std_dev = np.std(accuracies)
#     conf_interval = 1.96 * std_dev / np.sqrt(len(accuracies))
    
#     print("\n--- Fine-tuning Results ---")
#     print(f"Mean Accuracy: {mean_acc:.2f}%")
#     print(f"95% Confidence Interval: +/- {conf_interval:.2f}%")


# In examples/baselines.py, replace the run_finetune_baseline function again.
def run_finetune_baseline(args):
    print("\n--- Running STRONG BASELINE (Stage 2): Fine-tuning ---")
    
    if not os.path.exists(args.pretrained_model_path):
        print(f"ERROR: Pre-trained model not found at {args.pretrained_model_path}")
        return

    pretrained_state_dict = torch.load(args.pretrained_model_path, map_location=args.device)
    
    # FIX #1: Use train=True to test on background classes that were pre-trained on
    train_val_permutation = list(range(964))  # Match pre-training classes
    task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, 
                              num_cls=args.N_way, num_inst=args.K_shot, 
                              train=True) for _ in tqdm(range(args.num_tasks_test))]  # train=True!
    dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)

    accuracies = []
    for i in tqdm(range(len(dataset)), desc="Evaluating Tasks"):
        model = make_conv_network(in_channels=1, out_dim=args.N_way).to(args.device)
        
        filtered_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() 
                                   if not k.startswith('fc1.')}
        model.load_state_dict(filtered_pretrained_dict, strict=False)

        # FIX #2 & #3: Higher learning rate and more steps
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Proper learning rate
        
        task = dataset[i]
        xt, yt = task['x_train'], task['y_train']
        
        # FIX #4: More training steps
        for _ in range(100):  # Increase from 20 to 100
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(xt), yt)
            loss.backward()
            optimizer.step()
            
        eval_learner = Learner(model, nn.CrossEntropyLoss(), GPU=args.use_gpu)
        acc = measure_accuracy(task, eval_learner, train=False)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_dev = np.std(accuracies)
    conf_interval = 1.96 * std_dev / np.sqrt(len(accuracies))
    
    print("\n--- Fine-tuning Results ---")
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"95% Confidence Interval: +/- {conf_interval:.2f}%")
# def run_finetune_baseline(args):
#     print("\n--- Running STRONG BASELINE (Stage 2): Fine-tuning ---")
    
#     if not os.path.exists(args.pretrained_model_path):
#         print(f"ERROR: Pre-trained model not found at {args.pretrained_model_path}")
#         return

#     pretrained_state_dict = torch.load(args.pretrained_model_path, map_location=args.device)
    
#     train_val_permutation = list(range(1623))
#     task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, num_cls=args.N_way, num_inst=args.K_shot, train=False) for _ in tqdm(range(args.num_tasks_test), desc="Generating Test Tasks")]
#     dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)

#     accuracies = []
#     for i in tqdm(range(len(dataset)), desc="Evaluating Tasks"):
#         model = make_conv_network(in_channels=1, out_dim=args.N_way).to(args.device)
        
#         filtered_pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('fc1.')}
#         model.load_state_dict(filtered_pretrained_dict, strict=False)

#         # --- THIS IS THE KEY CHANGE ---
#         # Instead of only training the head, we will fine-tune the ENTIRE network.
#         # This gives the model more flexibility to adapt.
#         # Use a slightly smaller learning rate as we are updating more layers.
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.inner_lr / 10) 
#         # --- END OF CHANGE ---
        
#         task = dataset[i]
#         xt, yt = task['x_train'], task['y_train']
        
#         for _ in range(args.n_steps * 2):
#             optimizer.zero_grad()
#             loss = nn.CrossEntropyLoss()(model(xt), yt)
#             loss.backward()
#             optimizer.step()
            
#         eval_learner = Learner(model, nn.CrossEntropyLoss(), GPU=args.use_gpu)
#         acc = measure_accuracy(task, eval_learner, train=False)
#         accuracies.append(acc)

#     mean_acc = np.mean(accuracies)
#     std_dev = np.std(accuracies)
#     conf_interval = 1.96 * std_dev / np.sqrt(len(accuracies))
    
#     print("\n--- Fine-tuning Results (Full Network) ---")
#     print(f"Mean Accuracy: {mean_acc:.2f}%")
#     print(f"95% Confidence Interval: +/- {conf_interval:.2f}%")
# ==============================================================================
#      Main Script Logic
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run baseline experiments for Few-Shot Learning')
    
    # Remove the duplicate! Only keep this one:
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['from_scratch', 'pretrain', 'finetune', 'sanity_check'],
                        help='Which baseline to run.')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Omniglot data folder')
    
    # Common args
    parser.add_argument('--N_way', type=int, default=5)
    parser.add_argument('--K_shot', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=10, help='Inner loop steps for adaptation')
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--use_gpu', type=lambda x: (str(x).lower() == 'true'), default=True)
    
    # Args for testing
    parser.add_argument('--num_tasks_test', type=int, default=600, help='Number of tasks to evaluate on')
    
    # Args for pre-training
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_cnn.pth',
                        help='Path to save/load the pre-trained model')
    
    args = parser.parse_args()
    args.device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    
    if args.mode == 'sanity_check':
        sanity_check_finetuning(args)
    elif args.mode == 'from_scratch':
        run_from_scratch_baseline(args)
    elif args.mode == 'pretrain':
        run_pretraining(args)
    elif args.mode == 'finetune':
        run_finetune_baseline(args)