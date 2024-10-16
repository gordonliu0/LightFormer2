import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from dataset.dataset import LightFormerDataset
from models import LightFormer
from util import run_with_animation

def find_lr(model, train_loader, optimizer, criterion, min_lr=1e-8, max_lr=1e-3, num_iterations=20):
    model.train()
    lrs, losses = [], []
    log_lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_iterations)
    for lr in log_lrs:
        print(lr)
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad()

        # Load sample and label
        batch = next(iter(train_loader))
        X = batch['images'].to('mps')
        y = batch['label'].to('mps').view(len(X), 2, 2).permute(1, 0, 2)

        # Compute prediction
        st_class, lf_class = model(X)

        # Compute loss (averaged across the batch)
        st_loss = criterion(st_class, y[0])
        lf_loss = criterion(lf_class, y[1])
        loss = st_loss + lf_loss

        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())

    return lrs, losses

def plot_losses_vs_lrs(lrs, losses):
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.grid(True)
    plt.show()

# Usage

# Dataset and Dataloader
LISA_DAY_DIRECTORIES = [
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/daySequence1',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/daySequence2',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip1',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip2',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip3',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip4',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip5',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip6',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip7',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip8',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip9',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip10',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip11',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip12',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip13',]
LISA_NIGHT_DIRECTORIES = [
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightSequence1',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightSequence2',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip1',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip2',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip3',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip4',
    '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip5']
TRAIN_SPLIT = 0.8
TEST_SPLIT  = 0.1
VAL_SPLIT   = 0.1
full_dataset = LightFormerDataset(directory=LISA_DAY_DIRECTORIES)
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset, val_dataset = random_split(full_dataset,
                                                        [TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT],
                                                        generator=generator)
train_dataloader = DataLoader(train_dataset, batch_size=16)

# Model, Optimizer, Criterion
device='mps'
model = LightFormer().to(device)
optimizer = Adam(model.parameters(), lr=1e-8)
criterion = nn.CrossEntropyLoss()

output = run_with_animation(find_lr, args=(model, train_dataloader, optimizer, criterion), name="find_learning_rate")
plot_losses_vs_lrs(*output)
