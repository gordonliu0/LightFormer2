import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from models import LightFormer
from checkpointer import ModelCheckpointer
from lr_scheduler import WarmupCosineScheduler
from dataset import LightFormerDataset
from util import run_with_animation

# Constants
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

# Training Parameters
VERBOSE = True

# Training Hyperparameters
BATCH_SIZE = 16

# Constant Learning Rate, empirically determined from learning_rate_finder
# LEARNING_RATE = 5e-6
# EPOCHS = 15

# Learning Rate Scheduler: Warmup + SGDR
LEARNING_RATE = 1e-6
WARMUP_STEPS = 2
EPOCHS = 33 # 2 Warmup + 31 SGDR (broken into (1 + 2 + 4 + 8 + 16) steps)

# Dataset Functions

def count_classes(dataset):
    'Count each type of class in the dataset.'
    # Example Usage:
    # print("Start Counting Class Frequencies")
    # print(count_classes(train_dataset))
    # print(count_classes(val_dataset))
    # print(count_classes(test_dataset))
    class0 = 0
    class1 = 0
    class2 = 0
    class3 = 0
    for data in dataset:
        y = data['label']
        if y[0] == 1:
            class0 += 1
        if y[1] == 1:
            class1 += 1
        if y[2] == 1:
            class2 += 1
        if y[3] == 1:
            class3 += 1
    return class0, class1, class2, class3

def count_labels(dataset):
    'Count each type of label in the dataset.'
    type0 = (1., 0., 1., 0.)
    type1 = (1., 0., 0., 1.)
    type2 = (0., 1., 1., 0.)
    type3 = (0., 1., 0., 1.)
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for data in dataset:
        y = data['label']
        if y[0] == 1:
            if y[2] == 1:
                count0 += 1
            else:
                count1 += 1
        else:
            if y[2] == 1:
                count2 += 1
            else:
                count3 += 1
    return {type0: count0, type1: count1, type2: count2, type3: count3}

def create_weighted_sampler(dataset):
    'Weighted samplers help with unbalanced datasets.'
    # label_counts = count_labels(dataset)
    label_counts = {(1., 0., 1., 0.): 211, (1., 0., 0., 1.): 272, (0., 1., 1., 0.): 43, (0., 1., 0., 1.): 703}
    class_weights = {class_label: 1.0 / count for class_label, count in label_counts.items()}
    sample_weights = [class_weights[tuple(sample["label"].tolist())] for sample in dataset]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# Dataset, Dataloader
full_dataset = LightFormerDataset(directory=LISA_DAY_DIRECTORIES)
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset, val_dataset = random_split(full_dataset,
                                                        [TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT],
                                                        generator=generator)
weighted_sampler = run_with_animation(f=create_weighted_sampler,args=(train_dataset,), name='Weighted Sampler')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE) # val doesn't need resampling

# Setup Device, Model, Loss, Optimizer, Checkpointer, LR Scheduler, Tensorboard Writer
# To view a summary, run summary(model, input_size=(batch_size, 10, 3, 512, 960)).
# ** Note: Don't run this before a training job. **
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = LightFormer().to(device)
for param in model.backbone.resnet.parameters(): # freeze resnet
    param.requires_grad = False
loss_fn = nn.CrossEntropyLoss()
val_loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
checkpointer = ModelCheckpointer('checkpoints', max_saves=5)
scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_steps=WARMUP_STEPS, warmup_start_factor=0.1, warmup_end_factor=1, T_0=1, T_mult=2, eta_min=0.1)
writer = SummaryWriter()

# Training, Validation, and Training Loop
def train(dataloader, model, loss_fn, optimizer, global_step, batches_per_log=5):
    '''
    Run a training epoch.

    Args:
        dataloader: Dataloader instance used in training steps.
        model: Pytorch model to use in training backpropagation.
        loss_fn: Loss function to optimize for.
        optimizer: Pytorch optimizer controlling training backpropagation.
        global_step: Mutable list that holds global step count.
        batches_per_log: How many batches for every log. Only prints if VERBOSE = True.
    '''
    # Last Loss holds average loss for the last batches_per_log
    st = { 'running_loss': 0., 'last_loss': 0.}
    lf = { 'running_loss': 0., 'last_loss': 0.}
    total = { 'running_loss': 0., 'last_loss': 0.}

    size = len(dataloader.dataset)
    model.train()
    for batch_index, batch in enumerate(dataloader):

        # Load sample and label
        X = batch['images'].to(device)
        y = batch['label'].to(device).view(len(X), 2, 2).permute(1, 0, 2)

        # Compute prediction
        st_class, lf_class = model(X)

        # Compute loss (averaged across the batch)
        st_loss = loss_fn(st_class, y[0])
        lf_loss = loss_fn(lf_class, y[1])
        total_loss = st_loss + lf_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Gather Data and Report
        st['running_loss'] += st_loss.item()
        lf['running_loss'] += lf_loss.item()
        total['running_loss'] += total_loss.item()

        writer.add_scalar("Loss/train/st", st_loss.item(), global_step[0])
        writer.add_scalar("Loss/train/lf", lf_loss.item(), global_step[0])
        writer.add_scalar("Loss/train/total", total_loss.item(), global_step[0])
        global_step[0] += 1

        if VERBOSE and ((batch_index+1) % batches_per_log == 0):
            current_sample_index = (batch_index + 1) * len(X)
            st['last_loss'] = st['running_loss'] / batches_per_log
            lf['last_loss'] = lf['running_loss'] / batches_per_log
            total['last_loss'] = total['running_loss'] / batches_per_log
            print()
            print(f"st_loss: {st['last_loss']:>7f} lf_loss: {lf['last_loss']:>7f} total_loss: {total['last_loss']:>7f}  Sample [{current_sample_index:>5d}/{size:>5d}], Batch {batch_index}")
            st['running_loss'] = 0
            lf['running_loss'] = 0
            total['running_loss'] = 0

def validate(dataloader, model, loss_fn, epoch):
    '''
    Run a validation step.

    Args:
        dataloader: Dataloader instance used in validation.
        model: Pytorch model to use for validation.
        loss_fn: Loss function to calculate loss.
        epoch: Current epoch.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    st_loss, lf_loss, st_accuracy, lf_accuracy = 0, 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            # Load sample and label
            X = batch['images'].to(device)
            y = batch['label'].to(device).view(len(X), 2, 2).permute(1, 0, 2)

            # Compute predictions
            st_class, lf_class = model(X)

            # Compute loss
            step_st_loss = loss_fn(st_class, y[0])
            step_lf_loss = loss_fn(lf_class, y[1])

            # Gather data
            st_loss += step_st_loss
            lf_loss += step_lf_loss
            st_accuracy += (st_class.argmax(1) == y[0].argmax(1)).type(torch.float).sum().item()
            lf_accuracy += (lf_class.argmax(1) == y[1].argmax(1)).type(torch.float).sum().item()
    st_loss /= size
    lf_loss /= size
    total_loss = st_loss + lf_loss
    st_accuracy /= size
    lf_accuracy /= size
    total_accuracy = (st_accuracy + lf_accuracy) / 2

    if VERBOSE:
        print(f"Test Error: \n Straight Accuracy: {(100*st_accuracy):>0.1f}%, Left Accuracy: {(100*lf_accuracy):>0.1f}%, Average validation loss: {total_loss:>8f} \n")

    writer.add_scalar("Loss/val/st", st_loss, epoch)
    writer.add_scalar("Loss/val/lf", lf_loss, epoch)
    writer.add_scalar("Loss/val/total", total_loss, epoch)
    writer.add_scalar("Acc/val/st", st_accuracy, epoch)
    writer.add_scalar("Acc/val/lf", lf_accuracy, epoch)
    writer.add_scalar("Acc/val/total", total_accuracy, epoch)

    return st_loss, lf_loss, total_loss, st_accuracy, lf_accuracy, total_accuracy

def run_training():
    global_step = [0]
    for epoch in range(EPOCHS):
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}, lr={current_lr}")
        writer.add_scalar("LR", current_lr, epoch)
        train(train_dataloader, model, loss_fn, optimizer, global_step, batches_per_log=8)
        _, _, metric, _, _, _ = validate(val_dataloader, model, val_loss_fn, epoch)
        scheduler.step()
        checkpointer.save_checkpoint(model, epoch=epoch, metric=metric)
    print(f"🎉 Finished training {EPOCHS} epochs for LightFormer2!")
    torch.save(model.state_dict(), "checkpoints/final.pth")

run_training()
writer.flush()
