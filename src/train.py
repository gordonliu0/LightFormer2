import torch
import os
import subprocess
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from models import LightFormer
from dataset import LightFormerDataset, create_weighted_sampler
from utils.checkpointer import ModelCheckpointer
from utils.lr_scheduler import WarmupCosineScheduler

# Run Name
RUN_NAME = "test1"
DEBUG_VERBOSE = True

# Constants
LISA_DIRECTORY = '/Home/gordonliu1106/Necromancer/data/Kaggle_Dataset'
LISA_PREPROCESSED_DIRECTORY = '/Users/gordonliu/Documents/ml_projects/LightFormer2/data/LISA_Preprocessed'
LISA_DAY_SUBDIRECTORIES = [
    # 'daySequence1',
    # 'daySequence2',
    # 'dayTrain/dayClip1',
    # 'dayTrain/dayClip2',
    # 'dayTrain/dayClip3',
    # 'dayTrain/dayClip4',
    # 'dayTrain/dayClip5',
    # 'dayTrain/dayClip6',
    # 'dayTrain/dayClip7',
    # 'dayTrain/dayClip8',
    # 'dayTrain/dayClip9',
    # 'dayTrain/dayClip10',
    # 'dayTrain/dayClip11',
    # 'dayTrain/dayClip12',
    'dayTrain/dayClip13',]

LISA_NIGHT_SUBDIRECTORIES = [
    'nightSequence1',
    'nightSequence2',
    'nightTrain/nightClip1',
    'nightTrain/nightClip2',
    'nightTrain/nightClip3',
    'nightTrain/nightClip4',
    'nightTrain/nightClip5']
LISA_SAMPLE_SUBDIRECTORIES = [
    'sample-dayClip6',
    'sample-nightClip1',
]
TRAIN_SPLIT = 0.8
TEST_SPLIT  = 0.1
VAL_SPLIT   = 0.1

# Hyperparameters
BATCH_SIZE = 32

# Constant Learning Rate, empirically determined from learning_rate_finder
# LEARNING_RATE = 5e-6
# EPOCHS = 15

# Learning Rate Scheduler: Warmup + SGDR
LEARNING_RATE = 1e-6
WARMUP_STEPS = 2
EPOCHS = 33 # 2 Warmup + 31 SGDR (broken into (1 + 2 + 4 + 8 + 16) steps)

# Training Step, Validation Step, and Training Loop
def train(dataloader, model, loss_fn, optimizer, writer, global_step):
    '''
    Run a single training epoch.

    Args:
        dataloader: Dataloader instance used in training steps.
        model: Pytorch model to use in training backpropagation.
        loss_fn: Loss function to optimize for.
        optimizer: Pytorch optimizer controlling training backpropagation.
        writer: Tensorboard writer to use for scalar value logging.
        global_step: Mutable list that holds global step count.
    '''
    model.train()
    for batch in dataloader:
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

        # Report to tensorboard
        writer.add_scalar("Loss/train/st", st_loss.item(), global_step[0])
        writer.add_scalar("Loss/train/lf", lf_loss.item(), global_step[0])
        writer.add_scalar("Loss/train/total", total_loss.item(), global_step[0])
        global_step[0] += 1

def validate(dataloader, model, loss_fn):
    '''
    Run a validation step.

    Args:
        dataloader: Dataloader instance used in validation.
        model: Pytorch model to use for validation.
        loss_fn: Loss function to calculate loss.
        epoch: Current epoch.
    '''
    # Set to eval mode (no backprop)
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

    # Compute Val Metrics
    size = len(dataloader.dataset)
    st_loss /= size
    lf_loss /= size
    total_loss = st_loss + lf_loss
    st_accuracy /= size
    lf_accuracy /= size
    total_accuracy = (st_accuracy + lf_accuracy) / 2

    # Return
    return st_loss, lf_loss, total_loss, st_accuracy, lf_accuracy, total_accuracy

def run_training(epoch, epochs, train_dataloader, model, train_loss_fn, optimizer, writer, global_step, val_dataloader, val_loss_fn, scheduler, checkpointer: ModelCheckpointer):
    """
    Run training over a number of epochs.

    Args:
        epoch: Epoch to start training from.
        epochs: Epoch to end training before.
        train_dataloader: Dataloader to use in training steps.
        model: Model to perform training on.
        train_loss_fn: Loss function used in training steps.
        optimizer: Optimizer for backprop in training.
        writer: Tensorboard writer instance.
        global_step: Global step of training across epochs.
        val_dataloader: Dataloader for validation steps.
        val_loss_fn: Loss function for validation steps. For sums across batches rather than averages.
        scheduler: Learning rate scheduler to update optimizer instance.
        checkpointer: Controls saving checkpoints every epoch.
    """
    for e in range(epoch, epochs):
        # Train
        train(dataloader=train_dataloader,
              model=model,
              loss_fn=train_loss_fn,
              optimizer=optimizer,
              writer=writer,
              global_step=global_step)

        # Validate
        st_loss, lf_loss, total_loss, st_accuracy, lf_accuracy, total_accuracy = validate(val_dataloader, model, val_loss_fn)

        # Tensorboard Logging
        writer.add_scalar("LR", scheduler.get_last_lr()[0], e)
        writer.add_scalar("Loss/val/st", st_loss, e)
        writer.add_scalar("Loss/val/lf", lf_loss, e)
        writer.add_scalar("Loss/val/total", total_loss, e)
        writer.add_scalar("Acc/val/st", st_accuracy, e)
        writer.add_scalar("Acc/val/lf", lf_accuracy, e)
        writer.add_scalar("Acc/val/total", total_accuracy, e)

        # Update Learning Rate
        scheduler.step()

        # Save Checkpoint after each epoch
        checkpointer.save_checkpoint(
            epoch = e + 1,
            global_step = global_step[0],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=total_loss)

        # Flush writer after each epoch
        writer.flush()

# Constants no matter progress on run.

full_dataset = LightFormerDataset(directory=LISA_DIRECTORY,
                                  preprocessed_directory=LISA_PREPROCESSED_DIRECTORY,
                                  subdirectories=LISA_DAY_SUBDIRECTORIES)
generator=torch.Generator().manual_seed(42)
train_dataset, test_dataset, val_dataset = random_split(full_dataset,
                                                        [TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT],
                                                        generator=generator)
if DEBUG_VERBOSE: print("Created all datasets")

# weighted_sampler=create_weighted_sampler(train_dataset)
# if DEBUG_VERBOSE: print("Created weighted sampler")
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE) #temp unweighted sampling train dataloader
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE) # val doesn't need resampling
if DEBUG_VERBOSE: print("Created all dataloaders")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
if DEBUG_VERBOSE: print(f"Using device {device}")

train_loss_fn = nn.CrossEntropyLoss()
val_loss_fn = nn.CrossEntropyLoss(reduction='sum')
writer = SummaryWriter(log_dir=f"runs/{RUN_NAME}")
checkpointer = ModelCheckpointer(save_dir=f"checkpoints/{RUN_NAME}")
if DEBUG_VERBOSE: print("Created Loss Functions, Tensorboard writer, and Checkpointer")

# The rest depend on whether or not a checkpoint exists.
if len(checkpointer) == 0: # no checkpoints yet, instantiate new values
    if DEBUG_VERBOSE: print("No checkpoints: Instantiating new values.")
    epoch = 0
    global_step = [0]
    model = LightFormer().to(device) # from torchvision package, summary(model, input_size=(batch_size, 10, 3, 512, 960))
    for param in model.backbone.resnet.parameters(): # freeze resnet
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = WarmupCosineScheduler(optimizer=optimizer,
                                    warmup_steps=WARMUP_STEPS,
                                    warmup_start_factor=0.1,
                                    warmup_end_factor=1,
                                    T_0=1,
                                    T_mult=2,
                                    eta_min=1e-7)
else: # checkpoints exist, we are in the middle of training and grab states for next training epoch.
    if DEBUG_VERBOSE: print(f"Checkpoints exist: Loading {checkpointer.checkpoint_files[-1]} and continuing from last epoch")
    checkpoint = checkpointer.load_checkpoint(-1) # load the latest checkpoint
    epoch = checkpoint['epoch']
    global_step = [checkpoint['global_step']]
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']

# RUN TRAINING!
run_training(
    epoch=epoch,
    epochs=EPOCHS,
    train_dataloader=train_dataloader,
    model=model,
    train_loss_fn=train_loss_fn,
    optimizer=optimizer,
    writer=writer,
    global_step=global_step,
    val_dataloader=val_dataloader,
    val_loss_fn=val_loss_fn,
    scheduler=scheduler,
    checkpointer=checkpointer
)

# Run any cleanup bash commands after training finishes. Perform things like logging, storage updates, and stopping compute instances here.
cleanup = subprocess.run(['bash', '../cleanup.sh'], cwd=os.path.dirname(os.path.realpath(__file__)), capture_output=True, text=True, check=True)
print(cleanup.stdout)
