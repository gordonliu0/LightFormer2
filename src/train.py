import torch
import os
import subprocess
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from models import LightFormer, ChimeFormer
from dataset import LightFormerDataset, create_weighted_sampler
from utils.checkpointer import ModelCheckpointer
from utils.lr_scheduler import WarmupCosineScheduler
import json
import argparse
from torchinfo import summary

# Training Step, Validation Step, and Training Loop
def train(dataloader, model, loss_fn, optimizer, writer, global_step, verbose):
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

        # Verbose Prints
        if verbose:
            print(f"Step {global_step}: total_loss {total_loss}")

        # Report to tensorboard
        writer.add_scalar("Loss/train/st", st_loss.item(), global_step[0])
        writer.add_scalar("Loss/train/lf", lf_loss.item(), global_step[0])
        writer.add_scalar("Loss/train/total", total_loss.item(), global_step[0])
        global_step[0] += 1

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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

def run_training(epoch, epochs, train_dataloader, model, train_loss_fn, optimizer, writer, global_step, val_dataloader, val_loss_fn, checkpointer: ModelCheckpointer, verbose, scheduler=None):
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
        checkpointer: Controls saving checkpoints every epoch.
        verbose: Adds debug print statements.
        scheduler: Learning rate scheduler to update optimizer instance.
    """
    for e in range(epoch, epochs):
        # Train
        train(dataloader=train_dataloader,
              model=model,
              loss_fn=train_loss_fn,
              optimizer=optimizer,
              writer=writer,
              global_step=global_step,
              verbose=verbose)

        # Validate
        st_loss, lf_loss, total_loss, st_accuracy, lf_accuracy, total_accuracy = validate(val_dataloader, model, val_loss_fn)

        # Tensorboard Logging
        if scheduler:
            writer.add_scalar("LR", scheduler.get_last_lr()[0], e)
        writer.add_scalar("Loss/val/st", st_loss, e)
        writer.add_scalar("Loss/val/lf", lf_loss, e)
        writer.add_scalar("Loss/val/total", total_loss, e)
        writer.add_scalar("Acc/val/st", st_accuracy, e)
        writer.add_scalar("Acc/val/lf", lf_accuracy, e)
        writer.add_scalar("Acc/val/total", total_accuracy, e)

        # Update Learning Rate
        if scheduler:
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

        # Verbose Logs
        if verbose:
            print(f"Epoch {e} complete. Validation Metrics: total_loss {total_loss}, total_accuracy {total_accuracy}")

# Args and Configs
def parse_args():
    parser = argparse.ArgumentParser(description='Script to train LightFormer with multiple runs. Configure runs using multiple config json files. Example is provided in configs directory.')
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        nargs='+',
        help='Paths to one or more config JSON files'
    )
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load a single config file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{config_path}' is not a valid JSON file")
        return None

def load_configs(config_paths):
    """
    Load multiple config files.
    """
    configs = []
    for path in config_paths:
        config = load_config(path)
        if config is not None:
            configs.append(config)
    return configs

# Start Training Script
args = parse_args()
configs = load_configs(args.config)
config = load_config("/Users/gordonliu/Documents/ml_projects/LightFormer2/configs/chime_2.json")
chimeformer = ChimeFormer(config=config)
summary(chimeformer, (16, 10, 3, 512, 960))

# Run multiple trainings, each one starting if not finished.
for config in configs:
    RUN_NAME = config["run_name"]
    DEBUG_VERBOSE = config["debug_verbose"]

    LISA_ROOT_DIR = config["dataset"]["lisa"]["base"]
    LISA_PREPROCESSED_DIR = config["dataset"]["lisa"]["preprocessed"]
    LISA_DAY_SUBDIR = config["dataset"]["lisa"]["day_subdirectories"]
    LISA_NIGHT_SUBDIR = config["dataset"]["lisa"]["night_subdirectories"]
    LISA_SAMPLE_SUBDIR = config["dataset"]["lisa"]["sample_subdirectories"]

    TRAIN_SPLIT = config["splits"]["train"]
    TEST_SPLIT = config["splits"]["test"]
    VAL_SPLIT = config["splits"]["validation"]

    FREEZE_BACKBONE = config["training"]["freeze_backbone"]
    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["lr"]
    USE_LR_SCHEDULER = config["training"]["use_lr_scheduler"]
    WARMUP_STEPS = config["training"]["lr_scheduler"]["warmup_steps"]
    WARMUP_START_FACTOR = config["training"]["lr_scheduler"]["warmup_start_factor"]
    WARMUP_END_FACTOR = config["training"]["lr_scheduler"]["warmup_end_factor"]
    T_0 = config["training"]["lr_scheduler"]["t_0"]
    T_MULT = config["training"]["lr_scheduler"]["t_mult"]
    ETA_MIN = config["training"]["lr_scheduler"]["eta_min"]
    EPOCHS = config["training"]["epochs"]

    # Constants no matter progress on run.
    full_dataset = LightFormerDataset(directory=LISA_ROOT_DIR,
                                    preprocessed_directory=LISA_PREPROCESSED_DIR,
                                    subdirectories=LISA_DAY_SUBDIR)
    generator=torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(full_dataset,
                                                            [TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT],
                                                            generator=generator)
    if DEBUG_VERBOSE: print("Created all datasets")

    weighted_sampler=create_weighted_sampler(train_dataset)
    if DEBUG_VERBOSE: print("Created weighted sampler")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)

    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE) #temp unweighted sampling train dataloader
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
        model = ChimeFormer(config=config).to(device)
        if FREEZE_BACKBONE:
            for param in model.backbone.resnet.parameters(): # freeze resnet
                param.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if USE_LR_SCHEDULER:
            scheduler = WarmupCosineScheduler(optimizer=optimizer,
                                            warmup_steps=WARMUP_STEPS,
                                            warmup_start_factor=WARMUP_START_FACTOR,
                                            warmup_end_factor=WARMUP_END_FACTOR,
                                            T_0=T_0,
                                            T_mult=T_MULT,
                                            eta_min=ETA_MIN)
        else:
            scheduler = None
    else: # checkpoints exist, we are in the middle of training and grab states for next training epoch.
        checkpoint = checkpointer.load_latest() # load the latest checkpoint
        if DEBUG_VERBOSE: print(f"Checkpoints exist: Loading {checkpoint["name"]} and continuing from last epoch")
        epoch = checkpoint['epoch']
        global_step = [checkpoint['global_step']]
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

    # RUN TRAINING!
    print("Starting run:", RUN_NAME)
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
        checkpointer=checkpointer,
        verbose=DEBUG_VERBOSE,
        scheduler=scheduler,
    )

# Run any cleanup bash commands after training finishes. Perform things like logging, storage updates, and stopping compute instances here.
cleanup = subprocess.run(['bash', '../cleanup.sh'], cwd=os.path.dirname(os.path.realpath(__file__)), capture_output=True, text=True, check=True)
print(cleanup.stdout)
