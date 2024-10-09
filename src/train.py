import torch
from torch import nn
from models import LightFormer
from torch.utils.data import DataLoader, random_split
from dataset import LightFormerDataset
from torchinfo import summary
import os
from datetime import datetime

# Training Constants
DIRECTORIES = [ '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/daySequence1',
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
                '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/dayTrain/dayClip13',
                '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightSequence1',
                '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightSequence2',
                '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip1',
                '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip2',
                '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip3',
                '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip4',
                '/Users/gordonliu/Documents/ml_projects/LightForker-2/data/Kaggle_Dataset/nightTrain/nightClip5',]
TRAIN_SPLIT = 0.8
TEST_SPLIT  = 0.1
VAL_SPLIT   = 0.1

# Datasets
full_dataset = LightFormerDataset(directory=DIRECTORIES)
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset, val_dataset = random_split(full_dataset,
                                                        [TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT],
                                                        generator=generator)

# Dataloaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Check that dataset is working and labels are being pulled correctly
# def count_classes(dataset):
#     class0 = 0
#     class1 = 0
#     class2 = 0
#     class3 = 0
#     for data in dataset:
#         y = data['label']
#         if y[0] == 1:
#             class0 += 1
#         if y[1] == 1:
#             class1 += 1
#         if y[2] == 1:
#             class2 += 1
#         if y[3] == 1:
#             class3 += 1
#     return class0, class1, class2, class3
# print(count_classes(train_dataset))
# print(count_classes(val_dataset))
# print(count_classes(test_dataset))

# Model
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'Using device "{device}"')
if device == "mps":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # used to provide fallback if MPS
model = LightFormer().to(device)

# Freeze Resnet Layers
def freeze_backbone(model):
    for param in model.backbone.resnet.parameters():
        param.requires_grad = False
freeze_backbone(model=model)

# LightFormer Summary
# print(model)
# summary(model, input_size=(batch_size, 10, 3, 512, 960))

# Loss and Optimizer for Training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters())

def train(dataloader, model, loss_fn, optimizer, batches_per_log=5):
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

        print(f'{batch_index}', end=' ', flush=True)

        if (batch_index+1) % batches_per_log == 0:
            current_sample_index = (batch_index + 1) * len(X)
            st['last_loss'] = st['running_loss'] / batches_per_log
            lf['last_loss'] = lf['running_loss'] / batches_per_log
            total['last_loss'] = total['running_loss'] / batches_per_log
            print()
            print(f"st_loss: {st['last_loss']:>7f} lf_loss: {lf['last_loss']:>7f} total_loss: {total['last_loss']:>7f}  Sample [{current_sample_index:>5d}/{size:>5d}], Batch {batch_index}")
            st['running_loss'] = 0
            lf['running_loss'] = 0
            total['running_loss'] = 0

def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, st_correct, lf_correct,  = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            # Load sample and label
            X = batch['images'].to(device)
            y = batch['label'].to(device).view(len(X), 2, 2).permute(1, 0, 2)

            # Compute predictions
            st_class, lf_class = model(X)

            # Compute loss
            st_loss = loss_fn(st_class, y[0])
            lf_loss = loss_fn(lf_class, y[1])
            total_loss = st_loss+lf_loss

            # Gather data
            val_loss += total_loss.item()
            st_correct += (st_class.argmax(1) == y[0].argmax(1)).type(torch.float).sum().item()
            lf_correct += (lf_class.argmax(1) == y[1].argmax(1)).type(torch.float).sum().item()
    val_loss /= num_batches
    st_accuracy = st_correct/size
    lf_accuracy = lf_correct/size
    print(f"Test Error: \n Straight Accuracy: {(100*st_accuracy):>0.1f}%, Left Accuracy: {(100*lf_accuracy):>0.1f}%, Average validation loss: {val_loss:>8f} \n")
    return val_loss

epochs = 10
for t in range(epochs):
    best_vloss = 1000000
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    avg_vloss = validate(val_dataloader, model, loss_fn)
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = 'model_{}_epoch_{}'.format(timestamp, t)
        torch.save(model.state_dict(), model_path)
print("Done!")

torch.save(model.state_dict(), "model_final.pth")
print("Saved Final PyTorch Model State to model_final.pth")

# model = LightFormer().to(device)
# model.load_state_dict(torch.load("model.pth", weights_only=True))

# with torch.no_grad():
#     for i in range(10):
#         x, y = test_dataset[i]
#         x = x.to(device)
#         pred = model(x)
#         predicted, actual = int(pred[0].argmax(0).to('cpu')), y
#         if (not predicted == actual):
#             im = transforms.ToPILImage()(x)
#             im.show()
#         print(predicted is actual)
