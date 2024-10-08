import torch
from torch import nn
from models import LightFormer
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import LightFormerDataset
from torchinfo import summary

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

# Dataset and Dataloaders
full_dataset = LightFormerDataset(directory=DIRECTORIES)
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset, val_dataset = random_split(full_dataset,
                                                        [TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT],
                                                        generator=generator)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Model
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = LightFormer().to(device)

# Freeze Resnet Layers
def freeze_backbone(model):
    for param in model.backbone.resnet.parameters():
        param.requires_grad = False
freeze_backbone(model=model)
summary(model, input_size=(batch_size, 10, 3, 512, 960))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = LightFormer().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

with torch.no_grad():
    for i in range(10):
        x, y = test_dataset[i]
        x = x.to(device)
        pred = model(x)
        predicted, actual = int(pred[0].argmax(0).to('cpu')), y
        if (not predicted == actual):
            im = transforms.ToPILImage()(x)
            im.show()
        print(predicted is actual)
