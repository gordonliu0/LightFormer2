import torch
from torchvision import transforms
from models import LightFormer
from torch.utils.data import DataLoader, random_split
from dataset import LightFormerDataset
from torchvision.transforms.functional import invert
import os

# Training Constants
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
MODEL_CHECKPOINT = "/Users/gordonliu/Documents/ml_projects/LightForker-2/src/checkpoints/time_20241014_054238_epoch_13loss_1.110802173614502"
VERBOSE = False

# Datasets
full_dataset = LightFormerDataset(directory=LISA_DAY_DIRECTORIES)
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset, val_dataset = random_split(full_dataset,
                                                        [TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT],
                                                        generator=generator)

# Dataloaders
batch_size = 1
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

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
model.load_state_dict(torch.load(MODEL_CHECKPOINT, weights_only=True))
model.eval()

# untransform image for display
mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)
scale = 255
def untransform(image, mean, std, scale):
    a = image.permute(1, 2, 0)
    a = (a * std + mean)
    a = torch.clamp(a, 0, 1)
    a = a * scale
    a = a.permute(2, 0, 1)
    return a

with torch.no_grad():
    count_st_correct = 0
    count_lf_correct = 0
    for test_index, test_data in enumerate(test_dataloader):
        # grab data
        name = test_data['name'][0]
        x = test_data['images'].to(device)
        y = test_data['label'].view(2, 2).to(device)
        final_image = x[0, 9, :, :, :].squeeze()
        final_image = untransform(final_image, mean, std, scale)
        final_image = invert(final_image)

        # transform to prediction and actual
        st_class, lf_class = model(x)
        st_pred, lf_pred = int(st_class[0].argmax(0).to('cpu')), int(lf_class[0].argmax(0).to('cpu'))
        st_actual, lf_actual = y[0].argmax(0).item(), y[1].argmax(0).item()
        # compare
        st_correct = st_pred == st_actual
        lf_correct = lf_pred == lf_actual

        # gather data
        count_st_correct += st_correct
        count_lf_correct += lf_correct

        # print
        if VERBOSE:
            if test_index % 20 == 0:
                print("-"*(8+30+6+6+40))
                print(f"|{"NAME":<30}|{"st":>6}|{"lf":>6}|{"st_pred":>10}|{"st_actual":>10}|{"lf_pred":>10}|{"lf_actual":>10}|")
                print("-"*(8+30+6+6+40))
            print(f"|{name:<30}|{str(st_correct):>6}|{str(lf_correct):>6}|{st_pred:>10}|{st_actual:>10}|{lf_pred:>10}|{lf_actual:>10}|")


        # Display for debug
        # if not (st_correct and lf_correct):
        #     im = transforms.ToPILImage()(final_image)
        #     im = im.open()

    print(f"Correct Straight: {count_st_correct}/{len(test_dataset)}")
    print(f"Correct Left: {count_lf_correct}/{len(test_dataset)}")
