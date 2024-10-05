import torch
from torch.utils.data import  Dataset
import numpy as np
import json
import os
from torchvision.transforms import Resize, InterpolationMode
from pathlib import Path


class LightFormerDataset(Dataset):
    """
    Custom LightFormer Dataset.
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (list): Paths to the image directories, which should include a .json with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.img_dir = img_dir
        self.frames = []
        self.root_dir_list = []
        self.transform = transform

        # handle a list of img_dir
        for sub_path in self.img_dir:
            json_files = sorted(os.path.join(sub_path, x) for x in os.listdir(sub_path) if (x.endswith('.json')))
            for x in json_files:
                with open(x, 'r') as opened_file:
                    sample = json.load(opened_file)
                    self.frames += sample
                    self.root_dir_list += len(sample) * [Path(sub_path)]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_names = self.frames[idx]['images']
        root_dir = self.root_dir_list[idx]
        images = torch.from_numpy(np.zeros((10,3,512,960),dtype='float32'))
        for i,img_name in enumerate(img_names):
            image_path = os.path.join(root_dir, 'frames', img_name)
            image = io.imread(image_path)
            image = resize(image, (512, 960), anti_aliasing=True)
            # numpy image: H x W x C
            # torch image: C x H x W
            image = image.transpose((2, 0, 1)) / 255.0
            image = image.astype('float32')
            image = torch.from_numpy(image)
            image = self.transform(image)
            images[i] = image

        label = self.frames[idx]['label'][:4]
        label = np.array([label])
        label = label.astype('float32')
        label = torch.from_numpy(label)
        label = label.squeeze()
        sample = {
            'images': images,
            'label': label,
            'name': img_names[0],
        }

        return sample
