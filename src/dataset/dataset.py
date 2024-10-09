import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from torchvision.transforms import Normalize
from pathlib import Path
from torchvision.transforms.v2.functional import resize
from torchvision.io import read_image

class LightFormerDataset(Dataset):
    """
    Custom LightFormer Dataset.
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(self, directory):
        """
        Args:
            directory (list): Paths to the image directories, which should include a .json with annotations.
        """

        # frames holds image buffer names and label, root_dir_list holds image directory
        self.frames = []
        self.root_dir_list = []
        self.transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # handle a list of img_dir
        for sub_path in directory:
            json_files = sorted(os.path.join(sub_path, x) for x in os.listdir(sub_path) if (x.endswith('.json')))
            for x in json_files:
                with open(x, 'r') as opened_file:
                    sample = json.load(opened_file)
                    self.frames += sample
                    self.root_dir_list += len(sample) * [Path(sub_path)]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):

        # make sure index i is an Python list
        if torch.is_tensor(i):
            i = i.tolist()

        # get data from frames array
        image_names = self.frames[i]['images']
        root_directories = self.root_dir_list[i]

        # populate images
        images = torch.from_numpy(np.zeros((10,3,512,960),dtype='float32'))
        for index, img_name in enumerate(image_names):
            image_path = os.path.join(root_directories, 'frames', img_name)
            image = read_image(image_path)
            image = resize(image, (512, 960), antialias=True)
            image = image.to(torch.float32)
            image = image / 255.0
            image = self.transform(image)
            images[index] = image

        # populate label
        label = self.frames[i]['label']
        label = torch.tensor(label)
        label = label.to(torch.float32)

        # populate name
        name = image_names[0]

        # return dictionary
        sample = {
            'images': images,
            'label': label,
            'name': name,
        }

        return sample
