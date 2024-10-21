import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from torchvision.transforms import Normalize
from pathlib import Path
from torchvision.transforms.v2.functional import resize
from torchvision.io import read_image
import shutil

class LightFormerDataset(Dataset):
    """
    Custom LightFormer Dataset.
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(self, directory, subdirectories, preprocessed_directory = None, preprocessed=False):
        """
        Args:
            directory (Pathlike): Path to raw image subdirectories.
            preprocessed_directory (Pathlike): Path to the preprocessed image subdirectories, in which each image is a .pt tensor.
            subdirectories: (List of Pathlikes): Relative subpaths under either directory or preprocessed_directory. Each subpath should include a 'frames' directory containing images or .pt processed image tensors and a samples.json including the structure of the samples of that subpath.
            preprocessed (bool): preprocessed=True means the preprocessing has been run already.
        """
        # Instantiation args
        self.directory = directory
        self.preprocessed_directory = preprocessed_directory
        self.subdirectories = subdirectories
        self.preprocessed = preprocessed

        # frames holds buffers (lists of image names of len 10) and labels, root_dir_list holds corresponding image subdirectory
        self.frames = []
        self.cor_subdirectories = []


        for subdir in subdirectories:
            # if preprocessed, we only need to look in preprocessed directory.
            if preprocessed:
                root = os.path.join(self.preprocessed_directory, subdir)
            else:
                root = os.path.join(self.directory, subdir)
            json_files = sorted(os.path.join(root, x) for x in os.listdir(root) if (x.endswith('.json')))
            for x in json_files:
                with open(x, 'r') as opened_file:
                    sample = json.load(opened_file)
                    self.frames += sample
                    self.cor_subdirectories += len(sample) * [Path(subdir)]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):

        # make sure index i is an Python list
        if torch.is_tensor(i):
            i = i.tolist()

        # populate images
        image_names = self.frames[i]['images']
        subdir = self.cor_subdirectories[i]
        images = torch.from_numpy(np.zeros((10,3,512,960),dtype='float32'))
        for index, img_name in enumerate(image_names):
            image = self._get_preprocessed_image_tensor(subdirectory=subdir, image_name=img_name[:-4])
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

    def get_original_sample_from_index(self, i):

        # make sure index i is an Python list
        if torch.is_tensor(i):
            i = i.tolist()

        # populate images
        image_names = self.frames[i]['images']
        subdir = self.cor_subdirectories[i]
        images = torch.from_numpy(np.zeros((10,3,512,960),dtype='float32'))
        for index, img_name in enumerate(image_names):
            image_path=os.path.join(self.directory, subdir, img_name)+".jpg"
            image = read_image(image_path)
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

    def preprocess(self):
        for subdirectory in self.subdirectories:
            # create respective full paths
            src_subdirectory = os.path.join(self.directory, subdirectory)
            src_subdirectory_frames = os.path.join(src_subdirectory, 'frames')
            dst_subdirectory = os.path.join(self.preprocessed_directory, subdirectory, '')
            dst_subdirectory_frames = os.path.join(dst_subdirectory, 'frames')
            Path(dst_subdirectory_frames).mkdir(parents=True, exist_ok=True)

            # copy the samples.json into destination directory
            src_samples_json = os.path.join(src_subdirectory, "samples.json")
            shutil.copy(src_samples_json, dst_subdirectory)

            # process images and save them into destination directory
            imgs_names = sorted([img_name for img_name in os.listdir(src_subdirectory_frames) if img_name.endswith('.jpg')])
            for img_name in imgs_names:
                src_path = os.path.join(src_subdirectory, 'frames', img_name)
                tensor = self._read_transform_image(src_path)
                dst_path = os.path.join(dst_subdirectory_frames, img_name[:-4])+".pt"
                torch.save(tensor, dst_path)

    def _get_preprocessed_image_tensor(self, subdirectory: str, image_name: str):
        """
        Get the preprocessed image tensor.

        Args:
            subdirectory: subdirectory of image
            image_name: name of file containing image or image tensor, with no file extension
        """
        if self.preprocessed:
            tensor_path=os.path.join(self.preprocessed_directory, subdirectory, "frames", image_name)+".pt"
            tensor = torch.load(tensor_path)
            return tensor
        image_path=os.path.join(self.directory, subdirectory, "frames", image_name)+".jpg"
        tensor = self._read_transform_image(image_path)
        return tensor

    def _read_transform_image(self, image_path):
        transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        tensor = read_image(image_path)
        tensor = resize(tensor, (512, 960), antialias=True)
        tensor = tensor.to(torch.float32)
        tensor = tensor / 255.0
        tensor = transform(tensor)
        return tensor
