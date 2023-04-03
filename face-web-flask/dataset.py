import torch
from glob import glob
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class createDataset(data.Dataset):
    def __init__(self, hp):
        self.hp = hp
        self.root_dir = hp.root_dir
        self.test_photo_paths = sorted(glob(os.path.join(self.root_dir, '*')))
        self.test_transform = get_transform()

    def __getitem__(self, item):
        sample = {}
        positive_path = self.test_photo_paths[item]
        positive_img = Image.open(positive_path).convert('RGB')
        positive_img = self.test_transform(positive_img)
        # split_path 
        # windows
        positive_path = positive_path.split('\\')[-1]
        # linux
        positive_path = positive_path.split('/')[-1]
        sample = {'positive_img': positive_img, 'positive_path': positive_path,}
        return sample

    def __len__(self):
        return len(self.test_photo_paths)

def get_dataloader(hp):
    dataset_Test = createDataset(hp)
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False)
    return dataloader_Test


def get_transform():
    transform_list = []
    transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)
