import torch
from glob import glob
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
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

        self.root_dir = os.path.join(hp.root_dir, 'Dataset')
        # Windows环境更改为以下路径
        self.test_photo_paths = sorted(glob(os.path.join(self.root_dir, 'draw100', 'image', '*')))
        self.test_seq_sketch_dirs = sorted(glob(os.path.join(self.root_dir, 'draw100', 'sketch-seq', '*')))
        self.test_transform = get_transform('Test')

    def __getitem__(self, item):
        sample = {}

        sketch_path = self.test_seq_sketch_dirs[item]
        positive_path = self.test_photo_paths[item]

        positive_img = Image.open(positive_path).convert('RGB')
        sketch_seq_paths = sorted(glob(os.path.join(sketch_path, '*')))

        sketch_seq = []
        pixel_ratio = []
        for path in sketch_seq_paths:
            # 图像是白底黑画的，要转换成黑底白画
            sketch_img = 255 - np.array(Image.open(path).convert('L'))
            pixel_ratio.append(sketch_img.sum())
            sketch_img = Image.fromarray(sketch_img).convert('RGB')
            sketch_seq.append(self.test_transform(sketch_img))
        sketch_seq = torch.stack(sketch_seq)
        pixel_ratio = pixel_ratio / pixel_ratio[-1]
        # sketch_img = self.test_transform(sketch_img)
        positive_img = self.test_transform(positive_img)

        sample = {'sketch_seq': sketch_seq, 'positive_img': positive_img, 
                    'sketch_seq_paths': sketch_seq_paths, 'positive_path': positive_path, 'pixel_ratio': pixel_ratio}
        return sample

    def __len__(self):
        return len(self.test_seq_sketch_dirs)

def get_dataloader(hp):
    dataset_Test = createDataset(hp)
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False, num_workers=int(hp.nThreads))
    return dataloader_Test


def get_transform(type):
    transform_list = []
    transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)
