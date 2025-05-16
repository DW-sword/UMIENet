import os

import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
import warnings
import random

warnings.filterwarnings('ignore')


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, inp='input', target='target',depth='depth', real_dir='' ,img_options=None):
        super(DataLoaderTrain, self).__init__()

        self.img_options = img_options

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, inp)))
        self.inp_filenames = [os.path.join(rgb_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, target, x) for x in inp_files if is_image_file(x)]
        if self.img_options['with_dep']:
            self.dep_filenames = [os.path.join(rgb_dir, depth, x) for x in inp_files if is_image_file(x)]
        
        real_files = sorted(os.listdir(real_dir))
        self.real_filenames = [os.path.join(real_dir, x) for x in real_files if is_image_file(x)]

        self.sizex = len(self.inp_filenames)  # get the size of target

        self.transform = A.Compose([A.Resize(height=img_options['h'], width=img_options['w'])],is_check_shapes=False)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        tar_path = self.tar_filenames[index_]
        tar_img = Image.open(tar_path).convert('RGB')
        real_path = self.real_filenames[random.randint(0,len(self.real_filenames)-1)]
        real_img = Image.open(real_path).convert('RGB')

        if self.img_options['with_dep']:
            dep_path = self.dep_filenames[index_]
            dep_img = Image.open(dep_path).convert('RGB')

        if not self.img_options['ori']:
            inp_img = np.array(inp_img)
            tar_img = np.array(tar_img)
            real_img = np.array(real_img)
            if self.img_options['with_dep']:
                dep_img = np.array(dep_img)
                inp_img = self.transform(image=inp_img)['image']
                tar_img = self.transform(image=tar_img)['image']
                dep_img = self.transform(image=dep_img)['image']
            else:
                inp_img = self.transform(image=inp_img)['image']
                tar_img = self.transform(image=tar_img)['image']
                real_img = self.transform(image=real_img)['image']
        
        if self.img_options['with_dep']:
            inp_img = F.to_tensor(inp_img)
            dep_img = F.to_tensor(dep_img)
            tar_img = F.to_tensor(tar_img)
            return inp_img, dep_img, tar_img
        else:
            inp_img = F.to_tensor(inp_img)
            tar_img = F.to_tensor(tar_img)
            real_img = F.to_tensor(real_img)
            return inp_img, 0, tar_img,real_img


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, inp='input', target='target', depth='depth', real_dir='' , img_options=None):
        super(DataLoaderVal, self).__init__()

        self.img_options = img_options

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, inp)))
        self.inp_filenames = [os.path.join(rgb_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, target, x) for x in inp_files if is_image_file(x)]
        if self.img_options['with_dep']:
            self.dep_filenames = [os.path.join(rgb_dir, depth, x) for x in inp_files if is_image_file(x)]
        
        real_files = sorted(os.listdir(real_dir))
        self.real_filenames = [os.path.join(real_dir, x) for x in real_files if is_image_file(x)]

        self.sizex = len(self.inp_filenames)  # get the size of target

        self.transform = A.Compose([A.Resize(height=img_options['h'], width=img_options['w'])],is_check_shapes=False)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        tar_path = self.tar_filenames[index_]
        tar_img = Image.open(tar_path).convert('RGB')
        real_path = self.real_filenames[random.randint(0,len(self.real_filenames)-1)]
        real_img = Image.open(real_path).convert('RGB')

        if self.img_options['with_dep']:
            dep_path = self.dep_filenames[index_]
            dep_img = Image.open(dep_path).convert('RGB')

        if not self.img_options['ori']:
            inp_img = np.array(inp_img)
            tar_img = np.array(tar_img)
            real_img = np.array(real_img)
            if self.img_options['with_dep']:
                dep_img = np.array(dep_img)
                inp_img = self.transform(image=inp_img)['image']
                tar_img = self.transform(image=tar_img)['image']
                dep_img = self.transform(image=dep_img)['image']
            else:
                inp_img = self.transform(image=inp_img)['image']
                tar_img = self.transform(image=tar_img)['image']
                real_img = self.transform(image=real_img)['image']
        
        if self.img_options['with_dep']:
            inp_img = F.to_tensor(inp_img)
            dep_img = F.to_tensor(dep_img)
            tar_img = F.to_tensor(tar_img)
            return inp_img, dep_img, tar_img
        else:
            inp_img = F.to_tensor(inp_img)
            tar_img = F.to_tensor(tar_img)
            real_img = F.to_tensor(real_img)
            return inp_img, 0, tar_img,real_img


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, inp='input', dep='depth', img_options=None):
        super(DataLoaderTest, self).__init__()

        self.img_options = img_options
        
        inp_files = sorted(os.listdir(os.path.join(rgb_dir, inp)))
        self.inp_filenames = [os.path.join(rgb_dir, inp, x) for x in inp_files if is_image_file(x)]
        if self.img_options['with_dep']:
            self.dep_filenames = [os.path.join(rgb_dir, dep, x) for x in inp_files if is_image_file(x)]
        
        self.sizex = len(self.inp_filenames)  # get the size of target

        self.transform = A.Compose([A.Resize(height=img_options['h'], width=img_options['w'])],is_check_shapes=False)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        if self.img_options['with_dep']:
            dep_path = self.dep_filenames[index_]
            dep_img = Image.open(dep_path).convert('RGB')

        filename = os.path.split(inp_path)[-1]

        # 是否裁剪
        if not self.img_options['ori']:
            inp_img = np.array(inp_img)
            if self.img_options['with_dep']:
                dep_img = np.array(dep_img)
                inp_img = self.transform(image=inp_img)['image']
                dep_img = self.transform(image=dep_img)['image']
            else:
                inp_img = self.transform(image=inp_img)['image']
        
        if self.img_options['with_dep']:
            inp_img = F.to_tensor(inp_img)
            dep_img = F.to_tensor(dep_img)
            return inp_img, dep_img, filename
        else:
            inp_img = F.to_tensor(inp_img)
            return inp_img, 0, filename