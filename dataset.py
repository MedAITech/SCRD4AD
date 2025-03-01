from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import cv2
from models.noise import Simplex_CLASS
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull


torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms



        
        
class RSNADataset_train(torch.utils.data.Dataset):
    def __init__(self, root, json_dir):
        self.img_path = root
        self.json_dir = json_dir
        self.img_size = (256, 256)
        self.simplexNoise = Simplex_CLASS()
        self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            transforms.Resize(self.img_size)
                        ]
                    )
        # load dataset
        self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        with open(self.json_dir, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        train_data = json_data['train']
        train_0_datalist = train_data['0']
        self.train_img_names, self.train_labels = [], []
        for img_name in train_0_datalist:
            self.train_img_names.append(img_name)
            self.train_labels.append(0)
        return self.train_img_names, self.train_labels

    def __len__(self):
        return len(self.train_img_names)

    def __getitem__(self, idx):
        img_name = self.train_img_names[idx]
        img = cv2.imread(self.img_path + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img_normal = self.transform(img)
        img_normal = img_normal.float()
        ## simplex_noise
        size = 256
        
        h_noise = np.random.randint(10, int(size//8))
        w_noise = np.random.randint(10, int(size//8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256,256,3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0)
        # init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.6 * simplex_noise.transpose(1,2,0)
        img_noise = img + init_zero
        img_noise = self.transform(img_noise)
        img_noise = img_noise.float()
        return img_normal, img_noise


class RSNADataset_train_lambda(torch.utils.data.Dataset):
    def __init__(self, root, json_dir, mylambda=0.2):
        self.img_path = root
        self.json_dir = json_dir
        self.img_size = (256, 256)
        self.mylambda = mylambda
        self.simplexNoise = Simplex_CLASS()
        self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            transforms.Resize(self.img_size)
                        ]
                    )
        # load dataset
        self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        with open(self.json_dir, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        train_data = json_data['train']
        train_0_datalist = train_data['0']
        self.train_img_names, self.train_labels = [], []
        for img_name in train_0_datalist:
            self.train_img_names.append(img_name)
            self.train_labels.append(0)
        return self.train_img_names, self.train_labels

    def __len__(self):
        return len(self.train_img_names)

    def __getitem__(self, idx):
        img_name = self.train_img_names[idx]
        img = cv2.imread(self.img_path + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img_normal = self.transform(img)
        img_normal = img_normal.float()
        ## simplex_noise
        size = 256
        
        h_noise = np.random.randint(10, int(size//8))
        w_noise = np.random.randint(10, int(size//8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256,256,3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = self.mylambda * simplex_noise.transpose(1,2,0)
        img_noise = img + init_zero
        img_noise = self.transform(img_noise)
        img_noise = img_noise.float()
        return img_normal, img_noise



class RSNADataset_test(torch.utils.data.Dataset):
    def __init__(self, root, json_dir):
        self.img_path = root
        self.json_dir = json_dir
        self.img_size = (256, 256)
        self.simplexNoise = Simplex_CLASS()
        self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            transforms.Resize(self.img_size)
                        ]
                    )
        # load dataset
        self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        with open(self.json_dir, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        val_data = json_data['test']
        val_0_datalist = val_data['0']
        val_1_datalist = val_data['1']
        self.val_img_names, self.val_labels = [], []
        for img_name in val_0_datalist:
            self.val_img_names.append(img_name)
            self.val_labels.append(0)
        for img_name in val_1_datalist:
            self.val_img_names.append(img_name)
            self.val_labels.append(1)

    def __len__(self):
        return len(self.val_img_names)

    def __getitem__(self, idx):
        img_name = self.val_img_names[idx]
        img = cv2.imread(self.img_path + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img_normal = self.transform(img)
        label = self.val_labels[idx]
        img_normal = img_normal.float()
        return img_normal, label    
        # return img_normal, label, img_name   
    
    
