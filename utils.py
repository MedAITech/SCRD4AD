import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial

from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle


def global_cosine_param(a, b, param, stop_grad=False):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        weight = param[:,item].view(param.shape[0], -1)
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach() * weight,
                                            b[item].view(b[item].shape[0], -1)* weight)) 
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1) * weight,
                                            b[item].view(b[item].shape[0], -1) * weight))
    return loss



def cal_anomaly_map_param(fs_list, ft_list, out_size=224, amap_mode='mul', param=None):
    weight = param[0]
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i] * weight[i]
        ft = ft_list[i] * weight[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
            
    return anomaly_map, a_map_list



def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N



def evaluation_noseg_configB_mlp(model, dataloader,device, reduction='max'):
    model.eval()
    gt_list_sp = []
    pr_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
    
            inputs, outputs, param = model(img)

            anomaly_map, _ = cal_anomaly_map_param(inputs, outputs, img.shape[-1], amap_mode='a', param=param)
            
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt_list_sp.append(label.item())
            if reduction == 'max':
                pr_list_sp.append(np.max(anomaly_map))
            elif reduction == 'mean':
                pr_list_sp.append(np.mean(anomaly_map))

        thresh = return_best_thr(gt_list_sp, pr_list_sp)
        acc = round(accuracy_score(gt_list_sp, pr_list_sp >= thresh), 4)
        f1 = round(f1_score(gt_list_sp, pr_list_sp >= thresh), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_sp, f1, acc



