
import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from models.rd4ad_mlp import RdadAtten
from dataset import RSNADataset_test, RSNADataset_train, RSNADataset_train_lambda
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_noseg_configB_mlp
from utils import global_cosine_param

from torch.nn import functional as F

import warnings
import logging

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(img_path, json_path):
    setup_seed(111)

    total_iters = 40000
    batch_size = 16


    train_data = RSNADataset_train_lambda(img_path, json_path, mylambda=0.2)
    test_data = RSNADataset_test(img_path, json_path)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    
    model = RdadAtten()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5,0.999))
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data)))

    it = 0
    auroc_sp_best = 0
    model = model.to(device=device)
    
    for epoch in range(total_iters // len(train_dataloader) + 1):
        model.train()
        loss_list = []
        for img, img_noise in train_dataloader:
            img = img.to(device)
            img_noise = img_noise.to(device)
            inputs, outputs, param = model(img)
            inputs_noise, outputs_noise, _ = model(img_noise)
            
            loss1 = global_cosine_param(inputs, outputs, param)
            loss2 = global_cosine_param(inputs_noise, outputs, param)
            loss = loss1 / loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if (it + 1) % 250 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []
                auroc, f1, acc = evaluation_noseg_configB_mlp(model, test_dataloader, device, reduction='mean')
                print_fn('AUROC:{:.4f}, F1:{:.4f}, ACC:{:.4f}'.format(auroc, f1, acc))
                if auroc >= auroc_sp_best:
                    auroc_sp_best = auroc
                    torch.save(model, f'./best_models/auc_{auroc}_f1_{round(f1, 4)}_acc_{round(acc, 4)}.pth')
            it += 1
            if it == total_iters:
                break

    return auroc, auroc_sp_best


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='rsna_noise_atten_lambda0.1')
    parser.add_argument('--gpu', default='2', type=str, help='GPU id to use.')
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(args.img_path, args.json_path)
