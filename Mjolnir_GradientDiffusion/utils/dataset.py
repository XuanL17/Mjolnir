import glob
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from util import cross_entropy_for_onehot, label_to_onehot
from utils.trainNetworkHelper import LeNet
from utils.utils import weight_init, cal_sensitivity, Gaussian_Simple, channel_deal_3

device = 'cuda'


class MnistDataset(Dataset):
    def __init__(self, dst, model_path='model/lenet.pt', device='cuda', dp_clip=1, lr=0.001, lr_decay=0.1, epsilon=1,
                 delta=0.01):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = dst
        self.num_samples = len(dst)
        self.model = LeNet()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device)
        self.model.train()
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_decay)
        self.loss_client = 0
        self.sensitivity = cal_sensitivity(lr, dp_clip, 1)
        self.noise_scale = Gaussian_Simple(epsilon, delta)
        self.loss_fn = cross_entropy_for_onehot

    def __getitem__(self, idx):
        fname = self.dataset[idx]
        img = fname[0].to(self.device)
        img = img.view(1, *img.size())
        label = torch.Tensor([fname[1]]).long().to(self.device)
        label = label.view(1, )
        label = label_to_onehot(label, num_classes=10)
        self.model.zero_grad()
        log_probs = self.model(img)
        loss = self.loss_fn(log_probs, label)

        grad = torch.autograd.grad(loss, self.model.parameters())
        or_grad = list((_.detach().clone() for _ in grad))
        # dp_grad = list((_.detach().clone() for _ in grad))
        #  加噪
        # for k in range(len(dp_grad)):
        #     dp_grad[k] += torch.from_numpy(np.random.normal(loc=0, scale=self.sensitivity * self.noise_scale,
        #                                                     size=dp_grad[k].shape)).to(self.device)
        te = 0
        for e in or_grad:
            tt = np.array(e.cpu())
            tt = tt.flatten()
            tt = tt.tolist()
            or_grad[te] = tt
            te = te + 1
        channel1 = []
        for layer in range(8):
            channel1.extend(or_grad[layer])
        channel1 = channel_deal_3(channel1)
        channel1 = torch.Tensor(channel1).float().to(self.device)
        # channel1 = channel1.view(-1)
        channel1 = torch.unsqueeze(channel1, 0)
        return channel1

    def __len__(self):
        return self.num_samples


class TrainDataset(Dataset):
    def __init__(self, noise_dir, gt_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.noise_list = glob.glob(os.path.join(noise_dir, '*'))
        self.gt_list = glob.glob(os.path.join(gt_dir, '*'))

    def __getitem__(self, idx):
        noise_path = self.noise_list[idx]
        gt_path = self.gt_list[idx]
        noise_data = np.load(noise_path)
        gt_data = np.load(gt_path)
        # fdata = np.expand_dims(fdata, axis=0)
        noise_data = torch.from_numpy(noise_data)
        gt_data = torch.from_numpy(gt_data)
        return noise_data, gt_data

    def __len__(self):
        return len(self.noise_list)


class MyDataset(Dataset):
    def __init__(self, dst, is_resize=True):
        if is_resize:
            self.transform = transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = dst
        self.num_samples = len(dst)

    def __getitem__(self, idx):
        fname = self.dataset[idx]
        img = self.transform(fname[0]).float().to(device)
        # img = img.view(1, *img.size())
        label = torch.Tensor([fname[1]]).long().to(device)
        label = label.view(1, )

        return img, label

    def __len__(self):
        return self.num_samples
