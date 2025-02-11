# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init2(m):
    # if hasattr(m, "weight"):
    #     m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

class TextCNN(nn.Module):
    def __init__(self, vocab_size=12620, n_class=100, embed_dim=100, num_filters=256):
        super(TextCNN, self).__init__()


        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (f, embed_dim)) for f in [2, 3, 4]])
        self.fc = nn.Linear(in_features=num_filters*3, out_features=n_class)


    def forward(self, x: torch.Tensor):
        pooled = []
        for conv in self.convs:
            out = conv(x)
            out = F.relu(out)
            out = F.max_pool2d(out, (out.shape[-2], 1))  # [32, 256, 1, 1]
            out = out.squeeze()  # [32, 256]
            pooled.append(out)

        x = torch.cat(pooled, dim=-1)  # [32, 768]
        x = self.fc(x)
        return x

class Textencoder(nn.Module):
    def __init__(self, n_class=2, embed_dim=100, num_filters=256):

        super(Textencoder, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (f, embed_dim)) for f in [2, 3, 4]])




    def forward(self, x: torch.Tensor):
        pooled = []

        for conv in self.convs:
            out = conv(x)
            out = F.relu(out)
            out = F.max_pool2d(out, (out.shape[-2], 1))  # [32, 256, 1, 1]
            out = out.squeeze()  # [32, 256]
            pooled.append(out)
        x = torch.cat(pooled, dim=-1)  # [32, 768]
        return x

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class Imageencoder(nn.Module):
    def __init__(self):
        super(Imageencoder, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(588, 768)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        out=out.squeeze()
        return out


class ourmodel(nn.Module):
    def __init__(self):

        super(ourmodel, self).__init__()
        self.textencoder=Textencoder()
        self.imageencoder=Imageencoder()
        self.fc=nn.Linear(768*2,100)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1=self.textencoder(x1)
        x2=self.imageencoder(x2)
        x=torch.cat((x1,x2))
        out=self.fc(x)

        return out

