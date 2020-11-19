#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy

class CircleCore(nn.Module):
    def __init__(self, m: float = 0.35, s: float = 256) -> None:
        super(CircleCore, self).__init__()
        self.s, self.m = s, m
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        cosine = input
        alpha_p = F.relu(1 + self.m - cosine).detach()
        margin_p = 1 - self.m
        alpha_n = F.relu(cosine + self.m).detach()
        margin_n = self.m

        sp_y = alpha_p * (cosine - margin_p)
        sp_j = alpha_n * (cosine - margin_n)

        one_hot = torch.zeros(cosine.size()).to(label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * sp_y + ((1.0 - one_hot) * sp_j)
        output *= self.s
        loss = self.criteria(output, label)

        prec1, _    = accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        return loss, prec1



class LossFunction(nn.Module):
    def __init__(self, nOut: int, nClasses: int, margin: float = 0.35, scale: float = 256, **kwargs) -> None:
        super(LossFunction, self).__init__()
        self.circle_core = CircleCore(m=margin, s=scale)    
        self.test_normalize = True
        self.weight = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        # logits = self.classifier_linear(input)
        logits = nn.functional.linear(nn.functional.normalize(input, p=2, dim=1, eps=1e-12),
                                      nn.functional.normalize(self.weight, p=2, dim=1, eps=1e-12))
        return self.circle_core(logits, label)

if __name__ == '__main__':
    loss_helper = LossFunction(nOut=512, nClasses=1991, margin=0.35, scale=256)
    feat = torch.randn(10, 512)
    lbl = torch.randint(high=1990, size=(10,))
    circle_loss_1 = loss_helper(feat, lbl)
    print(circle_loss_1)
