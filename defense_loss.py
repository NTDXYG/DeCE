import math

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss

class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.5):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class In_trust_Loss(nn.Module):
    def __init__(self, num_classes, delta=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.delta = delta
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.alpha = 0.2
        self.beta = 0.8

    def forward(self, logits, labels):
        ce = self.cross_entropy(logits,labels)
        #Loss In_trust
        active_logits = logits.view(-1,self.num_classes)
        active_labels = labels.view(-1)
        pred = F.softmax(active_logits, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(active_labels,self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        dce = (-1*torch.sum(pred * torch.log(pred*self.delta + label_one_hot*(1-self.delta)), dim=1))
        # Loss
        loss = self.alpha * ce + self.beta * dce.mean()
        # loss = dce.mean()
        return loss

class DeceptionCrossEntropyLoss(_WeightedLoss):
    def __init__(self, num_classes, smoothing=0.05, delta=0.98):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.alpha = 1
        self.delta = delta

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets, cur_epoch):
        print(inputs.shape)
        print(inputs)

        targets = DeceptionCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        print(targets.shape)
        print(targets)
        self.alpha = self.alpha * (math.pow(self.delta, cur_epoch))
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        new_pred = self.alpha * pred + (1-self.alpha) * targets
        ce = -(targets * torch.log(new_pred)).sum(-1).mean()

        return ce
