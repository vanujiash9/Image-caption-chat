import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - label_smoothing
        self.vocab_size = tgt_vocab_size

    def forward(self, pred, target):
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)

        true_dist = pred.data.clone()
        true_dist.fill_(self.label_smoothing / (self.vocab_size - 1))
        ignore = target == self.ignore_index
        non_pad_mask = ~ignore
        target = target.masked_fill(ignore, 0)

        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        pred = F.log_softmax(pred, dim=-1)

        loss = self.criterion(pred, true_dist)
        return loss / non_pad_mask.sum()
