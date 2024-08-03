import torch.nn.functional as F


def ce_loss(preds, labels):
    return F.cross_entropy(preds, labels)
