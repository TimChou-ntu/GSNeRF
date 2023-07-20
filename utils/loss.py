### from semray

import torch
import torch.nn as nn


class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys = keys

    def __call__(self, data_pr, data_gt, **kwargs):
        pass

class SemanticLoss(Loss):
    def __init__(self, nb_class, ignore_label, weight=None):
        super().__init__(['loss_semantic'])
        self.nb_class = nb_class
        self.ignore_label = ignore_label
        self.weight = weight

    def __call__(self, data_pr, data_gt, **kwargs):
        def compute_loss(label_pr, label_gt):
            label_pr = label_pr.reshape(-1, self.nb_class)
            label_gt = label_gt.reshape(-1).long()
            valid_mask = (label_gt != self.ignore_label)
            label_pr = label_pr[valid_mask]
            label_gt = label_gt[valid_mask]
            if self.weight != None:
                return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean', weight=self.weight).unsqueeze(0)
            else:
                return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)

        loss = compute_loss(data_pr, data_gt)

        return loss