# encoding: utf-8
"""
based on
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels1, labels2, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [M, N]
      labels1: pytorch LongTensor, with shape [M]
      labels2: pytorch LongTensor
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    M, N = dist_mat.shape

    # shape [M, N]
    is_pos = labels1[:, None].expand(M, N).eq(labels2[:, None].expand(N, M).t())
    is_neg = labels1[:, None].expand(M, N).ne(labels2[:, None].expand(N, M).t())
    idxs_mat = torch.arange(N, device=dist_mat.device)[None].expand(M, N)
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [M, 1]
    pos_lens = is_pos.sum(dim=-1).tolist()
    pos_dists = dist_mat[is_pos].split(pos_lens)
    pos_dists_idx = idxs_mat[is_pos].split(pos_lens)
    dist_ap = []
    p_inds = []
    for dist_sub, idx_sub in zip(pos_dists, pos_dists_idx):
        val, r_idx = torch.max(dist_sub, dim=-1)
        idx_p = idx_sub[r_idx]
        dist_ap.append(val.view(-1))
        p_inds.append(idx_p.view(-1))
    dist_ap = torch.cat(dist_ap, dim=0)
    p_inds = torch.cat(p_inds, dim=0)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [M, 1]
    neg_lens = is_neg.sum(dim=-1).tolist()
    neg_dists = dist_mat[is_neg].split(neg_lens)
    neg_dists_idx = idxs_mat[is_neg].split(neg_lens)
    dist_an = []
    n_inds = []
    for dist_sub, idx_sub in zip(neg_dists, neg_dists_idx):
        val, r_idx = torch.min(dist_sub, dim=-1)
        idx_p = idx_sub[r_idx]
        dist_an.append(val.view(-1))
        n_inds.append(idx_p.view(-1))
    dist_an = torch.cat(dist_an, dim=0)
    n_inds = torch.cat(n_inds, dim=0)

    if return_inds:
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, reduction="none"):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction=reduction)

    def __call__(self, feats1, feats2, labels1, labels2, normalize_feature=False):
        if normalize_feature:
            feats1 = normalize(feats1, axis=-1)
            feats2 = normalize(feats2, axis=-1)
        dist_mat = torch.cdist(feats1[None], feats2[None]).squeeze(
            0
        )  # euclidean_dist(feats1, feats2)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels1, labels2)
        y = torch.ones_like(dist_an)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return {
            "loss_triplet": loss,
            "dist_ap": dist_ap.mean(),
            "dist_an": dist_an.mean(),
        }


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
