import torch
import torch.nn as nn
from torch.autograd import Function
from pops.utils import comm
import functools
import torch.nn.functional as F
from pops.config import configurable


def _lb_feat_update(lookup_table, labeled_feats, labeled_ids, labeled_mms, do_norm):
    for indx, label in enumerate(labeled_ids):
        if labeled_mms[indx] < 1:
            lookup_table[label] = (
                labeled_mms[indx] * lookup_table[label]
                + (1 - labeled_mms[indx]) * labeled_feats[indx]
            )
            if do_norm:
                lookup_table[label] /= lookup_table[label].norm()


# set the weight of proto in the lookup table as 0.5 by default
def _lb_feat_update_avg(lookup_table, labeled_feats, labeled_ids, labeled_mms, do_norm):
    id_set = list(set(labeled_ids.tolist()))
    for pid in id_set:
        id_mask = labeled_ids == pid
        p_feats = labeled_feats[id_mask].view(-1, labeled_feats.shape[-1])
        proto_feat = lookup_table[pid][None]
        p_mms = labeled_mms[id_mask].unsqueeze(1)
        proto_mm = p_mms.new_tensor([[0.5]])
        fuse_feat = torch.cat([proto_mm * proto_feat, p_mms * p_feats], dim=0).sum(
            dim=0
        ) / (proto_mm.sum() + p_mms.sum())
        lookup_table[pid] = fuse_feat
        if do_norm:
            lookup_table[pid] /= lookup_table[pid].norm()


def _ulb_feat_update(queue, tail, unlabeled_feats, length):
    num_feats = unlabeled_feats.shape[0]
    tail_v = tail[0]
    if num_feats <= queue.shape[0] - tail_v:
        queue[tail_v : tail_v + num_feats, :length] = unlabeled_feats[:, :length]
    else:
        num_left = num_feats - (queue.shape[0] - tail_v)
        queue[tail_v:, :length] = unlabeled_feats[:-num_left, :length]
        queue[:num_left, :length] = unlabeled_feats[-num_left:, :length]


def _update_table_sync(lookup_table, pid_labels, features, momentums, feat_update_func):
    # Update lookup table, but not by standard backpropagation with gradients
    labeled_mask = pid_labels > -1
    labeled_feats = features[labeled_mask]
    labeled_ids = pid_labels[labeled_mask]
    labeled_mms = momentums[labeled_mask]
    labeled_tuple = (labeled_feats, labeled_ids, labeled_mms)
    comm.synchronize()
    all_labeled_tuples = comm.all_gather(labeled_tuple)
    all_labeled_feats = []
    all_labeled_ids = []
    all_labeled_mms = []
    for feats, ids, mms in all_labeled_tuples:
        if feats.shape[0] == 0:
            continue
        all_labeled_feats.append(feats.to(labeled_feats.device))
        all_labeled_ids.append(ids.to(labeled_ids.device))
        all_labeled_mms.append(mms.to(labeled_mms.device))
    if len(all_labeled_feats) > 0:
        cat_labeled_feats = torch.cat(all_labeled_feats, dim=0)
        cat_labeled_ids = torch.cat(all_labeled_ids, dim=0)
        cat_labeled_mms = torch.cat(all_labeled_mms, dim=0)
        feat_update_func(
            lookup_table, cat_labeled_feats, cat_labeled_ids, cat_labeled_mms
        )


def _update_table_usync(
    lookup_table, pid_labels, features, momentums, feat_update_func
):
    # Update lookup table, but not by standard backpropagation with gradients
    labeled_mask = pid_labels > -1
    labeled_feats = features[labeled_mask]
    labeled_ids = pid_labels[labeled_mask]
    labeled_mms = momentums[labeled_mask]
    feat_update_func(lookup_table, labeled_feats, labeled_ids, labeled_mms)


def _update_queue_sync(queue, tail, pid_labels, features, uplen):
    # Update circular queue, but not by standard backpropagation with gradients
    unlabeled_mask = pid_labels == -1
    unlabeled_feats = features[unlabeled_mask]
    comm.synchronize()
    all_unlabeled_feats = comm.all_gather(unlabeled_feats)
    all_unlabeled_feats = [
        feats.to(unlabeled_feats.device)
        for feats in all_unlabeled_feats
        if feats.shape[0] > 0
    ]
    if len(all_unlabeled_feats) > 0:
        all_unlabeled_feats = torch.cat(all_unlabeled_feats, dim=0)
        _ulb_feat_update(queue, tail, all_unlabeled_feats, uplen)
        tail[0] = (tail[0] + all_unlabeled_feats.shape[0]) % queue.shape[0]


def _update_queue_usync(queue, tail, pid_labels, features, uplen):
    # Update circular queue, but not by standard backpropagation with gradients
    unlabeled_mask = pid_labels == -1
    unlabeled_feats = features[unlabeled_mask]
    if unlabeled_feats.shape[0] > 0:
        _ulb_feat_update(queue, tail, unlabeled_feats, uplen)
        tail = (tail + unlabeled_feats.shape[0]) % queue.shape[0]


class LabeledMatching(Function):
    @staticmethod
    def forward(ctx, features, pid_labels, lookup_table, momentum):
        # The lookup_table can't be saved with ctx.save_for_backward(), as we would
        # modify the variable which has the same memory address in backward()
        ctx.save_for_backward(features, pid_labels)
        ctx.lookup_table = lookup_table
        ctx.momentum = momentum

        scores = features.mm(lookup_table.t())
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_sync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update, do_norm=False),
        )
        return grad_feats, None, None, None


class LabeledMatchingUsync(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_usync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update, do_norm=False),
        )
        return grad_feats, None, None, None


class LabeledMatchingNorm(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_sync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update, do_norm=True),
        )
        return grad_feats, None, None, None


class LabeledMatchingNormUsync(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_usync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update, do_norm=True),
        )
        return grad_feats, None, None, None


class LabeledMatchingLayer(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, num_persons=5532, feat_len=256, sync=True):
        """
        Args:
            num_persons (int): Number of labeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(LabeledMatchingLayer, self).__init__()
        self.register_buffer("lookup_table", torch.zeros(num_persons, feat_len))
        self.sync = sync

    def _scores_sync(self, features, pids, mms):
        return LabeledMatching.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        return LabeledMatchingUsync.apply(features, pids, self.lookup_table, mms)

    @property
    def weight(self):
        return self.lookup_table

    def forward(self, features, pid_labels, momentums=None):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely the similarities
                                             between proposals and labeled persons.
        """
        if momentums is None:
                n_feats = features.shape[0]
                momentums = features.new_zeros(n_feats) + 0.5
        if self.sync:
                scores = self._scores_sync(features, pid_labels, momentums)
        else:
                scores = self._scores_usync(features, pid_labels, momentums)

        return scores
    def scores(self, features):
        return features.mm(self.lookup_table.t())


class LabeledMatchingLayerNorm(LabeledMatchingLayer):
    """
    Labeled matching of OIM loss function.
    """

    def _scores_sync(self, features, pids, mms):
        return LabeledMatchingNorm.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        return LabeledMatchingNormUsync.apply(features, pids, self.lookup_table, mms)


class LabeledMatchingAvg(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_sync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update_avg, do_norm=False),
        )
        return grad_feats, None, None, None


class LabeledMatchingAvgUsync(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_usync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update_avg, do_norm=False),
        )
        return grad_feats, None, None, None


class LabeledMatchingLayerAvg(LabeledMatchingLayer):
    def _scores_sync(self, features, pids, mms):
        return LabeledMatchingAvg.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        return LabeledMatchingAvgUsync.apply(features, pids, self.lookup_table, mms)


class LabeledMatchingNormAvg(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_sync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update_avg, do_norm=True),
        )
        return grad_feats, None, None, None


class LabeledMatchingNormAvgUsync(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_usync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update_avg, do_norm=True),
        )
        return grad_feats, None, None, None


class LabeledMatchingLayerNormAvg(LabeledMatchingLayerAvg):
    def _scores_sync(self, features, pids, mms):
        return LabeledMatchingNormAvg.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        return LabeledMatchingNormAvgUsync.apply(features, pids, self.lookup_table, mms)


class UnlabeledMatching(Function):
    @staticmethod
    def forward(ctx, features, pid_labels, queue, tail):
        # The queue/tail can't be saved with ctx.save_for_backward(), as we would
        # modify the variable which has the same memory address in backward()
        ctx.save_for_backward(features, pid_labels)
        ctx.queue = queue
        ctx.tail = tail

        scores = features.mm(queue.t())
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)
        # print(tail)
        _update_queue_sync(queue, tail, pid_labels, features, 64)
        return grad_feats, None, None, None


class UnlabeledMatchingUsync(UnlabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        _update_queue_usync(queue, tail, pid_labels, features, 64)
        return grad_feats, None, None, None


class UnlabeledMatchingLayer(nn.Module):
    """
    Unlabeled matching of OIM loss function.
    """

    def __init__(self, queue_size=5000, feat_len=256, sync=True):
        """
        Args:
            queue_size (int): Size of the queue saving the features of unlabeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(UnlabeledMatchingLayer, self).__init__()
        self.register_buffer("queue", torch.zeros(queue_size, feat_len))
        self.register_buffer("tail", torch.tensor([0]))
        # self.tail = 0
        self.sync = sync

    def _scores_sync(self, features, pids):
        return UnlabeledMatching.apply(features, pids, self.queue, self.tail)

    def _scores_usync(self, features, pids):
        return UnlabeledMatchingUsync.apply(features, pids, self.queue, self.tail)

    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, queue_size]): Unlabeled matching scores, namely the similarities
                                            between proposals and unlabeled persons.
        """
        if self.sync:
                scores = self._scores_sync(features, pid_labels)
        else:
                scores = self._scores_usync(features, pid_labels)
        return scores
    def scores(self, features):
        return features.mm(self.queue.t())


class UnlabeledMatchingFull(UnlabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        _update_queue_sync(queue, tail, pid_labels, features, features.shape[1])
        return grad_feats, None, None, None


class UnlabeledMatchingFullUsync(UnlabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        _update_queue_usync(queue, tail, pid_labels, features, features.shape[1])
        return grad_feats, None, None, None

    def _scores_sync(self, features, pids):
        return UnlabeledMatchingFull.apply(features, pids, self.queue, self.tail)

    def _scores_usync(self, features, pids):
        return UnlabeledMatchingFullUsync.apply(features, pids, self.queue, self.tail)


class UnlabeledMatchingFullLayer(UnlabeledMatchingLayer):
    def _scores_sync(self, features, pids):
        return UnlabeledMatchingFull.apply(features, pids, self.queue, self.tail)

    def _scores_usync(self, features, pids):
        return UnlabeledMatchingFullUsync.apply(features, pids, self.queue, self.tail)


layers_map = {
    "lb": LabeledMatchingLayer,
    "lb_norm": LabeledMatchingLayerNorm,
    "lb_norm_avg": LabeledMatchingLayerNormAvg,
    "lb_avg": LabeledMatchingLayerAvg,
    "ulb_part": UnlabeledMatchingLayer,
    "ulb_full": UnlabeledMatchingFullLayer,
}
"""
OIM required config:
LOSS:
    OIM:
        LB_LAYER: str in 'layers_map'
        ULB_LAYER: str in 'layers_map'
        LB_FACTOR: float
        ULB_FACTOR: float
        NUM_LB: int
        LEN_ULB: int
        FEAT_DIM: int
        NORM_FEAT: boolean
        SYNC_MEMORY: bolean
        USE_FOCAL: boolean
        FOCAL_ALPHA: float
        FOCAL_GAMMA: float
"""


class IncOIMLoss(nn.Module):
    @configurable
    def __init__(
        self,
        lb_layer,
        ulb_layer,
        lb_factor,
        ulb_factor,
        num_lb,
        num_ulb,
        feat_len,
        do_normalize=True,
        sync=True,
        use_focal=True,
        focal_alpha=1,
        focal_gamma=2,
        loss_weight=1.0,
    ):
        super().__init__()
        # for compatible with plain oim
        self.lb_layer = layers_map[lb_layer](num_lb[0], feat_len,sync)
        self.lb_layers = nn.ModuleList([layers_map[lb_layer](n, feat_len, sync) for n in num_lb]) 
        
        if len(num_ulb) == 0:
            self.ulb_layers = None
        else:
            # for compatible with plain oim
            self.ulb_layer =layers_map[ulb_layer](num_ulb[0], feat_len, sync)
            self.ulb_layers =nn.ModuleList([layers_map[ulb_layer](n, feat_len, sync) for n in num_ulb])
        self.lb_factor = lb_factor
        self.ulb_factor = ulb_factor
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.do_normalize = do_normalize
        self.loss_weight = loss_weight
        self.id_offset=0 if len(num_lb)==1 else sum(num_lb[:-1])
        #NOTE pre-sync to fix sync for multi-gpu small batch (more pure-ulb gpu-batches)
        self.sync=sync

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg.PERSON_SEARCH.REID.LOSS
        assert hasattr(loss_cfg, "INC_OIM")
        oim_cfg = loss_cfg.INC_OIM
        ret = {
            "lb_layer": oim_cfg.LB_LAYER,
            "ulb_layer": oim_cfg.ULB_LAYER,
            "lb_factor": oim_cfg.LB_FACTOR,
            "ulb_factor": oim_cfg.ULB_FACTOR,
            "num_lb": oim_cfg.NUM_LB,
            "num_ulb": oim_cfg.LEN_ULB,
            "feat_len": oim_cfg.FEAT_DIM,
            "do_normalize": oim_cfg.NORM_FEAT,
            "sync": oim_cfg.SYNC_MEMORY,
            "loss_weight": oim_cfg.LOSS_WEIGHT,
        }
        if oim_cfg.USE_FOCAL:
            ret["use_focal"] = True
            ret["focal_alpha"] = oim_cfg.FOCAL_ALPHA
            ret["focal_gamma"] = oim_cfg.FOCAL_GAMMA
        else:
            ret["use_focal"] = False
        return ret

    def forward(self, pfeats, pids, lb_mms=None):
        # NOTE assume that training is conducted with only samples of the latest task 
        pos_mask = pids > -2
        pfeats = pfeats[pos_mask]
        pids = pids[pos_mask]
        if pfeats.shape[0] == 0:
            loss_oim = pfeats.sum()*0.0 #pfeats.new_tensor([0]) , to avoid backward failure when only oim is used for training 
        else:
            if self.do_normalize:
                pfeats = F.normalize(pfeats, dim=-1)
            lb_matching_scores =[lb_layer.scores(pfeats) * self.lb_factor for lb_layer in self.lb_layers[:-1]]+[self.lb_layers[-1](pfeats, pids, lb_mms)* self.lb_factor ]
            lb_matching_scores=torch.cat(lb_matching_scores,dim=1)
            if self.ulb_layers is not None:
                ulb_matching_scores =[ulb_layer.scores(pfeats) * self.ulb_factor for ulb_layer in self.ulb_layers[:-1]] +[self.ulb_layers[-1](pfeats, pids) * self.ulb_factor]
                ulb_matching_scores=torch.cat(ulb_matching_scores,dim=1)
                matching_scores = torch.cat(
                    (lb_matching_scores, ulb_matching_scores), dim=1
                )
            else:
                matching_scores = lb_matching_scores
            pid_labels = pids.clone()
            pid_labels[pid_labels>-1]+=self.id_offset
            pid_labels[pid_labels == -2] = -1
            n_lb_feats = (pid_labels > -1).sum()
            if n_lb_feats == 0:
                loss_oim = matching_scores.sum() * 0.0 # lb_matching_scores.new_tensor([0])
            else:
                if self.use_focal:
                    p_i = F.softmax(matching_scores, dim=1)
                    focal_p_i = (
                        self.focal_alpha * (1 - p_i) ** self.focal_gamma * p_i.log()
                    )
                    loss_oim = F.nll_loss(
                        focal_p_i, pid_labels, reduction="none", ignore_index=-1
                    )
                else:
                    loss_oim = F.cross_entropy(
                        matching_scores, pid_labels, reduction="none", ignore_index=-1
                    )
        comm.synchronize()
        all_num_lb = comm.all_gather(n_lb_feats)
        num_lb = sum([num.to("cpu") for num in all_num_lb])
        num_lb = torch.clamp(num_lb / comm.get_world_size(), min=1).item()
        loss_val = loss_oim.sum() / num_lb
        return {"loss_oim": loss_val * self.loss_weight}

