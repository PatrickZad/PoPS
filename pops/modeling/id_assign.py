from typing import Iterable
import torch
from abc import ABCMeta, abstractmethod
import torchvision
from pops.config.config import configurable


class IdAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def __call__(self, det_bboxes, det_scores, gt_bboxes, gt_labels, **kws):
        """Assign boxes to either a ground truth boxes or a negative boxes."""


class IouIDAssigner(IdAssigner):
    @configurable()
    def __init__(self, iou_t=0.7, score_t=0.5) -> None:
        self.iou_t = iou_t
        self.score_t = score_t

    @classmethod
    def from_config(cls, ids_cfg):
        return {"iou_t": ids_cfg.POS_IOU, "score_t": ids_cfg.POS_SCORE}

    def __call__(self, det_bboxes, det_scores, gt_bboxes, gt_labels, **kws):
        """
        boxes: xyxy
        """
        if isinstance(det_bboxes, torch.Tensor):
            bs, ns = det_bboxes.shape[:2]
            output_ids = (
                torch.zeros(bs, ns, dtype=torch.long, device=det_bboxes[0].device) - 2
            )
        else:
            assert isinstance(det_bboxes, Iterable)
            bs = len(det_bboxes)
            output_ids = [
                torch.zeros(bi.shape[0], dtype=torch.long, device=det_bboxes[0].device)
                - 2
                for bi in det_bboxes
            ]
        for bi, gt_bboxes_i in enumerate(gt_bboxes):
            pred_bboxes = det_bboxes[bi]  # np x 4, x-y-x-y
            pairwise_ious = torchvision.ops.box_iou(pred_bboxes, gt_bboxes_i)  # np x ng
            pairwise_ious[pairwise_ious < self.iou_t] = 0
            max_iou, max_iou_inds = pairwise_ious.max(dim=1)
            output_ids[bi] = gt_labels[bi][max_iou_inds]
            output_ids[bi][max_iou == 0] = -2
            output_ids[bi][det_scores[bi] < self.score_t] = -2
        return output_ids


class DetIDAssigner(IdAssigner):
    def __init__(self, *args, **kws) -> None:
        super().__init__()

    def __call__(self, det_bboxes, det_scores, gt_bboxes, gt_labels, **kws):
        match_indices = kws["match_indices"]
        assign_ids = []
        for det_bboxes_i, gt_labels_i, (src_idxs, tgt_idxs) in zip(
            det_bboxes, gt_labels, match_indices
        ):
            ids_i = (
                torch.zeros(
                    det_bboxes_i.shape[0], dtype=torch.int64, device=det_bboxes_i.device
                )
                - 2
            )
            if src_idxs.shape[0] > 0 and tgt_idxs.shape[0] > 0:
                ids_i[src_idxs] = gt_labels_i[tgt_idxs]
            assign_ids.append(ids_i)
        if isinstance(det_bboxes, torch.Tensor):
            assign_ids = torch.stack(assign_ids)
        return assign_ids



def build_id_assigner(cfg):
    ids_cfg = cfg.PERSON_SEARCH.REID.ID_ASSIGN
    return {
        "iou": IouIDAssigner,
        "det": DetIDAssigner,
    }[ids_cfg.NAME](ids_cfg)
