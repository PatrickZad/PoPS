# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from pops.layers import nonzero_tuple
from pops.structures import Boxes, pairwise_giou, BoxMode
from pops.config.config import configurable


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self,
        thresholds: List[float],
        labels: List[int],
        allow_low_quality_matches: bool = False,
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # Currently torchscript does not support all + generator
        assert all(
            [low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])]
        )
        assert all([l in [-1, 0, 1] for l in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(
            self.labels, self.thresholds[:-1], self.thresholds[1:]
        ):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        _, pred_inds_with_highest_quality = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
        # This follows the implementation in Detectron, and is found to have no significant impact.
        match_labels[pred_inds_with_highest_quality] = 1


EPS = 1e-10


class HungarianMatcher(nn.Module):
    """
    NOTE assume pred boxes to be XYXY_ABS, gt boxes to be XYXY_ABS
    """

    @configurable()
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        use_focal: bool = False,
        focal_alpha=0.25,
        focal_gamma=2,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = focal_alpha
            self.focal_loss_gamma = focal_gamma
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @classmethod
    def from_config(cls, cfg):
        det_cfg = cfg.PERSON_SEARCH.DET
        return dict(
            cost_class=det_cfg.LOSS.WEIGHTS.CLS,
            cost_bbox=det_cfg.LOSS.WEIGHTS.L1,
            cost_giou=det_cfg.LOSS.WEIGHTS.GIOU,
            use_focal=det_cfg.LOSS.FOCAL.ENABLE,
            focal_alpha=det_cfg.LOSS.FOCAL.ALPHA,
            focal_gamma=det_cfg.LOSS.FOCAL.GAMMA,
        )

    def _cls_cost(self, outputs, targets, bs, nq, sizes):
        pred_logits = outputs["pred_logits"]
        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = pred_logits.flatten(
                0, 1
            ).sigmoid()  # [batch_size * num_queries, num_classes]
        else:
            out_prob = pred_logits.flatten(0, 1).softmax(
                -1
            )  # [batch_size * num_queries, num_classes]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v.gt_classes for v in targets])
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (
                (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]
        cost_class_bs = cost_class.view(bs, nq, -1)
        cost_class_bs = [c[i] for i, c in enumerate(cost_class_bs.split(sizes, -1))]

        return cost_class_bs

    def _bbox_cost(self, outputs, targets, bs, nq, sizes):
        pred_boxes = outputs["pred_boxes"]
        aug_whwh_bs = torch.tensor(
            [[v.image_size[1], v.image_size[0]] * 2 for v in targets],
            device=pred_boxes.device,
        )
        out_bbox = pred_boxes / aug_whwh_bs.unsqueeze(
            1
        )  # [batch_size , num_queries, 4]

        out_bbox = out_bbox.flatten(0, 1)  # (0,1) xyxy in aug image
        tgt_bbox = torch.cat(
            [v.gt_boxes.tensor / aug_whwh_bs[vi][None] for vi, v in enumerate(targets)]
        )
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_bbox_bs = cost_bbox.view(bs, nq, -1)
        cost_bbox_bs = [c[i] for i, c in enumerate(cost_bbox_bs.split(sizes, -1))]
        return cost_bbox_bs

    def _giou_cost(self, outputs, targets, bs, nq, sizes):
        pred_boxes = outputs["pred_boxes"]
        out_bbox = Boxes(
            pred_boxes.view(-1, 4)
        )  # [batch_size * num_queries, 4]# x,y,x,y  of aug image
        tgt_bbox = Boxes.cat([v.gt_boxes for v in targets])  # x,y,x,y  of aug image
        # Compute the giou cost betwen boxes
        cost_giou = -pairwise_giou(out_bbox, tgt_bbox)
        cost_giou_bs = cost_giou.view(bs, nq, -1)
        cost_giou_bs = [c[i] for i, c in enumerate(cost_giou_bs.split(sizes, -1))]

        return cost_giou_bs

    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: batched prediction dict

            targets: gt instances

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            sizes = [len(v.gt_boxes) for v in targets]

            cost_class_bs = self._cls_cost(outputs, targets, bs, num_queries, sizes)
            cost_bbox_bs = self._bbox_cost(outputs, targets, bs, num_queries, sizes)
            cost_giou_bs = self._giou_cost(outputs, targets, bs, num_queries, sizes)

            # Final cost matrix
            C = [
                self.cost_bbox * cost_bbox_bs[i]
                + self.cost_class * cost_class_bs[i]
                + self.cost_giou * cost_giou_bs[i]
                for i in range(bs)
            ]
            # C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c.cpu()) for c in C]
            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]


class DtHungarianMatcher(HungarianMatcher):
    """
    NOTE assume pred boxes to be CCWH_REL, gt boxes to be XYXY_ABS
    """

    def _bbox_cost(self, outputs, targets, bs, nq, sizes):
        pred_boxes = outputs["pred_boxes"]

        out_bbox = pred_boxes.flatten(0, 1)  # (0,1) xyxy in aug image
        tgt_bbox = torch.cat(
            [
                inst.gt_boxes.convert_mode(BoxMode.CCWH_REL, inst.image_size).tensor
                for inst in targets
            ]
        )
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_bbox_bs = cost_bbox.view(bs, nq, -1)
        cost_bbox_bs = [c[i] for i, c in enumerate(cost_bbox_bs.split(sizes, -1))]
        return cost_bbox_bs

    def _giou_cost(self, outputs, targets, bs, nq, sizes):
        pred_boxes = outputs["pred_boxes"]
        out_bbox = Boxes(pred_boxes.view(-1, 4), BoxMode.CCWH_REL).convert_mode(
            BoxMode.XYXY_REL
        )  # [batch_size * num_queries, 4]# x,y,x,y  of aug image

        tgt_bbox = Boxes.cat(
            [v.gt_boxes.convert_mode(BoxMode.XYXY_REL, v.image_size) for v in targets]
        )  # x,y,x,y  of aug image
        # Compute the giou cost betwen boxes
        cost_giou = -pairwise_giou(out_bbox, tgt_bbox)
        cost_giou_bs = cost_giou.view(bs, nq, -1)
        cost_giou_bs = [c[i] for i, c in enumerate(cost_giou_bs.split(sizes, -1))]

        return cost_giou_bs
