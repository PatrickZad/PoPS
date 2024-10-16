import torch
from abc import ABCMeta, abstractmethod


class BoxAugmentor(metaclass=ABCMeta):
    def __init__(self, num_labeled, append_gt, num_unlabled=None) -> None:
        self.num_labeled = num_labeled
        if num_unlabled is None:
            self.num_unlabeled = num_labeled
        else:
            self.num_unlabeled = num_unlabled
        self.append_gt = append_gt

    @abstractmethod
    def augment_boxes(
        self, gt_boxes, gt_id, det_boxes=None, det_pids=None, img_sizes=None
    ):
        """generate box for reid training"""


class DNDETR_BoxAugmentor(BoxAugmentor):
    def __init__(self, h_center, h_scale, *args, **kws) -> None:
        super().__init__(*args, **kws)
        self.h_center = h_center
        self.h_scale = h_scale

    @torch.no_grad()
    def augment_boxes(
        self, gt_boxes, gt_id, det_boxes=None, det_pids=None, img_sizes=None
    ):
        out_boxes = []
        out_ids = []
        for pi, (gt_boxes_i, gt_id_i, size_i) in enumerate(
            zip(gt_boxes, gt_id, img_sizes)
        ):
            anchor_c_xs = (gt_boxes_i[:, 0:1] + gt_boxes_i[:, 2:3]) / 2  # n x 1
            anchor_c_ys = (gt_boxes_i[:, 1:2] + gt_boxes_i[:, 3:]) / 2  # n x 1
            org_ws = gt_boxes_i[:, 2:3] - gt_boxes_i[:, :1]
            org_hs = gt_boxes_i[:, 3:] - gt_boxes_i[:, 1:2]
            d_xs = (
                (
                    torch.rand(
                        gt_boxes_i.shape[0], self.num_labeled, device=org_ws.device
                    )
                    - 0.5
                )
                * self.h_center
                * org_ws
                / 2
            )  # n x num_labeled
            d_ys = (
                (
                    torch.rand(
                        gt_boxes_i.shape[0], self.num_labeled, device=org_hs.device
                    )
                    - 0.5
                )
                * self.h_center
                * org_hs
                / 2
            )  # n x num_labeled
            new_c_xs, new_c_ys = anchor_c_xs + d_xs, anchor_c_ys + d_ys
            new_s_ws = (
                torch.rand(gt_boxes_i.shape[0], self.num_labeled, device=org_ws.device)
                - 0.5
            ) * (
                2 * self.h_scale * org_ws
            ) + org_ws  # n x num_labeled
            new_s_hs = (
                torch.rand(gt_boxes_i.shape[0], self.num_labeled, device=org_hs.device)
                - 0.5
            ) * (
                2 * self.h_scale * org_hs
            ) + org_hs  # n x num_labeled
            aug_boxes = torch.stack(
                [
                    (new_c_xs - new_s_ws / 2).clip(0, size_i[1] - 1),
                    (new_c_ys - new_s_hs / 2).clip(0, size_i[0] - 1),
                    (new_c_xs + new_s_ws / 2).clip(0, size_i[1] - 1),
                    (new_c_ys + new_s_hs / 2).clip(0, size_i[0] - 1),
                ],
                dim=-1,
            ).reshape(-1, 4)
            aug_box_ids = gt_id_i.unsqueeze(1).expand(-1, self.num_labeled).reshape(-1)
            if self.append_gt:
                aug_boxes = torch.cat([aug_boxes, gt_boxes_i], dim=0)
                aug_box_ids = torch.cat([aug_box_ids, gt_id_i], dim=0)
            if det_boxes is not None and det_pids is not None:
                aug_boxes = torch.cat([aug_boxes, det_boxes[pi]], dim=0)
                aug_box_ids = torch.cat([aug_box_ids, det_pids[pi]], dim=0)
            out_boxes.append(aug_boxes)
            out_ids.append(aug_box_ids)
        return out_boxes, out_ids


class SEIE_BoxAugmentor(BoxAugmentor):
    def __init__(self, d_center_min, d_center_max, *args, **kws) -> None:
        super().__init__(*args, **kws)
        self.d_center_min = d_center_min
        self.d_center_max = d_center_max

    def augment_boxes(
        self, gt_boxes, gt_id, det_boxes=None, det_pids=None, img_sizes=None
    ):
        out_boxes = []
        out_ids = []
        for pi, (gt_boxes_i, gt_id_i, size_i) in enumerate(
            zip(gt_boxes, gt_id, img_sizes)
        ):
            aug_boxes_cur = []
            aug_ids_cur = []
            for lmask, num_augs in zip(
                [gt_id_i > -1, gt_id_i == -1], [self.num_labeled, self.num_unlabeled]
            ):
                if lmask.sum() > 0:
                    gt_boxes_t = gt_boxes_i[lmask]
                    anchor_c_xs = (gt_boxes_t[:, 0:1] + gt_boxes_t[:, 2:3]) / 2  # n x 1
                    anchor_c_ys = (gt_boxes_t[:, 1:2] + gt_boxes_t[:, 3:]) / 2  # n x 1
                    org_ws = gt_boxes_t[:, 2:3] - gt_boxes_t[:, :1]
                    org_hs = gt_boxes_t[:, 3:] - gt_boxes_t[:, 1:2]
                    d_xs = (
                        torch.rand(gt_boxes_t.shape[0], num_augs, device=org_ws.device)
                        * (self.d_center_max - self.d_center_min)
                        + self.d_center_min
                    )  # n x num_labeled
                    d_ys = (
                        torch.rand(gt_boxes_t.shape[0], num_augs, device=org_hs.device)
                        * (self.d_center_max - self.d_center_min)
                        + self.d_center_min
                    )  # n x num_labeled
                    new_c_xs, new_c_ys = anchor_c_xs + d_xs, anchor_c_ys + d_ys
                    aug_boxes = torch.stack(
                        [
                            (new_c_xs - org_ws / 2).clip(0, size_i[1] - 1),
                            (new_c_ys - org_hs / 2).clip(0, size_i[0] - 1),
                            (new_c_xs + org_ws / 2).clip(0, size_i[1] - 1),
                            (new_c_ys + org_hs / 2).clip(0, size_i[0] - 1),
                        ],
                        dim=-1,
                    ).reshape(
                        -1, 4
                    )  # n x num_aug x 4
                    aug_box_ids = (
                        gt_id_i[lmask].unsqueeze(1).expand(-1, num_augs).reshape(-1)
                    )  # n x num_aug
                    aug_boxes_cur.append(aug_boxes)
                    aug_ids_cur.append(aug_box_ids)
            aug_boxes = torch.cat(aug_boxes_cur, dim=0)
            aug_box_ids = torch.cat(aug_ids_cur, dim=0)
            if self.append_gt:
                aug_boxes = torch.cat([aug_boxes, gt_boxes_i], dim=0)
                aug_box_ids = torch.cat([aug_box_ids, gt_id], dim=0)
            if det_boxes is not None and det_pids is not None:
                aug_boxes = torch.cat([aug_boxes, det_boxes[pi]], dim=0)
                aug_box_ids = torch.cat([aug_box_ids, det_pids[pi]], dim=0)
            out_boxes.append(aug_boxes)
            out_ids.append(aug_box_ids)
        return out_boxes, out_ids


class DKD_BoxAugmentor(BoxAugmentor):
    def __init__(self, d_p, t_size, alpha, *args, **kws) -> None:
        super().__init__(*args, **kws)
        self.d_p = d_p
        self.t_size = t_size
        self.alpha = alpha

    def augment_boxes(self, gt_boxes, gt_id, det_boxes=None, det_pids=None):
        pass


def build_box_augmentor(ba_cfg):
    ba_type = ba_cfg.NAME
    if ba_type == "dn_detr":
        return DNDETR_BoxAugmentor(
            ba_cfg.H_CENTER,
            ba_cfg.H_SCALE,
            ba_cfg.NUM_LABELED,
            ba_cfg.APPEND_GT,
            ba_cfg.NUM_UNLABLED,
        )
    elif ba_type == "seie":
        return SEIE_BoxAugmentor(
            ba_cfg.D_CENTER_MIN,
            ba_cfg.D_CENTER_MAX,
            ba_cfg.NUM_LABELED,
            ba_cfg.APPEND_GT,
            ba_cfg.NUM_UNLABLED,
        )
    elif ba_type == "dkd":
        return SEIE_BoxAugmentor(
            ba_cfg.D_P,
            ba_cfg.T_SIZE,
            ba_cfg.ALPHA,
            ba_cfg.NUM_LABELED,
            ba_cfg.APPEND_GT,
            ba_cfg.NUM_UNLABLED,
        )
