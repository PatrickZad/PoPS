import pops.utils.comm as comm

import torch
import logging

from .evaluator import DatasetEvaluator
import itertools
import os
import numpy as np

from sklearn.metrics import average_precision_score
import copy
from collections import OrderedDict

from pops.structures import  pairwise_iou


logger = logging.getLogger(__name__)


# NOTE evaluation in augmented box range
class InfDetEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        topk=100,
        s_threds=[0.05, 0.1, 0.2],
    ) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.threshs = s_threds
        self.iou_threshs = [0.5, 0.7]
        self.topk = topk
        # (image name,torch concatenated [boxes, scores, reid features])
        self.inf_results = {}
        # (image name,torch concatenated [boxes, ids])
        self.gallery_gts = {}
        self.gallery_labels = {}
        self.granks_of_local_0s = []
        # det statistic
        self.y_trues = {}
        self.y_scores = {}
        self.count_gt = 0
        self.count_gt_lb = 0
        self.count_tp = 0
        self.count_tp_lb = 0
        # in-depth analysis
        self.count_tp_ulb = 0
        self.count_false = 0
        self.tp_ious = []

        

    def reset(self):
        self.inf_results = {}
        self.gallery_gts = {}
        self.granks_of_local_0s = []
        self.gallery_fnames = {}  # add for vis
        # det statistic
        self.y_trues = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
        self.y_scores = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
        self.count_gt = {iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs}
        self.count_tp = {iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs}
        self.count_gt_lb = {
            iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
        }
        self.count_tp_lb = {
            iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
        }
        self.count_tp_ulb = {
            iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
        }
        self.count_false = {
            iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
        }
        self.tp_ious = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}

    def process(self, inputs, outputs):
        """
        Args:
            inputs:
                a batch of
                {
                    "image": augmented image tensor
                    "instances": an Instances object with attrs
                        {
                            image_size: hw (int, int)
                            file_name: full path string
                            image_id: filename string
                            gt_boxes: Boxes
                            gt_classes: tensor full with 0s
                            gt_pids: person identity tensor in [-1, max_id]
                            org_img_size: hw before augmentation (int, int)
                            org_gt_boxes: Boxes before augmentation
                        }
                }
            outputs:
                a batch of instances with attrs
                {
                    pred_boxes: Boxes in augmentation range (resizing/cropping/flipping/...)
                    pred_scores: tensor
                    reid_feats: tensor
                }
        """

        for bi, in_dict in enumerate(inputs):
            gt_instances = in_dict["instances"].to(self._cpu_device)
            pred_instances = outputs[bi].to(self._cpu_device)
            # save gt info
            gt_img_name = gt_instances.image_id
            org_gt_boxes = gt_instances.org_gt_boxes
            gt_pids = gt_instances.gt_pids
            org_gt_label_t = torch.cat(
                [org_gt_boxes.tensor, gt_pids.view(-1, 1)], dim=-1
            )
            self.gallery_gts[gt_img_name] = org_gt_label_t
            self.gallery_fnames[gt_img_name] = gt_instances.file_name
            # save inf results
            scores_t = pred_instances.pred_scores.view(-1)
            if scores_t.shape[0] > self.topk:
                _, idx = torch.topk(scores_t, self.topk)
                scores_t = scores_t[idx]
                out_boxes = pred_instances.pred_boxes[idx]
                out_feats_t = pred_instances.reid_feats[idx]
            else:
                out_boxes = pred_instances.pred_boxes
                out_feats_t = pred_instances.reid_feats
            inf_rt = torch.cat(
                [out_boxes.tensor, scores_t.view(-1, 1), out_feats_t], dim=-1
            )
            self.inf_results[gt_img_name] = inf_rt
        for st in self.threshs:
            self._det_proc(inputs, outputs, st)

    def _det_proc(self, inputs, outputs, sthred):
        for bi, in_dict in enumerate(inputs):
            gt_instances = in_dict["instances"].to(self._cpu_device)
            pred_instances = outputs[bi].to(self._cpu_device)
            gt_boxes = gt_instances.org_gt_boxes
            gt_pids_t = gt_instances.gt_pids
            det_boxes = pred_instances.pred_boxes
            det_scores_t = pred_instances.pred_scores.view(-1)
            num_gts = len(gt_boxes)
            num_lb_gts = (gt_pids_t > -1).sum().item()
            if len(det_boxes) > 0:
                if len(det_boxes) > self.topk:
                    topk_scores, topk_idxs = torch.topk(det_scores_t, self.topk)
                    topk_det_boxes = det_boxes[topk_idxs]
                    det_scores_t = topk_scores[topk_scores >= sthred]
                    det_boxes = topk_det_boxes[topk_scores >= sthred]
                    num_dets = len(det_boxes)
                else:
                    det_scores_mask = det_scores_t >= sthred
                    det_boxes = det_boxes[det_scores_mask]
                    num_dets = len(det_boxes)
            else:
                num_dets = 0
            if num_dets == 0:
                for iou_t in self.iou_threshs:
                    self.count_gt[iou_t][sthred] += num_gts
                    self.count_gt_lb[iou_t][sthred] += num_lb_gts
                continue

            ious = pairwise_iou(gt_boxes, det_boxes)
            for iou_t in self.iou_threshs:
                tfmat = ious >= iou_t
                # for each det, keep only the largest iou of all the gt
                for j in range(num_dets):
                    largest_ind = torch.argmax(ious[:, j])
                    for i in range(num_gts):
                        if i != largest_ind:
                            tfmat[i, j] = False
                # for each gt, keep only the largest iou of all the det
                for i in range(num_gts):
                    largest_ind = np.argmax(ious[i, :])
                    for j in range(num_dets):
                        if j != largest_ind:
                            tfmat[i, j] = False
                for j in range(num_dets):
                    self.y_scores[iou_t][sthred].append(det_scores_t[j].item())
                    self.y_trues[iou_t][sthred].append(tfmat[:, j].any())
                self.count_tp[iou_t][sthred] += tfmat.sum().item()
                self.count_gt[iou_t][sthred] += num_gts
                tfmat_lb = tfmat[gt_pids_t > -1, :]
                if tfmat_lb.shape[0] > 0:
                    cur_tp_lb = tfmat_lb.sum().item()
                else:
                    cur_tp_lb = 0
                self.count_tp_lb[iou_t][sthred] += cur_tp_lb
                self.count_gt_lb[iou_t][sthred] += num_lb_gts
                tfmat_ulb = tfmat[gt_pids_t == -1, :]
                if tfmat_ulb.shape[0] > 0:
                    cur_tp_ulb = tfmat_ulb.sum().item()
                else:
                    cur_tp_ulb = 0
                self.count_tp_ulb[iou_t][sthred] += cur_tp_ulb
                self.count_false[iou_t][sthred] += (
                    tfmat.shape[1] - cur_tp_lb - cur_tp_ulb
                )
                tp_ious = ious[tfmat].cpu().numpy().tolist()
                self.tp_ious[iou_t][sthred].extend(tp_ious)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            if comm.get_local_rank() == 0:
                self.granks_of_local_0s.append(comm.get_rank())
            save_ranks = comm.all_gather(self.granks_of_local_0s)
            save_ranks = list(set(itertools.chain(*save_ranks)))
            save_rts = {}
            save_gts = {}
            save_gtfs = {}
            for rk in save_ranks:
                g_gts = comm.gather(self.gallery_gts, dst=rk)
                inf_rts = comm.gather(self.inf_results, dst=rk)
                f_gts = comm.gather(self.gallery_fnames, dst=rk)
                if len(inf_rts) > 0:
                    for rts in inf_rts:
                        save_rts.update(rts)
                if len(g_gts) > 0:
                    for gts in g_gts:
                        save_gts.update(gts)
                if len(f_gts) > 0:
                    for fs in f_gts:
                        save_gtfs.update(fs)
            y_true_rs = comm.gather(self.y_trues, dst=0)
            y_score_rs = comm.gather(self.y_scores, dst=0)
            count_gt_rs = comm.gather(self.count_gt, dst=0)
            count_tp_rs = comm.gather(self.count_tp, dst=0)
            count_gt_lb_rs = comm.gather(self.count_gt_lb, dst=0)
            count_tp_lb_rs = comm.gather(self.count_tp_lb, dst=0)
            count_tp_ulb_rs = comm.gather(self.count_tp_ulb, dst=0)
            count_false_rs = comm.gather(self.count_false, dst=0)
            tp_ious_rs = comm.gather(self.tp_ious, dst=0)
            y_trues = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
            y_scores = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
            count_gts = {iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs}
            count_tps = {iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs}
            count_gts_lb = {
                iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
            }
            count_tps_lb = {
                iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
            }
            count_tps_ulb = {
                iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
            }
            count_false = {
                iou: {k: 0 for k in self.threshs} for iou in self.iou_threshs
            }
            tp_ious = {iou: {k: [] for k in self.threshs} for iou in self.iou_threshs}
            for iou_t in self.iou_threshs:
                for st in self.threshs:
                    y_trues[iou_t][st] = list(
                        itertools.chain(*[yt[iou_t][st] for yt in y_true_rs])
                    )
                    y_scores[iou_t][st] = list(
                        itertools.chain(*[ys[iou_t][st] for ys in y_score_rs])
                    )
                    count_gts[iou_t][st] = sum([cg[iou_t][st] for cg in count_gt_rs])
                    count_tps[iou_t][st] = sum([ct[iou_t][st] for ct in count_tp_rs])
                    count_gts_lb[iou_t][st] = sum(
                        [cg[iou_t][st] for cg in count_gt_lb_rs]
                    )
                    count_tps_lb[iou_t][st] = sum(
                        [ct[iou_t][st] for ct in count_tp_lb_rs]
                    )
                    count_tps_ulb[iou_t][st] = sum(
                        [ct[iou_t][st] for ct in count_tp_ulb_rs]
                    )
                    count_false[iou_t][st] = sum(
                        [ct[iou_t][st] for ct in count_false_rs]
                    )
                    tp_ious[iou_t][st] = list(
                        itertools.chain(*[ti[iou_t][st] for ti in tp_ious_rs])
                    )
            if len(save_rts) == 0 or len(save_gts) == 0 or len(save_gtfs) == 0:
                comm.synchronize()
                return {}

        else:
            save_rts = self.inf_results
            save_gts = self.gallery_gts
            save_gtfs = self.gallery_fnames
            y_trues = self.y_trues
            y_scores = self.y_scores
            count_gts = self.count_gt
            count_tps = self.count_tp
            count_gts_lb = self.count_gt_lb
            count_tps_lb = self.count_tp_lb
            count_tps_ulb = self.count_tp_ulb
            count_false = self.count_false
            tp_ious = self.tp_ious
        save_dict = {"gts": save_gts, "infs": save_rts, "gt_fnames": save_gtfs}
        save_path = os.path.join(
            self._output_dir,
            "_gallery_gt_inf.pt"
            if "GT" not in self.dataset_name
            else "_gallery_gt_infgt.pt",
        )
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        torch.save(save_dict, save_path)
        comm.synchronize()
        # det eval
        if not comm.is_main_process():
            return {}
        det_result = OrderedDict()
        det_result["detection"] = {}
        for iou_t in self.iou_threshs:
            for st in self.threshs:
                count_tp = count_tps[iou_t][st]
                count_gt = count_gts[iou_t][st]
                count_tp_lb = count_tps_lb[iou_t][st]
                count_tp_ulb = count_tps_ulb[iou_t][st]
                count_neg = count_false[iou_t][st]
                count_gt_lb = count_gts_lb[iou_t][st]
                y_true = y_trues[iou_t][st]
                y_score = y_scores[iou_t][st]
                det_rate = count_tp * 1.0 / count_gt
                det_rate_lb = count_tp_lb * 1.0 / count_gt_lb
                if count_tp == 0:
                    det_result["detection"].update(
                        {"AP_{}".format(st): 0, "Recall_{}".format(st): 0}
                    )
                    continue
                y_true = np.array(y_true, dtype=np.int32)
                y_score = np.array(y_score, dtype=np.float32)
                ap = average_precision_score(y_true, y_score) * det_rate
                det_result["detection"].update(
                    {
                        "AP_iou{}_score{}".format(iou_t, st): ap,
                        "Recall_iou{}_score{}".format(iou_t, st): det_rate,
                        "RecallLb_iou{}_score{}".format(iou_t, st): det_rate_lb,
                        "Num_Lb_iou{}_score{}".format(iou_t, st): count_tp_lb,
                        "Num_Ulb_iou{}_score{}".format(iou_t, st): count_tp_ulb,
                        "Num_Neg_iou{}_score{}".format(iou_t, st): count_neg,
                    }
                )
        return copy.deepcopy(det_result)

    
