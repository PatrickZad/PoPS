from .query_evaluator import QueryEvaluator
from copy import copy
import pops.utils.comm as comm
import torch
import logging
import itertools
import numpy as np
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score
import copy
import re
from pops.structures import Boxes, BoxMode, pairwise_iou

logger = logging.getLogger(__name__)


def _get_img_cid(img_id):
    cn = re.match(r"c(\d+)s.*", img_id)
    cn = int(cn.groups()[0])
    return cn


class PrwQueryEvaluator(QueryEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
    ):
        super().__init__(
            dataset_name, distributed, output_dir, s_threds
        )
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}

    def reset(self):
        super().reset()
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}

    def process(self, inputs, outputs):
        """
        Args:
            inputs:
                a batch of
                { "query":
                    {
                        "image": augmented image tensor
                        "instances": an Instances object with attrs
                            {
                                image_size: hw (int, int)
                                file_name: full path string
                                image_id: filename string
                                gt_boxes: Boxes (1 , 4)
                                gt_classes: tensor full with 0s
                                gt_pids: person identity tensor (1,)
                                org_img_size: hw before augmentation (int, int)
                                org_gt_boxes: Boxes before augmentation
                            }
                    }
                }
            outputs:
                a batch of instances with attrs
                {
                    reid_feats: tensor (1,pfeat_dim)
                }
                or
                a batch of empty instances
        """

        for bi, in_dict in enumerate(inputs):
            query_dict = in_dict["query"]
            q_instances = query_dict["instances"].to(self._cpu_device)
            q_imgid = q_instances.image_id
            q_pid = q_instances.gt_pids
            q_box = q_instances.org_gt_boxes
            q_cid = _get_img_cid(q_imgid)
            y_trues = {dst: [] for dst in self.det_score_thresh}
            y_scores = {dst: [] for dst in self.det_score_thresh}
            count_gts = {dst: 0 for dst in self.det_score_thresh}
            count_tps = {dst: 0 for dst in self.det_score_thresh}
            y_trues_mlv = {dst: [] for dst in self.det_score_thresh}
            y_scores_mlv = {dst: [] for dst in self.det_score_thresh}
            count_gts_mlv = {dst: 0 for dst in self.det_score_thresh}
            count_tps_mlv = {dst: 0 for dst in self.det_score_thresh}
            # Find all occurence of this query
            # Construct gallery set for this query
            gallery_imgs = []
            query_gts = {}
            for gt_img_id, gt_img_label in self.gts.items():
                if gt_img_id != q_imgid:
                    gt_boxes_t = gt_img_label[:, :4]
                    gt_ids = gt_img_label[:, 4].long()
                    gallery_imgs.append(
                        {
                            "image_id": gt_img_id,
                            "boxes_t": gt_boxes_t,
                            "ids": gt_ids,
                        }
                    )
                    if q_pid in gt_ids.tolist():
                        query_gts[gt_img_id] = gt_boxes_t[gt_ids == q_pid].squeeze(0)

            for dst in self.det_score_thresh:
                pred_instances = outputs[bi].to(self._cpu_device)
                if pred_instances.has("reid_feats"):
                    feat_q = pred_instances.reid_feats
                else:
                    query_img_boxes_t, query_img_feats = self._get_gallery_dets(
                        q_imgid, dst
                    )[:, :4], self._get_gallery_feats(q_imgid, dst)
                    if query_img_boxes_t.shape[0] == 0:
                        # no detection in this query image
                        logger.warning(
                            "Undetected query person in {} !".format(q_imgid)
                        )
                        continue
                    else:
                        ious = pairwise_iou(
                            q_box, Boxes(query_img_boxes_t, BoxMode.XYXY_ABS)
                        )
                        max_iou, nmax = torch.max(ious, dim=1)
                        if max_iou < 0.4:
                            logger.warning(
                                "Low-quality {} query person detected in {} !".format(
                                    max_iou.item(), q_imgid
                                )
                            )
                        feat_q = query_img_feats[nmax.item()]

                # feat_q = F.normalize(feat_q[None]).squeeze(0)  # NOTE keep post norm
                name2sim = {}
                # save for vis
                g_img_ids = []
                # 1. Go through the gallery samples defined by the protocol
                for item in gallery_imgs:
                    gallery_imname = item["image_id"]
                    g_cid = _get_img_cid(gallery_imname)
                    # some contain the query (gt not empty), some not
                    count_gts[dst] += gallery_imname in query_gts
                    if g_cid != q_cid:
                        # some contain the query (gt not empty), some not
                        count_gts_mlv[dst] += gallery_imname in query_gts
                    # compute distance between query and gallery dets
                    if gallery_imname not in self.infs:
                        continue
                    det, feat_g = self._get_gallery_dets(
                        gallery_imname, dst
                    ), self._get_gallery_feats(gallery_imname, dst)
                    # no detection in this gallery, skip it
                    if det.shape[0] == 0:
                        continue
                    # get L2-normalized feature matrix NxD
                    # feat_g = F.normalize(feat_g)  # NOTE keep post norm
                    # compute cosine similarities
                    sim = torch.mm(feat_g, feat_q.view(-1)[:, None]).squeeze(
                        1
                    )  # n x 1 -> n
                    if gallery_imname in name2sim:
                        continue
                    name2sim[gallery_imname] = sim
                    # save for vis
                    g_img_ids.append(gallery_imname)

                    label = torch.zeros(sim.shape[0], dtype=torch.int)
                    if gallery_imname in query_gts:
                        gt = query_gts[gallery_imname]
                        w, h = gt[2] - gt[0], gt[3] - gt[1]
                        iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                        inds = torch.argsort(sim)
                        inds = inds.tolist()[::-1]
                        inds = torch.tensor(inds, dtype=torch.long)
                        sim = sim[inds]
                        det = det[inds]
                        # only set the first matched det as true positive
                        for j, roi in enumerate(det[:, :4]):
                            if (
                                box_iou(roi[None, :], gt[None, :]).squeeze().item()
                                >= iou_thresh
                            ):
                                label[j] = 1
                                count_tps[dst] += 1
                                if q_cid != g_cid:
                                    count_tps_mlv[dst] += 1
                                break
                    y_trues[dst].extend(label.tolist())
                    y_scores[dst].extend(sim.tolist())
                    # multi view
                    if g_cid != q_cid:
                        y_trues_mlv[dst].extend(label.tolist())
                        y_scores_mlv[dst].extend(sim.tolist())

                # 2. Compute AP for this probe (need to scale by recall rate)

                y_score = np.asarray(y_scores[dst])
                y_true = np.asarray(y_trues[dst])
                assert count_tps[dst] <= count_gts[dst]
                recall_rate = count_tps[dst] * 1.0 / count_gts[dst]
                ap = (
                    0
                    if count_tps[dst] == 0
                    else average_precision_score(y_true, y_score) * recall_rate
                )
                self.aps[dst].append(ap)
                inds = np.argsort(y_score)[::-1]
                y_score = y_score[inds]
                y_true = y_true[inds]
                self.accs[dst].append([min(1, sum(y_true[:k])) for k in self.topks])
                # mlv
                y_score_mlv = np.asarray(y_scores_mlv[dst])
                y_true_mlv = np.asarray(y_trues_mlv[dst])
                assert count_tps_mlv[dst] <= count_gts_mlv[dst]
                recall_rate_mlv = count_tps_mlv[dst] * 1.0 / count_gts_mlv[dst]
                ap_mlv = (
                    0
                    if count_tps_mlv[dst] == 0
                    else average_precision_score(y_true_mlv, y_score_mlv)
                    * recall_rate_mlv
                )
                self.aps_mlv[dst].append(ap_mlv)
                inds_mlv = np.argsort(y_score_mlv)[::-1]
                y_score_mlv = y_score_mlv[inds_mlv]
                y_true_mlv = y_true_mlv[inds_mlv]
                self.accs_mlv[dst].append(
                    [min(1, sum(y_true_mlv[:k])) for k in self.topks]
                )

    def evaluate(self):
        mix_eval_results = super().evaluate()
        if self._distributed:
            comm.synchronize()
            aps_all = comm.gather(self.aps_mlv, dst=0)
            accs_all = comm.gather(self.accs_mlv, dst=0)
            if not comm.is_main_process():
                return {}
            aps = {}
            accs = {}
            for dst in self.det_score_thresh:
                aps[dst] = list(itertools.chain(*[ap[dst] for ap in aps_all]))
                accs[dst] = list(itertools.chain(*[acc[dst] for acc in accs_all]))
        else:
            aps = self.aps_mlv
            accs = self.accs_mlv

        for dst in self.det_score_thresh:
            logger.info(
                "Multi-view Search eval_{:.2f} on {} queries. ".format(
                    dst, len(aps[dst])
                )
            )
            mAP = np.mean(aps[dst])
            mix_eval_results["search"].update({"mAP_{:.2f}_mlv".format(dst): mAP})
            acc = np.mean(accs[dst], axis=0)
            # logger.info(str(acc))
            for i, v in enumerate(acc.tolist()):
                # logger.info("{:.2f} on {} acc. ".format(v, i))
                k = self.topks[i]
                mix_eval_results["search"].update(
                    {"top{:2d}_{:.2f}_mlv".format(k, dst): v}
                )
        return copy.deepcopy(mix_eval_results)

