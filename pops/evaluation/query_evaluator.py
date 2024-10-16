from copy import copy
from typing import OrderedDict
import pops.utils.comm as comm
import torch
import logging
from .evaluator import DatasetEvaluator
import itertools
import os
import numpy as np
import copy

logger = logging.getLogger(__name__)  # setup_logger()

class QueryEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
    ) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        # {image name: torch concatenated [boxes, scores, reid features]
        inf_rts = torch.load(
            os.path.join(
                self._output_dir,
                "_gallery_gt_inf.pt"
                if "GT" not in self.dataset_name
                else "_gallery_gt_infgt.pt",
            ),
            map_location=self._cpu_device,
        )
        self.gts = inf_rts["gts"]
        self.infs = inf_rts["infs"]
        self.gtfs = inf_rts["gt_fnames"]
        self.topks = [1, 5, 10]
        # statistics
        self.det_score_thresh = s_threds
        self.ignore_cam_id = False
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}

    def reset(self):
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}

    def _get_gallery_dets(self, img_id, score_thred):
        img_dets_all = self.infs[img_id][:, :5]
        return img_dets_all[img_dets_all[:, 4] >= score_thred]

    def _get_gallery_feats(self, img_id, score_thred):
        img_save_all = self.infs[img_id]
        return img_save_all[img_save_all[:, 4] >= score_thred][:, 5:]

    def _get_gt_boxs(self, img_id):
        return self.gts[img_id][:, :4]

    def _get_gt_ids(self, img_id):
        return self.gts[img_id][:, 4].long()

    def process(self, inputs, outputs):
        raise NotImplementedError

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            aps_all = comm.gather(self.aps, dst=0)
            accs_all = comm.gather(self.accs, dst=0)

            if not comm.is_main_process():
                return {}
            aps = {}
            accs = {}
            for dst in self.det_score_thresh:
                aps[dst] = list(itertools.chain(*[ap[dst] for ap in aps_all]))
                accs[dst] = list(itertools.chain(*[acc[dst] for acc in accs_all]))
        else:
            aps = self.aps
            accs = self.accs
        search_result = OrderedDict()
        search_result["search"] = {}

        for dst in self.det_score_thresh:
            logger.info("Search eval_{:.2f} on {} queries. ".format(dst, len(aps[dst])))
            mAP = np.mean(aps[dst])
            search_result["search"].update({"mAP_{:.2f}".format(dst): mAP})
            acc = np.mean(np.array(accs[dst]), axis=0)
            # logger.info(str(acc))
            for i, v in enumerate(acc.tolist()):
                # logger.info("{:.2f} on {} acc. ".format(v, i))
                k = self.topks[i]
                search_result["search"].update({"top{:2d}_{:.2f}".format(k, dst): v})


        return copy.deepcopy(search_result)

    def _get_query_gallery_gts(self, query_dict):
        raise NotImplementedError

