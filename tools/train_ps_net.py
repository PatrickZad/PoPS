#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in pops.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use pops as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import sys

sys.path.append("./")
import logging
import os
from collections import OrderedDict
import torch
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import pops.utils.comm as comm
from pops.checkpoint import DetectionCheckpointer
from pops.config import get_cfg
from pops.data import MetadataCatalog
from pops.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from pops.evaluation import (
    InfDetEvaluator,
    PrwQueryEvaluator,
    CuhkQueryEvaluator,
    DatasetEvaluators,
    verify_results,
    MovieNetQueryEvaluator,
)
import re
from pops.modeling import GeneralizedRCNNWithTTA
import copy

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    single_gpu = comm.get_world_size() == 1
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type is "det":
        evaluator_list.append(
            InfDetEvaluator(
                dataset_name,
                distributed=not single_gpu,
                output_dir=output_folder,
                s_threds=cfg.TEST.DETECTION_SCORE_TS,
                topk=cfg.TEST.DETECTIONS_PER_IMAGE,
            )
        )
    elif evaluator_type is "query":
        if "CUHK-SYSU" in dataset_name:
            evaluator_list.append(
                CuhkQueryEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    s_threds=cfg.TEST.DETECTION_SCORE_TS,
                )
            )
        elif "PRW" in dataset_name:
            evaluator_list.append(
                PrwQueryEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    s_threds=cfg.TEST.DETECTION_SCORE_TS,
                )
            )
        elif "MovieNet" in dataset_name:
            evaluator_list.append(
                MovieNetQueryEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    s_threds=cfg.TEST.DETECTION_SCORE_TS,
                )
            )
        else:
            raise ValueError("Unknown dataset {}".format(dataset_name))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        # NOTE not tested
        logger = logging.getLogger("pops.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_train_loader(cls, cfg):
        from pops.data.catalog import MapperCatalog
        from pops.data.build import build_detection_train_loader

        mapper = MapperCatalog.get(cfg.DATASETS.TRAIN[0])(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        from pops.data.catalog import MapperCatalog
        from pops.data.build import get_detection_dataset_dicts, trivial_batch_collator
        from pops.data.common import DatasetFromList, MapDataset
        from pops.data.samplers import InferenceSampler
        import torch.utils.data as torchdata

        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
                for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        dataset_idx=cfg.DATASETS.TEST.index(dataset_name)
        if not isinstance(cfg.INPUT.MIN_SIZE_TEST_INC,int):
            cfg=copy.deepcopy(cfg)
            cfg.defrost()
            cfg.INPUT.MIN_SIZE_TEST=cfg.INPUT.MIN_SIZE_TEST_INC[dataset_idx]
            cfg.INPUT.MAX_SIZE_TEST=cfg.INPUT.MAX_SIZE_TEST_INC[dataset_idx]
            cfg.freeze()
        mapper = MapperCatalog.get(dataset_name)(cfg, is_train=False)
        # batched test
        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset, copy=False)
        dataset = MapDataset(dataset, mapper)
        sampler = InferenceSampler(len(dataset))

        batch_sampler = torchdata.sampler.BatchSampler(
            sampler, cfg.TEST.IMS_PER_PROC, drop_last=False
        )
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader

    @classmethod
    def build_optimizer(cls, cfg, model):
        from pops.solver.build import maybe_add_gradient_clipping

        logger = logging.getLogger("pops.trainer")
        frozen_params = []
        learn_param_keys = []
        param_groups = [{"params": [], "lr": cfg.SOLVER.BASE_LR}] + [
            {"params": [], "lr": cfg.SOLVER.BASE_LR * lf}
            for lf in cfg.SOLVER.LR_FACTORS
        ]
        freeze_regex = [re.compile(reg) for reg in cfg.SOLVER.FREEZE_PARAM_REGEX]
        lr_group_regex = [re.compile(reg) for reg in cfg.SOLVER.LR_GROUP_REGEX]

        def _find_match(pkey, prob_regs):
            match_idx = -1
            for mi, mreg in enumerate(prob_regs):
                if re.match(mreg, pkey):
                    assert match_idx == -1, "Ambiguous matching of {}".format(pkey)
                    match_idx = mi
            return match_idx

        for key, value in model.named_parameters(recurse=True):
            match_freeze = _find_match(key, freeze_regex)
            if match_freeze > -1:
                value.requires_grad = False
            if not value.requires_grad:
                frozen_params.append(key)
                continue
            match_learn = _find_match(key, lr_group_regex)
            if match_learn > 0:
                param_groups[match_learn]["params"].append(value)
            else:
                param_groups[0]["params"].append(value)
            learn_param_keys.append(key)
        # logger.info("Frozen parameters:\n{}".format("\n".join(frozen_params)))
        logger.info("Training parameters:\n{}".format("\n".join(learn_param_keys)))
        optim = cfg.SOLVER.OPTIM
        if optim == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optim == "Adam":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optim == "AdamW":
            return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError("Unsupported optimizer {}".format(optim))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg, model_find_unused_parameters=not args.n_find_unparams)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
