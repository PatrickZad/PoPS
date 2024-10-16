import pops.data.transforms as dT
import torchvision.transforms as tT

import numpy as np

from ..detection_utils import read_image


import numbers
import warnings

import math
from pops.structures import Instances, Boxes, BoxMode
import torch
from pops.utils.simple_tokenizer import SimpleTokenizer

class SearchMapper(object):
    def __init__(self, cfg, is_train) -> None:
        self.is_train = is_train
        self.in_fmt = cfg.INPUT.FORMAT
        if self.is_train:
            self.augs = dT.AugmentationList(
                [
                    dT.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN,
                        cfg.INPUT.MAX_SIZE_TRAIN,
                        size_divisibility=1, # cfg.INPUT.SIZE_DIVISIBILITY, ImageList takes care
                        sample_style="choice",
                    ),
                    dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                ]
            )
        else:
            self.augs = dT.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                size_divisibility=1, # cfg.INPUT.SIZE_DIVISIBILITY, ImageList takes care
                sample_style="choice",
            )
        self.totensor = (
            tT.ToTensor()
        )  # tT.Normalize(mean=img_mean, std=img_std) is moved to model: preprocess_input

    def __call__(self, img_dict):
        """
        Args:
            {
                "file_name": full path,
                "image_id": image name,
                 "annotations": [
                        K x
                        {
                            "bbox": single box,
                            "bbox_mode": BoxMode,
                            "person_id": person id,
                        }
                    ],
            }
            or
            {
                "query:
                    {
                        "file_name": full path,
                        "image_id": image name,
                        "annotations":
                            [
                                K x {
                                    "bbox": single box,
                                    "bbox_mode": BoxMode,
                                    "person_id": person id,
                                }
                            ],
                    }
                "gallery": (optional)
                    [
                        N x {
                            "file_name": full path,
                            "image_id": image name,
                            "annotations":
                                [
                                    1 x {
                                        "bbox": single box,
                                        "bbox_mode": BoxMode,
                                        "person_id": person id,
                                    }
                                    or
                                    1 x {
                                        "bbox": empty [],
                                        "bbox_mode": BoxMode,
                                        "person_id": None,
                                    }
                                ],
                        }
                    ]

            }
        Outs:
            {
                "image":image tensor in (0,1), unnormed,
                "height": h,
                "width": w,
                "file_name": image ful path,
                "image_id": image name,
                "gt_boxes": augmented Boxes,
                "gt_pids": person ids tensor,
                "gt_classes": class ids tensor,
                "org_img_size": original image size (oh,ow),
                "org_gt_boxes": original Boxes,
            }
            or
            {
                "query":
                    {
                        "image":image tensor in (0,1), unnormed,
                        "height": h,
                        "width": w,
                        "file_name": image ful path,
                        "image_id": image name,
                        "gt_boxes": augmented Boxes,
                        "gt_pids": person ids tensor,
                        "gt_classes": class ids tensor,
                        "org_img_size": original image size (oh,ow),
                        "org_gt_boxes": original Boxes,
                    }
                "gallery":
                    [
                        N x {
                            "file_name": image ful path,
                            "image_id": image name,
                            "gt_pids": person ids tensor,
                            "org_gt_boxes": original Boxes,
                        }
                    ]
            }
        """
        if "query" in img_dict.keys():
            rst = {}
            rst["query"] = self._transform_annotations(img_dict["query"])
            if "gallery" in img_dict:
                rst["gallery"] = []
                for gdict in img_dict["gallery"]:
                    gann = gdict["annotations"][0]
                    box_mode = gann["bbox_mode"]
                    boxes = gann["bbox"]
                    img_path = gdict["file_name"]
                    if box_mode != BoxMode.XYXY_ABS and boxes.size > 0:
                        img_arr = read_image(img_path, self.in_fmt)
                        boxes = Boxes(boxes, box_mode).convert_mode(
                            BoxMode.XYXY_ABS, img_arr.shape[:2]
                        )
                    else:
                        # boxes=torch.zeros((0,4))
                        boxes = Boxes(boxes, BoxMode.XYXY_ABS)
                    g_instance = Instances(
                        None,
                        file_name=img_path,
                        image_id=gdict["image_id"],
                        org_gt_boxes=boxes,
                        gt_pids=torch.tensor([gann["person_id"]], dtype=torch.int)
                        if gann["person_id"] is not None
                        else None,
                    )
                    rst["gallery"].append(g_instance)
            return rst
        else:
            return self._transform_annotations(img_dict)

    def _transform_annotations(self, img_dict):
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.in_fmt)
        boxes = []
        ids = []
        for ann in img_dict["annotations"]:
            box_mode = ann["bbox_mode"]
            boxes.append(
                Boxes(ann["bbox"], box_mode)
                .convert_mode(BoxMode.XYXY_ABS, img_arr.shape[:2])
                .tensor[0]
                .tolist()
            )
            ids.append(ann["person_id"])
        org_boxes = np.array(boxes, dtype=np.float32)
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs(aug_input)
        aug_img = aug_input.image
        h, w = aug_img.shape[:2]
        aug_boxes = aug_input.boxes
        img_t = self.totensor(aug_img.copy())
        return {
            "image": img_t,
            "instances": Instances(
                (h, w),
                file_name=img_path,
                image_id=img_dict["image_id"],
                gt_boxes=Boxes(aug_boxes, BoxMode.XYXY_ABS),
                gt_pids=torch.tensor(ids, dtype=torch.int64),
                gt_classes=torch.zeros(len(ids), dtype=torch.int64),
                org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                org_gt_boxes=Boxes(org_boxes, BoxMode.XYXY_ABS),
            ),
        }


