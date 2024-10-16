from scipy.io import loadmat
from os.path import join as opj

import numpy as np
from pops.structures import BoxMode

from tqdm import tqdm
import logging

logger = logging.getLogger(__name__) 
__all__ = ["subsets", "load_prw"]
subsets = ("Train", "Query", "Gallery")
id_remap_dict = {}
for i in range(1, 479):
    id_remap_dict[i] = i - 1
for i in range(480, 484):
    id_remap_dict[i] = i - 2
for i in range(484, 932):
    id_remap_dict[i] = i
id_remap_dict[932] = 482
id_remap_dict[479] = 483
id_remap_dict[933] = 932
id_remap_dict[-2] = -1


def load_prw(dataset_dir, subset=subsets[0]):
    """
    Returns:
        train/test set meta dict
            [
                {
                    "file_name": image paths,
                    "image_id": image name,
                    "annotations":
                        [
                            {
                                "bbox": person xyxy_abs boxes,
                                "bbox_mode": format of bbox
                                "person_id":  person id
                            },...
                        ]
                },
                ...
            ]
        or
        query meta dict
            [
                {
                    "query":
                        {
                            "file_name": image paths,
                            "image_id": image name,
                            "annotations":
                                [
                                    {
                                        "bbox": person xyxy_abs boxes,
                                        "bbox_mode": format of bbox
                                        "person_id":  person id
                                    }
                                ]
                        }
                }
            ]
    """
    assert subset in subsets

    if subset == subsets[0]:
        return _load_subset_full_anno(dataset_dir, "train")
    elif subset == subsets[1]:
        q_dicts = []
        logger.info("Loading PRW query info :")
        with open(opj(dataset_dir, "query_info.txt"), "r") as qf:
            qinfo_lines = qf.readlines()
        with tqdm(total=len(qinfo_lines)) as pbar:
            for line in qinfo_lines:
                line = line.strip()
                if len(line) > 0:
                    infos = line.split(" ")
                    pid = int(infos[0])
                    box = np.array(infos[1:-1], dtype=np.float32)
                    box[2:] = box[:2] + box[2:]
                    img_path = opj(dataset_dir, "frames", infos[-1] + ".jpg")
                    q_dicts.append(
                        {
                            "query": {
                                "file_name": img_path,
                                "image_id": infos[-1] + ".jpg",
                                "annotations": [
                                    {
                                        "bbox": box,
                                        "bbox_mode": BoxMode.XYXY_ABS,
                                        "person_id": get_resort_id(pid),
                                    }
                                ],
                            }
                        }
                    )
                pbar.update(1)
        return q_dicts
    else:
        return _load_subset_full_anno(dataset_dir, "test")


def get_resort_id(labeled_id):
    global id_remap_dict
    """global id_remap_dict
    global id_next
    if labeled_id < 0:
        return -1
    elif labeled_id in id_remap_dict.keys():

        return id_remap_dict[labeled_id]
    else:
        resort_id = id_next
        id_remap_dict[labeled_id] = resort_id
        id_next += 1
        return resort_id"""
    return id_remap_dict[labeled_id]


def _load_subset_full_anno(dataset_dir, subset_name):

    sub_f_mat = loadmat(opj(dataset_dir, "frame_" + subset_name + ".mat"))[
        "img_index_" + subset_name
    ].squeeze()
    sub_img_fns = sub_f_mat.tolist()
    subset_dicts = []
    logger.info("Loading PRW {} :".format(subset_name))
    with tqdm(total=len(sub_img_fns)) as pbar:
        for img_fn in sub_img_fns:
            img_fn = img_fn[0]
            img_name = img_fn + ".jpg"
            img_anno_mat = loadmat(opj(dataset_dir, "annotations", img_name + ".mat"))
            ks = sorted(img_anno_mat.keys())
            img_anno = img_anno_mat[ks[-1]]
            boxes = img_anno[:, 1:].copy().astype(np.float32)  # n x 4 xywh-boxes
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
            ids = img_anno[:, 0].astype(np.int32).copy()
            subset_dicts.append(
                {
                    "file_name": opj(dataset_dir, "frames", img_name),
                    "image_id": img_name,
                    "annotations": [
                        {
                            "bbox": boxes[i],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "person_id": get_resort_id(ids[i]),
                        }
                        for i in range(boxes.shape[0])
                    ],
                }
            )
            pbar.update(1)
    return subset_dicts
