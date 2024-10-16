import logging

from os.path import join as opj

import numpy as np
from pops.structures import BoxMode
from scipy.io import loadmat
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

__all__ = ["load_cuhk_sysu", "subset_names"]
subset_names = (
    [
        "Train",
    ]
    + ["TestG" + gs for gs in ("50", "100", "500", "1000", "2000", "4000")]
    + ["Gallery"]
)
id_map_dict = {}
next_id = 0


def _set_box_pid(boxes, box, pids, pid):
    # nid = _get_map_id(pid)
    for i in range(boxes.shape[0]):
        if np.all(boxes[i] == box):
            pids[i] = pid
            return
    logging.warning("Person: %s, box: %s cannot find in images." % (pid, box))


def load_cuhk_sysu(dataset_dir, subset="Train"):
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
                                ],
                        },
                    "gallery":
                        [
                            {
                                "file_name": image paths,
                                "image_id": image name,
                                "annotations":
                                    [
                                        {
                                            "bbox": person xyxy_abs boxes,
                                            "bbox_mode": format of bbox,                                                "person_id":  person id
                                        }
                                    ],
                            },
                            ...
                        ]
                }
            ]
    """
    global cuhk_sysu_test_subs
    assert subset in subset_names
    # init all labels
    all_imgs = loadmat(opj(dataset_dir, "annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    img_boxes_dict = {}
    img_pids_dict = {}
    all_img_names = []
    for img_name, _, boxes in all_imgs:
        img_name = str(img_name[0])
        boxes = np.asarray([b[0] for b in boxes[0]])
        boxes = boxes.reshape(boxes.shape[0], 4)
        valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
        assert valid_index.size > 0, "Warning: %s has no valid boxes." % img_name
        boxes = boxes[valid_index]
        img_boxes_dict[img_name] = boxes.astype(np.int32)
        img_pids_dict[img_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)
        all_img_names.append(img_name)
    max_train_id = 5531
    if subset == subset_names[0]:
        # set pids of boxes ---- train set
        train_mat = loadmat(opj(dataset_dir, "annotation/test/train_test/Train.mat"))
        train_anno = train_mat["Train"].squeeze()

        logger.info("Loading Training Person annotations :")

        with tqdm(total=train_anno.shape[0]) as pbar:
            for index, item in enumerate(train_anno):
                scenes = item[0, 0][2].squeeze()
                # pid_str = str(item[0, 0][0][0])
                for img_name, box, _ in scenes:
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(
                        img_boxes_dict[img_name], box, img_pids_dict[img_name], index
                    )
                pbar.update(1)
        # build anno dict
        test_img_names = loadmat(
            opj(
                dataset_dir,
                "annotation",
                "pool.mat",
            )
        )["pool"].squeeze()
        test_img_names = [img_name_arr[0] for img_name_arr in test_img_names.tolist()]
        train_img_names = list(set(all_img_names) - set(test_img_names))
        train_set_dicts = []
        for img_name in train_img_names:
            boxes = img_boxes_dict[img_name].copy().astype(np.float32)
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
            ids = img_pids_dict[img_name]
            train_set_dicts.append(
                {
                    "file_name": opj(dataset_dir, "Image", "SSM", img_name),
                    "image_id": img_name,
                    "annotations": [
                        {
                            "bbox": boxes[i],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "person_id": ids[i],
                        }
                        for i in range(boxes.shape[0])
                    ],
                }
            )
        return train_set_dicts
    else:
        if subset == subset_names[-1]:
            load = subset_names[1]
        else:
            load = subset
        test_mat = loadmat(
            opj(
                dataset_dir,
                "annotation/test/train_test",
                load + ".mat",
            )
        )
        test_anno = test_mat[load].squeeze()  # .tolist()
        logger.info("Loading {} annotations :".format(load))
        test_query_gallery = []
        with tqdm(total=test_anno.shape[0]) as pbar:
            for index, item in enumerate(test_anno):
                # test_id = max_train_id + 1 + index
                # query
                qimg_name = str(item["Query"][0, 0][0][0])
                # qid_str = max_train_id+1+index
                qbox = item["Query"][0, 0][1].squeeze().astype(np.int32)
                _set_box_pid(
                    img_boxes_dict[qimg_name],
                    qbox,
                    img_pids_dict[qimg_name],
                    index + max_train_id + 1,
                )
                # gallery
                gallery = item["Gallery"].squeeze()
                gallery_dicts = []
                for gimg_name, gbox, _ in gallery:
                    gimg_name = str(gimg_name[0])

                    gallery_img_path = opj(dataset_dir, "Image", "SSM", gimg_name)
                    if gbox.size == 0:
                        gallery_dicts.append(
                            {
                                "file_name": gallery_img_path,
                                "image_id": gimg_name,
                                "annotations": [
                                    {
                                        "bbox": np.array([], dtype=np.float32),
                                        "bbox_mode": BoxMode.XYXY_ABS,
                                        "person_id": None,
                                    }
                                ],
                            }
                        )
                        continue
                    gbox = gbox.squeeze().astype(np.int32)
                    _set_box_pid(
                        img_boxes_dict[gimg_name],
                        gbox,
                        img_pids_dict[gimg_name],
                        index + max_train_id + 1,
                    )
                    gbox_xyxy = gbox.copy().astype(np.float32)
                    gbox_xyxy[2:] = gbox_xyxy[:2] + gbox_xyxy[2:]
                    gallery_dicts.append(
                        {
                            "file_name": gallery_img_path,
                            "image_id": gimg_name,
                            "annotations": [
                                {
                                    "bbox": gbox_xyxy,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "person_id": index + max_train_id + 1,
                                }
                            ],
                        }
                    )
                # to return
                box = qbox.copy().astype(np.float32)
                box[2:] = box[:2] + box[2:]
                test_query_gallery.append(
                    {
                        "query": {
                            "file_name": opj(dataset_dir, "Image", "SSM", qimg_name),
                            "image_id": qimg_name,
                            "annotations": [
                                {
                                    "bbox": box,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "person_id": index + max_train_id + 1,
                                }
                            ],
                        },
                        "gallery": gallery_dicts,
                    }
                )
                pbar.update(1)
        # build anno dict
        if subset in subset_names[1:-1]:
            return test_query_gallery
        else:
            test_img_names = loadmat(
                opj(
                    dataset_dir,
                    "annotation",
                    "pool.mat",
                )
            )["pool"].squeeze()
            test_img_names = [
                img_name_arr[0] for img_name_arr in test_img_names.tolist()
            ]
            logger.info("Loading Gallery annotations :".format(load))
            gallery_set_dicts = []
            with tqdm(total=len(test_img_names)) as pbar:
                for img_name in test_img_names:
                    boxes = img_boxes_dict[img_name].copy().astype(np.float32)
                    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
                    ids = img_pids_dict[img_name]
                    gallery_set_dicts.append(
                        {
                            "file_name": opj(dataset_dir, "Image", "SSM", img_name),
                            "image_id": img_name,
                            "annotations": [
                                {
                                    "bbox": boxes[i],
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "person_id": ids[i],
                                }
                                for i in range(boxes.shape[0])
                            ],
                        }
                    )
                    pbar.update(1)
            return gallery_set_dicts
