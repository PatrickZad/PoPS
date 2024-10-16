import contextlib
import io
import logging
import numpy.random as npr
from fvcore.common.timer import Timer
from os.path import join as opj
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = [
    "load_coco_ch",
]
subsets = ["train", "val"]

error_imgs = ["000000550395.jpg"]  # raises reg loss NaN

def _load_coco_person(dataset_dir, subset="train", allow_crowd=True):
    from pycocotools.coco import COCO

    assert subset in subsets
    json_file = opj(dataset_dir, "annotations", "instances_{}2017.json".format(subset))
    img_root = opj(dataset_dir, "{}2017".format(subset))
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}

    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    # filter out none-person images
    anns_person = []
    img_ids_person = []
    for img_id in img_ids:
        org_ann = coco_api.imgToAnns[img_id]
        cur_anns = []
        for ann_dict in org_ann:
            if ann_dict["category_id"] == 1:  # person
                if allow_crowd:
                    cur_anns.append(ann_dict)
                elif ann_dict["iscrowd"] == 0:
                    cur_anns.append(ann_dict)
        if len(cur_anns) > 0:
            anns_person.append(cur_anns)
            img_ids_person.append(img_id)
    anns = anns_person
    img_ids = img_ids_person
    imgs = coco_api.loadImgs(img_ids)
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique!".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file)
    )

    dataset_dicts = []

    ann_keys = ["bbox"]  # "category_id"]
    next_pid = 0

    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        if img_dict["file_name"] in error_imgs:
            continue
        record["file_name"] = opj(img_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )
            obj["bbox"][2] += obj["bbox"][0]
            obj["bbox"][3] += obj["bbox"][1]
            obj["bbox_mode"] = BoxMode.XYXY_ABS

            obj["person_id"] = next_pid
            next_pid += 1
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts



def _load_crowd_human(
    dataset_dir, subset="train", allow_crowd=True, filter_esmall=False
):
    from pycocotools.coco import COCO

    assert subset in subsets
    json_file = opj(dataset_dir, "annotations", "{}.json".format(subset))
    img_root = opj(dataset_dir, "CrowdHuman_{}".format(subset))
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    if not allow_crowd:
        anns_person = []
        img_ids_person = []
        for img_id in img_ids:
            org_ann = coco_api.imgToAnns[img_id]
            cur_anns = []
            for ann_dict in org_ann:
                if ann_dict["iscrowd"] == 0:
                    cur_anns.append(ann_dict)
            if len(cur_anns) > 0:
                anns_person.append(cur_anns)
                img_ids_person.append(img_id)
        anns = anns_person
        img_ids = img_ids_person
        imgs = coco_api.loadImgs(img_ids)
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique!".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file)
    )

    dataset_dicts = []

    ann_keys = ["bbox"]  # , "category_id"]
    next_pid = 0

    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        record["file_name"] = opj(img_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )
            if filter_esmall:
                if obj["bbox"][2] * obj["bbox"][3] < 16**2:
                    continue
            obj["bbox"][2] += obj["bbox"][0]
            obj["bbox"][3] += obj["bbox"][1]
            obj["bbox_mode"] = BoxMode.XYXY_ABS

            obj["person_id"] = next_pid
            next_pid += 1
            objs.append(obj)
        if len(objs) == 0:
            # filter emplty
            continue
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def load_coco_ch(common_dir, subset="train", allow_crowd=True):
    coco_dicts = _load_coco_person(opj(common_dir, "coco"), subset, allow_crowd)
    ch_dicts = _load_crowd_human(opj(common_dir, "crowd_human"), subset, allow_crowd)
    return ch_dicts + coco_dicts

