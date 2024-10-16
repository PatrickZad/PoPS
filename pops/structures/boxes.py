# Copyright (c) Facebook, Inc. and its affiliates.
import math
import numpy as np
from enum import IntEnum, unique
from typing import List, Tuple, Union
import torch
from torch import device

from pops.utils.env import TORCH_VERSION

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x


# NOTE abs or rel is considered by the model
@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    XYXY_REL = 1
    """
    (x0, y0, x1, y1) in floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 2
    XYWH_REL = 3
    """
    (x0, y0, w, h) in floating points coordinates.
    """
    XYWHA_ABS = 4
    XYWHA_REL = 5
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """
    CCWH_ABS = 6
    CCWH_REL = 7
    """
    (xc, yc, w, h) in floating points coordinates.
    """

    @staticmethod
    def convert(
        box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode", img_size=None
    ) -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()
        if to_mode in ABS_MODE and from_mode in REL_MODE:
            h, w = img_size[0], img_size[1]
            factor = torch.tensor([[w, h, w, h]], device=arr.device)
            arr[:, :4] = arr[:, :4] * factor
            from_mode = from_mode - 1
        elif to_mode in REL_MODE and from_mode in ABS_MODE:
            h, w = img_size[0], img_size[1]
            factor = torch.tensor([[w, h, w, h]], device=arr.device)
            arr[:, :4] = arr[:, :4] / factor
            from_mode = from_mode + 1
        if from_mode == to_mode:
            return arr

        if (from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS) or (
            from_mode == BoxMode.XYWHA_REL and to_mode == BoxMode.XYXY_REL
        ):
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif (from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS) or (
            from_mode == BoxMode.XYWH_REL and to_mode == BoxMode.XYWHA_REL
        ):
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if (to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS) or (
                to_mode == BoxMode.XYXY_REL and from_mode == BoxMode.XYWH_REL
            ):
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif (from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS) or (
                from_mode == BoxMode.XYXY_REL and to_mode == BoxMode.XYWH_REL
            ):
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            elif (to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.CCWH_ABS) or (
                to_mode == BoxMode.XYXY_REL and from_mode == BoxMode.CCWH_REL
            ):
                arr[:, :2] = arr[:, :2] - arr[:, 2:] / 2
                arr[:, 2:] = arr[:, :2] + arr[:, 2:]
            elif (from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.CCWH_ABS) or (
                from_mode == BoxMode.XYXY_REL and to_mode == BoxMode.CCWH_REL
            ):
                arr[:, 2:] = arr[:, 2:] - arr[:, :2]
                arr[:, :2] = arr[:, :2] + arr[:, 2:] / 2
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


ABS_MODE = [BoxMode.CCWH_ABS, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS]
REL_MODE = [BoxMode.CCWH_REL, BoxMode.XYWH_REL, BoxMode.XYWHA_REL, BoxMode.XYXY_REL]


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor, box_mode=BoxMode.XYXY_ABS):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = (
            tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        )
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.dim() == 1:
            tensor = tensor[None]
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor
        self.box_mode = box_mode

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def convert_mode(self, to_mode, img_size=None):
        box_t = BoxMode.convert(self.tensor, self.box_mode, to_mode, img_size)
        return Boxes(box_t, to_mode)

    @_maybe_jit_unused
    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        if self.box_mode in ABS_MODE:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.XYXY_ABS)
        else:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.XYXY_REL)
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        if self.box_mode in ABS_MODE:
            mode_temp = BoxMode.XYXY_ABS
        else:
            mode_temp = BoxMode.XYXY_REL
        box = BoxMode.convert(self.tensor, self.box_mode, mode_temp)
        x1 = box[:, 0].clamp(min=0, max=w)
        y1 = box[:, 1].clamp(min=0, max=h)
        x2 = box[:, 2].clamp(min=0, max=w)
        y2 = box[:, 3].clamp(min=0, max=h)
        box = torch.stack((x1, y1, x2, y2), dim=-1)
        box = BoxMode.convert(box, mode_temp, self.box_mode)
        self.tensor = box

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        if self.box_mode in ABS_MODE:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.XYXY_ABS)
        else:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.XYXY_REL)
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert (
            b.dim() == 2
        ), "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(
        self, box_size: Tuple[int, int], boundary_threshold: int = 0
    ) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        if self.box_mode in ABS_MODE:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.XYXY_ABS)
        else:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.XYXY_REL)
        height, width = box_size
        inds_inside = (
            (box[..., 0] >= -boundary_threshold)
            & (box[..., 1] >= -boundary_threshold)
            & (box[..., 2] < width + boundary_threshold)
            & (box[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        if self.box_mode in ABS_MODE:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.CCWH_ABS)
        else:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.CCWH_REL)
        return box[:, :2]

    def get_sizes(self) -> torch.Tensor:
        if self.box_mode in ABS_MODE:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.CCWH_ABS)
        else:
            box = BoxMode.convert(self.tensor, self.box_mode, BoxMode.CCWH_REL)
        return torch.cat([box[:, -1:], box[:, -2:-1]], dim=-1)

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    @_maybe_jit_unused
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    if boxes1.box_mode in ABS_MODE:
        boxes1 = BoxMode.convert(boxes1.tensor, boxes1.box_mode, BoxMode.XYXY_ABS)
    else:
        boxes1 = BoxMode.convert(boxes1.tensor, boxes1.box_mode, BoxMode.XYXY_REL)
    if boxes2.box_mode in ABS_MODE:
        boxes2 = BoxMode.convert(boxes2.tensor, boxes2.box_mode, BoxMode.XYXY_ABS)
    else:
        boxes2 = BoxMode.convert(boxes2.tensor, boxes2.box_mode, BoxMode.XYXY_REL)
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Similar to :func:`pariwise_iou` but compute the IoA (intersection over boxes2 area).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoA, sized [N,M].
    """
    if boxes1.box_mode in ABS_MODE:
        boxes1 = BoxMode.convert(boxes1.tensor, boxes1.box_mode, BoxMode.XYXY_ABS)
    else:
        boxes1 = BoxMode.convert(boxes1.tensor, boxes1.box_mode, BoxMode.XYXY_REL)
    if boxes2.box_mode in ABS_MODE:
        boxes2 = BoxMode.convert(boxes2.tensor, boxes2.box_mode, BoxMode.XYXY_ABS)
    else:
        boxes2 = BoxMode.convert(boxes2.tensor, boxes2.box_mode, BoxMode.XYXY_REL)
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    ioa = torch.where(
        inter > 0, inter / area2, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa


def pairwise_point_box_distance(points: torch.Tensor, boxes: Boxes):
    """
    Pairwise distance between N points and M boxes. The distance between a
    point and a box is represented by the distance from the point to 4 edges
    of the box. Distances are all positive when the point is inside the box.

    Args:
        points: Nx2 coordinates. Each row is (x, y)
        boxes: M boxes

    Returns:
        Tensor: distances of size (N, M, 4). The 4 values are distances from
            the point to the left, top, right, bottom of the box.
    """
    x, y = points.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
    if boxes.box_mode in ABS_MODE:
        boxes_t = BoxMode.convert(boxes.tensor, boxes.box_mode, BoxMode.XYXY_ABS)
    else:
        boxes_t = BoxMode.convert(boxes.tensor, boxes.box_mode, BoxMode.XYXY_REL)
    x0, y0, x1, y1 = boxes_t.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
    return torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)


def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes that have the same number of boxes.
    Similar to :func:`pairwise_iou`, but computes only diagonal elements of the matrix.

    Args:
        boxes1 (Boxes): bounding boxes, sized [N,4].
        boxes2 (Boxes): same length as boxes1
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    if boxes1.box_mode in ABS_MODE:
        boxes1 = BoxMode.convert(boxes1.tensor, boxes1.box_mode, BoxMode.XYXY_ABS)
    else:
        boxes1 = BoxMode.convert(boxes1.tensor, boxes1.box_mode, BoxMode.XYXY_REL)
    if boxes2.box_mode in ABS_MODE:
        boxes2 = BoxMode.convert(boxes2.tensor, boxes2.box_mode, BoxMode.XYXY_ABS)
    else:
        boxes2 = BoxMode.convert(boxes2.tensor, boxes2.box_mode, BoxMode.XYXY_REL)
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou


def pairwise_giou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    if boxes1.box_mode in ABS_MODE:
        boxes1 = boxes1.convert_mode(BoxMode.XYXY_ABS)
    else:
        boxes1 = boxes1.convert_mode(BoxMode.XYXY_REL)
    if boxes2.box_mode in ABS_MODE:
        boxes2 = boxes2.convert_mode(BoxMode.XYXY_ABS)
    else:
        boxes2 = boxes2.convert_mode(BoxMode.XYXY_REL)
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)
    union = area1[:, None] + area2 - inter
    iou = torch.where(
        inter > 0,
        inter / union,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )

    lt = torch.min(boxes1.tensor[:, None, :2], boxes2.tensor[:, :2])
    rb = torch.max(boxes1.tensor[:, None, 2:], boxes2.tensor[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area_c = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area_c - union) / area_c
