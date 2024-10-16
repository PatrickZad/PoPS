# Copyright (c) Facebook, Inc. and its affiliates.

# TODO figure out _C error

from .deform_conv import DeformConv, ModulatedDeformConv, DeformConvPack
from .mask_ops import paste_masks_in_image
from .nms import batched_nms, batched_nms_rotated, nms, nms_rotated
from .roi_align import ROIAlign, roi_align

from .roi_align_rotated import ROIAlignRotated, roi_align_rotated
from .blocks import CNNBlockBase, DepthwiseSeparableConv2d
from .aspp import ASPP
from .batch_norm import (
    FrozenBatchNorm2d,
    get_norm,
    get_norm1d,
    NaiveSyncBatchNorm,
    CycleBatchNormList,
)
from .wrappers import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    cat,
    interpolate,
    Linear,
    nonzero_tuple,
    cross_entropy,
    move_device_like,
)
from .shape_spec import ShapeSpec
from .pos_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from .drop import DropPath, DropBlock2d
import torch.nn as nn
import copy
from .losses import ciou_loss, diou_loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


__all__ = [k for k in globals().keys() if not k.startswith("_")]
