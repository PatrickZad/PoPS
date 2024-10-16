# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import division
from typing import Any, List, Tuple, Optional
import torch
from torch import device
from torch.nn import functional as F
from torch import Tensor
from math import ceil


def _as_tensor(x: Tuple[int, int]) -> torch.Tensor:
    """
    An equivalent of `torch.as_tensor`, but works under tracing if input
    is a list of tensor. `torch.as_tensor` will record a constant in tracing,
    but this function will use `torch.stack` instead.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x)
    if isinstance(x, (list, tuple)) and all([isinstance(t, torch.Tensor) for t in x]):
        return torch.stack(x)
    return torch.as_tensor(x)


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
        mask: Optional[Tensor] = None,
    ):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
            mask: indicating padding area (True for padding)
        """
        assert image_sizes is not None or mask is not None
        self.tensor = tensor
        if image_sizes is None:
            self.mask = mask
            inv_mask = 1 - mask.int()
            self.image_sizes = torch.stack(
                [inv_mask.sum(dim=1)[:, 0], inv_mask.sum(dim=2)[:, 0]], dim=-1
            ).tolist()  # B x 2 hw
        elif mask is None:
            self.image_sizes = image_sizes  # unpadded image sizes
            mask = torch.ones(
                (tensor.shape[0], tensor.shape[2], tensor.shape[3]),
                dtype=torch.bool,
                device=tensor.device,
            )
            for bi in range(tensor.shape[0]):
                mask[bi, : image_sizes[bi][0], : image_sizes[bi][1]] = 0
            self.mask = mask.bool()

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        mask = self.mask
        cast_mask = mask.to(device)
        return ImageList(cast_tensor, self.image_sizes, cast_mask)

    @property
    def device(self) -> device:
        return self.tensor.device

    def decompose(self):
        return self.tensors, self.mask

    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad

        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [_as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)) // stride * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: List[int] = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [
                0,
                max_size[-1] - image_size[1],
                0,
                max_size[-2] - image_size[0],
            ]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(
                0
            )
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)

    def __repr__(self):
        return str(self.tensors)

    def select_by_indices(self, indices):
        return ImageList(
            torch.stack([self.tensor[i] for i in indices]),
            [self.image_sizes[i] for i in indices],
        )