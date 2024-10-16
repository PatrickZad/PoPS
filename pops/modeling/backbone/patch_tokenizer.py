""" Modified from timm

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch.nn as nn
import torch.nn.init as init
from torch.nn.init import _calculate_fan_in_and_fan_out
import collections
from itertools import repeat

try:
    from torch._six import container_abcs
except:
    import collections.abc as container_abcs
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from pops.layers import Conv2d, ShapeSpec

class PatchEmbed_(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        pretrain_img_size = to_2tuple(pretrain_img_size)
        patch_size = to_2tuple(patch_size)
        self.pretrain_img_size = pretrain_img_size
        self.patch_size = patch_size
        self.pretrain_grid_size = (
            pretrain_img_size[0] // patch_size[0],
            pretrain_img_size[1] // patch_size[1],
        )
        self.num_patches = self.pretrain_grid_size[0] * self.pretrain_grid_size[1]
        self.flatten = flatten

        self.proj = Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        # TODO: consider padding
        assert (H % self.patch_size[0] == 0) and (W % self.patch_size[1] == 0)
        x = self.proj(x)

        _, C_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        if not self.flatten:
            x = x.transpose(1, 2).contiguous().view(B, C_new, H_new, W_new)
        return x


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise NotImplementedError("Unsupported mode: {}!".format(mode))

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        init.trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")



def parse(x, n):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, n))


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x
    return tuple(repeat(x, 2))


# swin
class PatchEmbed(PatchEmbed_, Backbone):
    def __init__(self, *args, **kws):
        PatchEmbed_.__init__(self, *args, **kws)
        self._out_features = ["out"]

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return self.patch_size[0]

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self.embed_dim,
                stride=self.patch_size[0],
            )
            for name in self._out_features
        }

    def forward(self, x):
        ret = super().forward(x)
        return {self._out_features[-1]: ret}


@BACKBONE_REGISTRY.register()
def build_patch_embed(cfg, input_shape):
    pt_cfg = cfg.MODEL.PATCH_EMBED
    return PatchEmbed(
        pt_cfg.PRETRAIN_IMG_SIZE,
        pt_cfg.PATCH_SIZE,
        input_shape.channels,
        pt_cfg.EMBED_DIM,
        norm_layer=None,
        flatten=False,
    )
@BACKBONE_REGISTRY.register()
def build_patch_embed_ln(cfg, input_shape):
    pt_cfg = cfg.MODEL.PATCH_EMBED
    return PatchEmbed(
        pt_cfg.PRETRAIN_IMG_SIZE,
        pt_cfg.PATCH_SIZE,
        input_shape.channels,
        pt_cfg.EMBED_DIM,
        norm_layer=nn.LayerNorm,
        flatten=False,
    )
