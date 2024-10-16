from collections import OrderedDict
import pickle as pkl
import sys
import torch
# refer to mmdet swin_converter
def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x
def swin_to_mmdet(k,v):
        if k.startswith('head'):
            return k,v
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k
        new_k='backbone.' + new_k
        return new_k,new_v

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if 'state_dict' in obj:
        obj = obj['state_dict']
    elif 'model' in obj:
        obj = obj['model']
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        # 1st convert to mmdet format
        mid_k,new_v=swin_to_mmdet(k,obj.pop(old_k))
        # 2nd convert to solider in this repo
        if "patch_embed" in mid_k:
            new_k = mid_k.replace("backbone.patch_embed", "backbone")
            new_k=new_k.replace("projection","proj")
        else:
            new_k = mid_k.replace("backbone", "swin")
            new_k=new_k.replace("swin.norm.","swin.norm3.")
        newmodel[new_k] = new_v.detach().numpy()
        print(old_k, "->", mid_k,"->",new_k)
    res = {"model": newmodel, "__author__": "swin", "matching_heuristics": False}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())