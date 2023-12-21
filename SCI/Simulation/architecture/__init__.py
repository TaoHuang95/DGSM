import torch
import torch.nn as nn

from .DGSM_Swin import DGSM_Swin


def model_generator(method, pretrained_model_path=None):
    if method == 'DGSM_Swin_T2P4':
        model = DGSM_Swin(Ch=28, stages=2, img_size=256, in_chans=28,
                          embed_dim=64, depths=[4, 4, 4, 4, 4], num_heads=[4, 4, 4, 4, 4],
                          window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv')
    elif method == 'DGSM_Swin_T4P4':
        model = DGSM_Swin(Ch=28, stages=4, img_size=256, in_chans=28,
                          embed_dim=64, depths=[4, 4, 4, 4, 4], num_heads=[4, 4, 4, 4, 4],
                          window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv')
    elif method == 'DGSM_Swin_T3P6':
        model = DGSM_Swin(Ch=28, stages=3, img_size=256, in_chans=28,
                          embed_dim=64, depths=[6, 6, 6, 6, 6], num_heads=[4, 4, 4, 4, 4],
                          window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                          use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv')
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    return model
