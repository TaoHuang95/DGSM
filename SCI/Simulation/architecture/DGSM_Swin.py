import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .Swin_Trans_Block import *


class Resblock(nn.Module):
    def __init__(self, HBW):
        super(Resblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out


class DGSM_Swin(nn.Module):
    def __init__(self, Ch, stages, img_size=256, in_chans=3,
                 embed_dim=64, depths=[4, 4, 4, 4, 4], num_heads=[4, 4, 4, 4, 4],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., resi_connection='1conv',
                 **kwargs):
        super(DGSM_Swin, self).__init__()
        self.Ch = Ch
        self.s = stages
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.window_size = window_size
        self.input_resolution = [img_size / 1, img_size / 2, img_size / 4, img_size / 2, img_size / 1]
        self.Isdownsample = [True, True, False, False, False]
        self.Isupsample = [False, False, True, True, False]
        self.patch_norm = patch_norm
        self.embed_dim = embed_dim

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        ## Dense connection
        self.conv = nn.Conv2d(Ch, self.embed_dim, kernel_size=3, stride=1, padding=1)
        self.Den_con1 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.Den_con2 = nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, stride=1, padding=0)
        # self.Den_con3 = nn.Conv2d(self.embed_dim * 3, self.embed_dim, kernel_size=1, stride=1, padding=0)
        # self.Den_con4 = nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=1, stride=1, padding=0)
        # self.Den_con5 = nn.Conv2d(64 * 5, 64, kernel_size=1, stride=1, padding=0)
        # self.Den_con6 = nn.Conv2d(64 * 6, 64, kernel_size=1, stride=1, padding=0)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)

        self.ape = ape
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(img_size=img_size, embed_dim=embed_dim,
                                          norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(int(self.input_resolution[i_layer]),
                                           int(self.input_resolution[i_layer])),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=PatchMerging if self.Isdownsample[i_layer] else None,
                         upsample=PatchMerging if self.Isupsample[i_layer] else None,
                         resi_connection=resi_connection
                         )
            if self.Isdownsample[i_layer]:
                embed_dim = embed_dim * 2
            else:
                embed_dim = int(embed_dim * 0.5)
            self.layers.append(layer)
        self.short_cat_1 = nn.Linear(self.embed_dim * 4, self.embed_dim * 2, bias=False)
        self.short_cat_2 = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=False)
        self.norm = norm_layer(self.num_features)
        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.conv_U = nn.Conv2d(self.embed_dim, 28, 3, 1, 1)
        self.conv_W = nn.Sequential(nn.Conv2d(self.embed_dim, 28, 3, 1, 1))

        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)

        self._initialize_weights()
        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def recon(self, res1, res2, Xt, i):
        if i == 0:
            delta = self.delta_0
        elif i == 1:
            delta = self.delta_1
        # elif i == 2:
        #     delta = self.delta_2
        # elif i == 3:
        #     delta = self.delta_3
        # elif i == 4:
        #     delta = self.delta_4
        # elif i == 5:
        #     delta = self.delta_5

        Xt = Xt - 2 * delta * (res1 + res2)
        return Xt

    def y2x(self, y):
        ##  Spilt operator
        sz = y.size()
        x = torch.zeros([sz[0], 28, sz[2], sz[2]]).cuda()
        for t in range(28):
            temp = y[:, :, :, 0 + 2 * t: sz[2] + 2 * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x

    def x2y(self, x):
        ##  Shift and Sum operator
        sz = x.size()
        y = torch.zeros([sz[0], 1, sz[2], sz[2] + 2 * 27]).cuda()
        for t in range(28):
            y[:, :, :, 0 + 2 * t: sz[2] + 2 * t] = x[:, t, :, :].unsqueeze(1) + y[:, :, :, 0 + 2 * t: sz[2] + 2 * t]
        return y

    def forward(self, y, mask):
        ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
        Xt = self.y2x(y)

        feature_list = []

        for i in range(0, self.s):
            Feat_list = []

            AXt = self.x2y(Xt * mask.repeat(1, 28, 1, 1))

            AXt_y = self.y2x(AXt - y)

            Res1 = AXt_y * mask.repeat(1, 28, 1, 1)

            feat = self.conv(Xt)
            shortcut = feat

            if i == 0:
                feature_list.append(feat)
                fufeat = self.Den_con1(feat)  # 256 x 256 x 64
            elif i == 1:
                feature_list.append(feat)
                fufeat = self.Den_con2(torch.cat(feature_list, 1))
            # elif i == 2:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con3(torch.cat(feature_list, 1))
            # elif i == 3:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con4(torch.cat(feature_list, 1))
            # elif i == 4:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con5(torch.cat(feature_list, 1))
            # elif i == 5:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con6(torch.cat(feature_list, 1))

            # shortcut = fufeat
            ## generate U and w
            fufeat_size = (fufeat.shape[2], fufeat.shape[3])
            x_feat = self.patch_embed(fufeat)
            if self.ape:
                x_feat = x_feat + self.absolute_pos_embed
            x_feat = self.pos_drop(x_feat)

            for layer in self.layers:

                sz = x_feat.shape[1]
                x_size = (int(sz ** (0.5)), int(sz ** (0.5)))
                x_feat, short_Feat = layer(x_feat, x_size)
                Feat_list.append(short_Feat)
                if len(Feat_list) == 3:
                    x_feat = self.short_cat_1(torch.cat((Feat_list[1], x_feat), dim=2))
                elif len(Feat_list) == 4:
                    x_feat = self.short_cat_2(torch.cat((Feat_list[0], x_feat), dim=2))

            x_feat = self.norm(x_feat)  # B L C
            x_feat = self.patch_unembed(x_feat, fufeat_size)
            x_feat = x_feat + shortcut

            U = self.conv_U(x_feat)
            W = torch.exp(self.conv_W(x_feat))
            # W = self.conv_W(x_feat) # S2L4

            ## w * (x − u)
            Res2 = (Xt - U).mul(W)

            ## Reconstructing HSIs
            Xt = self.recon(Res1, Res2, Xt, i)

        return Xt


if __name__ == '__main__':
    from thop import profile
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    upscale = 4
    window_size = 8
    height = 256
    width = 256
    model = DGSM_Swin(Ch=28, stages=2, img_size=256, in_chans=28,
                      embed_dim=64, depths=[4, 4, 4, 4, 4], num_heads=[4, 4, 4, 4, 4],
                      window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                      drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                      norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                      use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv')

    model = model.cuda()
    print(model)

    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    x = torch.randn((1, 1, height, width + 54)).to(device)
    m = torch.randn((1, 1, height, width)).to(device)
    with torch.no_grad():
        x = model(x, m)
    print(x.shape)
