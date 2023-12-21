import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='DGSM_Swin',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default="5")

# Data specifications
parser.add_argument('--data_root', type=str, default='/home/huangtao/Data/Spectral_SCI/Data/ZiyiMeng/Cave_1024/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='/home/huangtao/Code/Spectral_SCI/TPAMI_SCI_Code/Code2/exp/DGSM_Swin_T2P4/result/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='DGSM_Swin_T2P4', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default='/home/huangtao/Code/Spectral_SCI/TPAMI_SCI_Code/Code2/exp/DGSM_Swin_T2P4/model/pretrained_model.pth', help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='Y',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default=None,
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=1, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=500, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[150], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=4000, help='the number of samples per epoch')
parser.add_argument("--patch_size", type=int, default=256, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)

opt = parser.parse_args()
template.set_template(opt)

# dataset
opt.data_path = f"{opt.data_root}"
opt.mask_path = f"/home/huangtao/Data/Spectral_SCI/Data/"
opt.test_path = f"/home/huangtao/Data/Spectral_SCI/Data/Testing_data/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
