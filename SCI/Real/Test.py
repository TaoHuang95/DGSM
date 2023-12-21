import torch
import torch.utils.data as tud
import os
import argparse
from Utils import *
import scipy.io as sio
import numpy as np
from Dataset import dataset
from torch.autograd import Variable
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"


parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
parser.add_argument('--data_path', default='/home/huangtao/Data/Spectral_SCI/Data/Real_testing/', type=str,help='path of data')
parser.add_argument('--mask_path', default='/home/huangtao/Data/Spectral_SCI/Data/Real_mask_660/mask.mat', type=str,help='path of mask')

parser.add_argument("--size", default=256, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=10, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
opt = parser.parse_args()
print(opt)

def prepare_data(path, file_num):
    HR_HSI = np.zeros((((660,714,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + 'scene' + str(idx+1) + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,idx] = data['meas_real']
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI


mask = sio.loadmat(opt.mask_path)
mask = mask['mask']
mask  = torch.FloatTensor(mask).unsqueeze(0)
HR_HSI = prepare_data(opt.data_path, 5)

# dataset = dataset(opt, HR_HSI)
# loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)

for i in range(55,151,10):  # number of model.pth
    model = torch.load("/home/huangtao/Code/Spectral_SCI/Transformer/Real_PAMI/Checkpoint/model_%03d.pth"%i)
    model = model.eval()
    model = dataparallel(model, 1)
    psnr_total = 0
    k = 0
    for j in range(5):
        with torch.no_grad():
            meas = HR_HSI[:,:,j]
            meas = meas / meas.max() * 0.8

            meas = torch.FloatTensor(meas).unsqueeze(2).permute(2, 0, 1)
            input = meas.unsqueeze(0)
            input = Variable(input)
            input = input.cuda()
            mask = Variable(mask)
            mask = mask.cuda()
            out = model(input,mask)
            result = out
            result = result.clamp(min=0., max=1.)

        k = k + 1
        model_dir = '/home/huangtao/Code/Spectral_SCI/Transformer/Real_PAMI/Result/' + str(i)
        if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
            os.makedirs(model_dir)
        res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
        save_path = '/home/huangtao/Code/Spectral_SCI/Transformer/Real_PAMI/Result/' + str(i) + '/' + str(j + 1) + '.mat'
        sio.savemat(save_path, {'res':res})

    print(k)
