import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import random

def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=None):
    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)


def psnr(img1, img2):
   mse = np.mean((img1/255. - img2/255.) ** 2)
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def PSNR_GPU(im_true, im_fake):
    im_true *= 255
    im_fake *= 255
    im_true = im_true.round()
    im_fake = im_fake.round()
    data_range = 255
    esp = 1e-12
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone()
    Ifake = im_fake.clone()
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum() / (C*H*W)
    psnr = 10. * np.log((data_range**2)/(err.data + esp)) / np.log(10.)
    return psnr


def PSNR_Nssr(im_true, im_fake):
    mse = ((im_true - im_fake)**2).mean()
    psnr = 10. * np.log10(1/mse)
    return psnr


def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:

            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

# load HSIs
def prepare_data_cave(path, file_list, file_num):
    HR_HSI = np.zeros((((512,512,28,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['data_slice'] / 65535.0
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

def prepare_data_KASIT(path, file_list, file_num):
    HR_HSI = np.zeros((((2704,3376,28,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI']
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI


def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    random.shuffle(pathlist)
    return pathlist

def A_torch(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At_torch(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_torch(inputs,step=1):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step)
    for i in range(nC):
        output[:, i, :, i*step:i*step+col] = inputs[:, i,:,:]
    return output.cuda()

def shift_back_torch(inputs, step=2):          # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,:] = inputs[:, i, :, step*i:step*i+col-(nC-1)*step]
    return output

def KL_Loss(U, S):
    l = 1 + 2 * S - U**2 - S.exp()**2
    return (-0.5 * l).mean()