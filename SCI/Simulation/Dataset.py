import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio
import os

class dataset(tud.Dataset):
    def __init__(self, opt, HSI, isTrain=True):
        super(dataset, self).__init__()
        self.opt = opt
        self.isTrain = isTrain
        self.size = opt.patch_size
        self.path = opt.data_path
        if self.isTrain == True:
            self.num = opt.epoch_sam_num
        else:
            self.num = 10
        self.HSI = HSI

        ## load mask
        mask = sio.loadmat(opt.mask_path + 'mask.mat')
        self.mask = mask['mask']

    def shift_back(self, inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
        [c, row, col] = inputs.shape
        nC = 28
        output = torch.zeros(nC, row, col - (nC - 1) * step).cuda().float()
        for i in range(nC):
            output[i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
        return output

    def shift(self, inputs, step=2):
        [nC, row, col] = inputs.shape
        output = torch.zeros(nC, row, col + (nC - 1) * step).cuda().float()
        for i in range(nC):
            output[i, :, step * i:step * i + col] = inputs[i, :, :]
        return output

    def generate_shift_masks(self):
        mask = sio.loadmat(self.opt.mask_path + 'mask_3d_shift.mat')
        mask_3d_shift = mask['mask_3d_shift']
        mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
        mask_3d_shift = torch.from_numpy(mask_3d_shift)
        [nC, H, W] = mask_3d_shift.shape
        Phi = mask_3d_shift.expand([nC, H, W]).float()
        Phi_s = torch.sum(Phi ** 2, 1)
        Phi_s[Phi_s == 0] = 1
        # print(Phi.shape, Phi_s.shape)
        return Phi, Phi_s

    def __getitem__(self, index):
        if self.isTrain == True:
            index1   = random.randint(0, len(self.HSI)-1)
            hsi  =  self.HSI[index1]
        else:
            index1 = index
            hsi = self.HSI[index1, :, :, :]

        mask = self.mask
        mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
        ## image patch
        shape = np.shape(hsi)
        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        label = hsi[px:px + self.size:1, py:py + self.size:1, :]

        ## mask patch
        pxm = random.randint(0, 256 - self.size)
        pym = random.randint(0, 256 - self.size)
        mask = mask[pxm:pxm + self.size:1, pym:pym + self.size:1]
        mask_3d = mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :]

        if self.isTrain == True:

            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label  =  np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()

        temp = mask_3d * label
        temp_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        temp_shift[:, 0:self.size, :] = temp
        for t in range(28):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)
        input = meas / 28 * 2

        label = torch.FloatTensor(label.copy()).permute(2,0,1)
        input = torch.FloatTensor(input.copy()).unsqueeze(2).permute(2,0,1)
        mask_3d = torch.FloatTensor(mask_3d.copy()).permute(2, 0, 1)

        if self.opt.input_mask == 'Phi':
            mask = self.shift(mask_3d)
        elif self.opt.input_mask == 'Mask':
            mask  = torch.FloatTensor(mask).unsqueeze(0)
        elif self.opt.input_mask == 'Phi_PhiPhiT':
            mask  = self.generate_shift_masks()
        elif self.opt.input_mask == None:
            mask = torch.FloatTensor(mask).unsqueeze(0)

        if self.opt.input_setting == 'H':
            input = self.shift_back(input)
        elif self.opt.input_setting == 'HM':
            H = self.shift_back(input)
            input = torch.mul(H, mask_3d)
        elif self.opt.input_setting == 'Y':
            input = input

        return input, label, mask

    def __len__(self):
        return self.num
