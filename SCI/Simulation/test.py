from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
import time
from Dataset import dataset
from option import opt
from torch.autograd import Variable
import torch.utils.data as tud

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

test_set = LoadTest(opt.test_path)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def test(model):
    psnr_list, ssim_list = [], []

    Test_Dataset = dataset(opt, test_set, isTrain=False)
    loader_test = tud.DataLoader(Test_Dataset, num_workers=2, batch_size=10, shuffle=False)
    model.eval()

    for _, (input, label, mask) in enumerate(loader_test):
        with torch.no_grad():
            input, label, mask = Variable(input), Variable(label), Variable(mask)
            input, label, mask = input.cuda(), label.cuda(), mask.cuda()
            begin = time.time()
            model_out = model(input, mask)
            end = time.time()
            for idx in range(10):
                psnr_val = torch_psnr(model_out[idx, :, :, :], label[idx, :, :, :])
                ssim_val = torch_ssim(model_out[idx, :, :, :], label[idx, :, :, :])
                psnr_list.append(psnr_val.detach().cpu().numpy())
                ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(label.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print('===> Testing results: psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(psnr_mean, ssim_mean,(end - begin)))
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean = test(model)
    name = opt.outf + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})

if __name__ == '__main__':
    main()