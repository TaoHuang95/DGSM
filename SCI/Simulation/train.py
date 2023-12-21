import os
from option import opt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import datetime
from Dataset import dataset
import torch.utils.data as tud

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')


# dataset
train_set = LoadTraining(opt.data_path)
test_set = LoadTest(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
print("===> Generate Model")
model = model_generator(opt.method, opt.pretrained_model_path)

# GPU
print("===> Setting GPU")

model = dataparallel(model, len(opt.gpu_id.split(',')))  # set the number of parallel GPUs

# optimizing
print("===> Setting Optimizer and Scheduler")
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)

# Define Loss
# mse = torch.nn.MSELoss().cuda()
criterion = nn.L1Loss()


# Train
def train(epoch, logger):
    model.train()

    Dataset = dataset(opt, train_set, isTrain=True)
    loader_train = tud.DataLoader(Dataset, num_workers=4, batch_size=opt.batch_size, shuffle=True)

    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    for i, (input, label, mask) in enumerate(loader_train):
        input, label, mask = Variable(input), Variable(label), Variable(mask)
        input, label, mask = input.cuda(), label.cuda(), mask.cuda()

        optimizer.zero_grad()
        model_out = model(input, mask)
        # loss = torch.sqrt(mse(model_out, label))
        loss = criterion(model_out, label)

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    end = time.time()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    logger.info("===> Epoch {} Complete: Lr: {:.6f} Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, lr, epoch_loss / batch_num, (end - begin)))
    return 0

# Test
def test(epoch, logger):
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
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Method/Model:{}\n".format(opt.method))
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        # train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            # if psnr_mean > 35:
            #     name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
            #     scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
            #     checkpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


