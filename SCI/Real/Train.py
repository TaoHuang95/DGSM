from Model import DGSMP
from Dataset import dataset
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


if __name__=="__main__":


    ## Model Config
    parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
    parser.add_argument('--data_path_CAVE', default='/home/huangtao/Data/Spectral_SCI/Data/Training_data/', type=str, help='path of data')
    parser.add_argument('--data_path_KAIST', default='/home/huangtao/Data/Spectral_SCI/Data/KAIST/28/', type=str, help='path of data')
    parser.add_argument('--mask_path', default='/home/huangtao/Data/Spectral_SCI/Data/Real_mask_660/mask.mat', type=str,help='Path of mask')
    parser.add_argument("--size", default=200, type=int, help='The training image size')
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--testset_num", default=10, type=int, help='Total number of testset')
    parser.add_argument("--seed", default=1, type=int, help='Random_seed')
    parser.add_argument("--batch_size", default=4, type=int, help='Batch_size')
    parser.add_argument("--isTrain", default=True, type=bool, help='Train or test')
    opt = parser.parse_args()

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    print(opt)


    print("===> New Model")
    model = DGSMP(upscale=1, img_size=(opt.size, opt.size),
                  window_size=10, img_range=1., depths=[6, 6, 6, 6, 6],
                  embed_dim=64, num_heads=[4, 4, 4, 4, 4], mlp_ratio=4)

    print("===> Setting GPU")
    model = dataparallel(model, 4)  # set the number of parallel GPUs

    ## Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.constant_(layer.bias, 0.0)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.constant_(layer.bias, 0.0)

    ## Load training data
    print("===> Loading CAVE")
    key = 'train_list.txt'
    file_path1 = opt.data_path_CAVE + key
    file_list1 = loadpath(file_path1)
    CAVE = prepare_data_cave(opt.data_path_CAVE, file_list1, 30)

    print("===> Loading KAIST")
    key = 'train_list.txt'
    file_path2 = opt.data_path_KAIST + key
    file_list2 = loadpath(file_path2)
    KAIST = prepare_data_KASIT(opt.data_path_KAIST, file_list2, 30)

    ## Load trained model
    initial_epoch = findLastCheckpoint(save_dir="/home/huangtao/Code/Spectral_SCI/Transformer/Real_PAMI/Checkpoint")
    if initial_epoch > 0:
        print('Load model: resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join("/home/huangtao/Code/Spectral_SCI/Transformer/Real_PAMI/Checkpoint", 'model_%03d.pth' % initial_epoch))


    ## Loss function
    criterion = nn.L1Loss()
    # criterion2 = nn.MSELoss()


    ## optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = MultiStepLR(optimizer, milestones=[], gamma=1)
    scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95)


    ## pipline of training
    for epoch in range(initial_epoch, 500):
        model.train()

        Dataset = dataset(opt, CAVE, KAIST)
        loader_train = tud.DataLoader(Dataset, num_workers=1, batch_size=opt.batch_size, shuffle=True)

        scheduler.step(epoch)
        epoch_loss = 0

        start_time = time.time()
        for i, (input, label, mask) in enumerate(loader_train):
            input, label, mask = Variable(input), Variable(label), Variable(mask)
            input, label, mask = input.cuda(), label.cuda(), mask.cuda()

            out = model(input, mask)


            loss = criterion(out, label)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % (2000) == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch + 1, i, len(Dataset)// opt.batch_size, epoch_loss / ((i+1) * opt.batch_size), datetime.datetime.now()))

        elapsed_time = time.time() - start_time
        print('epoch = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(Dataset), elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / i, elapsed_time)), fmt='%2.4f')
        torch.save(model, os.path.join("/home/huangtao/Code/Spectral_SCI/Transformer/Real_PAMI/Checkpoint", 'model_%03d.pth' % (epoch + 1)))
