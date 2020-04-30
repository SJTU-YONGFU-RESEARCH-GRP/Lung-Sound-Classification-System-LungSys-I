import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import joblib
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--size_m', type=int, default=56)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rates',nargs='+')
parser.add_argument('--decay_epoch',nargs='+')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--comment', type=str, default='./experiment1')
parser.add_argument('--classweight', type=str, default='true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--input', '-i', default='data/stft_mfcc/mfcc_stft.p', type=str,
        help='path to directory with input data archives')
parser.add_argument('--test', default='data/stft_mfcc/mfcc_stft_test.p', type=str,
        help='path to directory with test data archives')
args = parser.parse_args()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.droupout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
#        print("input:"+str(x.size()))
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.droupout(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.droupout(out)
#        print("output:"+str(out.size()))
        return out + shortcut


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class BilinearCNN(nn.Module):
    
    def __init__(self,dim):
        super(BilinearCNN, self).__init__()
        
        self.conv0 = nn.Conv2d(1, 64, 3, 1)
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.ResNet_0_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0 = ResBlock(64, 64)
        self.ResNet_1 = ResBlock(64, 64)
        self.ResNet_2 = ResBlock(64, 64)
        self.ResNet_3 = ResBlock(64, 64)
        self.ResNet_4 = ResBlock(64, 64)
        self.ResNet_5 = ResBlock(64, 64)
        self.ResNet_6 = ResBlock(64, 64)
        self.ResNet_7 = ResBlock(64, 64)
        self.ResNet_8 = ResBlock(64, 64)
        self.ResNet_9 = ResBlock(64, 64)
        self.ResNet_10 = ResBlock(64, 64)
        self.ResNet_11 = ResBlock(64, 64)
        self.ResNet_12 = ResBlock(64, 64)
        self.ResNet_13 = ResBlock(64, 64)
        self.ResNet_14 = ResBlock(64, 64)
        self.ResNet_15 = ResBlock(64, 64)
        self.ResNet_16 = ResBlock(64, 64)
        self.ResNet_17 = ResBlock(64, 64)
        self.norm0 = norm(dim)
        self.norm1 = norm(dim)
        self.relu0 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 4)
        self.dropout = nn.Dropout(args.dropout)
        self.flat = Flatten()
        
        
    def forward(self,stft,mfcc):
        
        out_s = self.conv0(stft)
        out_s = self.ResNet_0_0(out_s)
        out_s = self.ResNet_0_1(out_s)
        out_s = self.ResNet_0(out_s)
        out_s = self.ResNet_2(out_s)
        out_s = self.ResNet_4(out_s)
        out_s = self.ResNet_6(out_s)
        out_s = self.ResNet_8(out_s)
        out_s = self.ResNet_10(out_s)
        out_s = self.ResNet_12(out_s)
#        out_s = self.ResNet_14(out_s)
#        out_s = self.ResNet_16(out_s)
        out_s = self.norm0(out_s)
        out_s = self.relu0(out_s)
        out_s = self.pool0(out_s)
        
        out_m = self.conv1(mfcc)
        out_m = self.ResNet_1_0(out_m)
        out_m = self.ResNet_1_1(out_m)
        out_m = self.ResNet_1(out_m)
        out_m = self.ResNet_3(out_m)
        out_m = self.ResNet_5(out_m)
        out_m = self.ResNet_7(out_m)
        out_m = self.ResNet_9(out_m)
        out_m = self.ResNet_11(out_m)
        out_m = self.ResNet_13(out_m)
#        out_m = self.ResNet_15(out_m)
#        out_m = self.ResNet_17(out_m)
        out_m = self.norm1(out_m)
        out_m = self.relu1(out_m)
        out_m = self.pool1(out_m)

        out = torch.matmul(out_s,out_m)

#        out = torch.bmm(out_s, torch.transpose(out_m, 1, 2))
        out = self.flat(out)
        out = self.linear(out)
        out = self.dropout(out)
       
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class myDataset(data.Dataset):
    def __init__(self, stft, mfcc, targets):
        self.stft = stft
        self.mfcc = mfcc
        self.targets = targets

    def __getitem__(self, index):
        sample_stft = self.stft[index]
        sample_mfcc = self.mfcc[index]
        target = self.targets[index]
        min_s = np.min(sample_stft)
        max_s = np.max(sample_stft)
        sample_stft = (sample_stft-min_s)/(max_s-min_s) 
        min_m = np.min(sample_mfcc)
        max_m = np.max(sample_mfcc)
        sample_mfcc = (sample_mfcc-min_m)/(max_m-min_m) 
        
        output_stft = torch.FloatTensor([sample_stft])
        crop_s = transforms.Resize([args.size,args.size])
        img_s = transforms.ToPILImage()(output_stft)
        croped_img=crop_s(img_s)
        output_stft = transforms.ToTensor()(croped_img)
        
        output_mfcc = torch.FloatTensor([sample_mfcc])
        crop_m = transforms.Resize([args.size_m,args.size_m])
        img_m = transforms.ToPILImage()(output_mfcc)
        croped_img_m=crop_m(img_m)
        output_mfcc = transforms.ToTensor()(croped_img_m)
#        print(output.size())
        return output_stft,output_mfcc,target
    def __len__(self):
        return len(self.targets)

        
def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    stft, mfcc, labels = joblib.load(open(args.input, mode='rb'))
    stft_test, mfcc_test, labels_test = joblib.load(open(args.test, mode='rb'))

    train_loader = DataLoader(
        myDataset(stft, mfcc, labels), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

#    train_eval_loader = DataLoader(
#        myDataset(features_test, labels_test),
#        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
#    )

    test_loader = DataLoader(
        myDataset(stft_test, mfcc_test, labels_test),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, test_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for stft,mfcc, y in dataset_loader:
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        y = one_hot(np.array(y.numpy()), 4)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(stft,mfcc).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)
 
def Loss(model, dataset_loader):
    total_loss = 0
    for stft,mfcc, y in dataset_loader:
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        y = y.to(device)
        logits = model(stft,mfcc)
        entroy = nn.CrossEntropyLoss().to(device)
        loss = entroy(logits, y).cpu().numpy()
        total_loss += loss
    return total_loss  / (len(dataset_loader.dataset)/args.batch_size)

def confusion_matrix(model, dataset_loader):
    targets = []
    outputs = []

    for stft,mfcc, y in dataset_loader:
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        y = one_hot(np.array(y.numpy()), 4)
        
        target_class = np.argmax(y, axis=1)
        targets = np.append(targets,target_class)
        predicted_class = np.argmax(model(stft,mfcc).cpu().detach().numpy(), axis=1)
        outputs = np.append(outputs,predicted_class)
        
    Confusion_matrix=sk_confusion_matrix(targets.tolist(), outputs.tolist())
    print('Confusion_matrix:')
    print(Confusion_matrix)
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    print('classification_report:')
    print(classification_report(targets.tolist(), outputs.tolist(), target_names=target_names))
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    writer = SummaryWriter(comment=args.comment)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    model = BilinearCNN(64).to(device)
#    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    weights = [1.0, 1.7, 4.1, 5.7]
    class_weights = torch.FloatTensor(weights)
    if args.classweight == 'true':
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    elif args.classweight == 'false':
        criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60,100,120],
        decay_rates=[1,0.1,0.01,0.001]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    end = time.time()
    writer.add_graph(model)
    for itr in range(args.nepochs * batches_per_epoch):
        torch.cuda.empty_cache()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        stft, mfcc, y = data_gen.__next__()
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        y = y.to(device)
        logits = model(stft, mfcc)     
#        print(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        batch_time_meter.update(time.time() - end)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_loader)
                val_acc = accuracy(model, test_loader)
                train_loss = Loss(model, train_loader)
                val_loss = Loss(model, test_loader)
                writer.add_scalar('train/loss',train_loss,itr // batches_per_epoch)
                writer.add_scalar('test/loss',val_loss,itr // batches_per_epoch)
                writer.add_scalar('train/acc',train_acc,itr // batches_per_epoch)
                writer.add_scalar('test/acc',val_acc,itr // batches_per_epoch)
                confusion_matrix(model, test_loader)
                confusion_matrix(model, train_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) |  "
                    "Train Acc {:.4f} | Test Acc {:.4f} | Train Loss {:.4f} | Test Loss {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, train_acc, val_acc, train_loss , val_loss
                    )
                )
