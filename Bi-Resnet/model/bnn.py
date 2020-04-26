import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import joblib
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from tensorboardX import SummaryWriter
from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--size_m', type=int, default=128)
parser.add_argument('--mixup', type=eval, default=True, choices=[True, False])
parser.add_argument('--nonLocal', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=float, default=0.2)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--save', type=str, default='')
parser.add_argument('--comment', type=str, default='./experiment1')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--input', '-i', default='', type=str, help='path to directory with input data archives')
parser.add_argument('--test', default='', type=str, help='path to directory with test data archives')
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

class nonLocal(nn.Module):
    
    def __init__(self, inplanes, planes):
        super(nonLocal, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.theta = conv1x1(inplanes, planes, stride=1)
        self.phi = conv1x1(inplanes, planes, stride=1)
        self.g = conv1x1(inplanes, planes, stride=1)
        self.final = conv1x1(planes, inplanes, stride=1)
        
    def forward(self,x):
        old = x
        H = list(old.size())[-1]
        batch = list(old.size())[0]
        mid_theta = self.theta(x) #[batch_number,channel,H,W]
        mid_phi = self.phi(x)
        mid_g = self.g(x)
        paste = torch.empty(1,self.planes,H,H,device='cuda')
        for i in range(batch):
            i_mid_theta = mid_theta[i].reshape(self.planes,-1)#[channel,HW]
            i_mid_phi = mid_phi[i].reshape(self.planes,-1).t()#[HW,channel]
            i_mid_g = mid_g[i].reshape(self.planes,-1).t()
        
            mid_tp = torch.mm(i_mid_phi,i_mid_theta)
            HW = list(mid_tp.size())[0]
            mid_tp = mid_tp.view(-1)
            output_tp = torch.nn.functional.softmax(mid_tp)
            
            output_tp = output_tp.reshape(HW,HW)
            output = torch.mm(output_tp,i_mid_g).t()
            
            cat_output = output[0].reshape([H,H]).t()
            for i in range(1, len(output)):
                output_mid = output[i].reshape([H,H]).t()
                cat_output = torch.cat([cat_output,output_mid])
                
        
            cat_output = cat_output.reshape([1,-1,H,H])
            paste = torch.cat([paste,cat_output])
        paste = paste[1:]
        paste = self.final(paste)
        paste = paste+old
        return paste

class ResBlock(nn.Module):

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


class BiResNet(nn.Module):
    
    def __init__(self,dim):
        super(BiResNet, self).__init__()
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
   
        self.nonLocal1 = nonLocal(64, 32)
        self.nonLocal2 = nonLocal(64, 32)
        self.nonLocal3 = nonLocal(64, 32)
        self.nonLocal4 = nonLocal(64, 32)

        self.norm0 = norm(dim)
        self.norm1 = norm(dim)
        self.relu0 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 4)
        self.dropout = nn.Dropout(args.dropout)
        self.flat = Flatten()
        
        
    def forward(self,stft):
        
        out_s = self.conv0(stft)
        out_s = self.ResNet_0_0(out_s)
        out_s = self.ResNet_0_1(out_s)
        out_s = self.ResNet_0(out_s)
        if args.nonLocal:
            out_s = self.nonLocal1(out_s)
            out_s = self.relu0(out_s)
        out_s = self.ResNet_2(out_s)
        out_s = self.ResNet_4(out_s)
        if args.nonLocal:
            out_s = self.nonLocal2(out_s)
            out_s = self.relu0(out_s)
        out_s = self.ResNet_6(out_s)

        out_s = self.ResNet_8(out_s)
        if args.nonLocal:
            out_s = self.nonLocal3(out_s)
            out_s = self.relu0(out_s)
        out_s = self.ResNet_10(out_s)
        if args.nonLocal:
            out_s = self.nonLocal4(out_s)
            out_s = self.relu0(out_s)
        out_s = self.ResNet_12(out_s)

        out_s = self.norm0(out_s)
        out_s = self.relu0(out_s)
        out_s = self.pool0(out_s)
        
#        out_m = self.conv1(mfcc)
#        out_m = self.ResNet_1_0(out_m)
#        out_m = self.ResNet_1_1(out_m)
#        out_m = self.ResNet_1(out_m)   
#        if args.nonLocal:
#            out_m = self.nonLocal3(out_m)    
#        out_m = self.ResNet_3(out_m)
#        out_m = self.ResNet_5(out_m)
#        out_m = self.ResNet_7(out_m)
#        out_m = self.ResNet_9(out_m)
#        out_m = self.ResNet_11(out_m)
#
#        out_m = self.norm1(out_m)
#        out_m = self.relu1(out_m)
#        out_m = self.pool1(out_m)
#
#        out = torch.matmul(out_s,out_m)

#        out = torch.bmm(out_s, torch.transpose(out_m, 1, 2))
        out = self.flat(out_s)
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
    def __init__(self, stft, targets):
        self.stft = stft
#        self.mfcc = mfcc
        self.targets = targets

    def __getitem__(self, index):
               
        sample_stft = self.stft[index]
#        sample_mfcc = self.mfcc[index]
        target = self.targets[index]
        target = torch.from_numpy(target)
        
        min_s = np.min(sample_stft)
        max_s = np.max(sample_stft)
        sample_stft = (sample_stft-min_s)/(max_s-min_s) 
#        min_m = np.min(sample_mfcc)
 #       max_m = np.max(sample_mfcc)
#        sample_mfcc = (sample_mfcc-min_m)/(max_m-min_m) 
        
        output_stft = torch.FloatTensor([sample_stft])
        crop_s = transforms.Resize([args.size,args.size])
        img_s = transforms.ToPILImage()(output_stft)
        croped_img=crop_s(img_s)
        output_stft = transforms.ToTensor()(croped_img)
        
#        output_mfcc = torch.FloatTensor([sample_mfcc])
#        crop_m = transforms.Resize([args.size_m,args.size_m])
#        img_m = transforms.ToPILImage()(output_mfcc)
#        croped_img_m=crop_m(img_m)
#        output_mfcc = transforms.ToTensor()(croped_img_m)
#        print(output.size())
        return output_stft,target
    def __len__(self):
        return len(self.targets)

def turn_numpy(labels):
    mid = np.array([labels[0]])    
    new_labels = one_hot(mid, 4)
    for i in range(1,len(labels)):
        mid = np.array([labels[i]])        
        new_labels = np.append(new_labels, one_hot(mid, 4), axis=0)    
    new_labels = list(new_labels)
    #new_labels = [[0,0,0,1],[0,0,0,1],[0,0,0,1]...]    
    return new_labels

def shuffle(stft,labels):
    bond = random.shuffle(list(zip(stft,turn_numpy(labels))))
    stft = []
    new_labels = []
    for i in bond:
        stft.append(i[0])
        new_labels.append(i[1])    
    return stft, new_labels

        
def get_mnist_loaders( batch_size=128, perc=1.0):
    stft, mfcc, labels = joblib.load(open(args.input, mode='rb'))
    labels = turn_numpy(labels)
#    stft, labels = shuffle(stft,labels)
        
    stft_test, mfcc_test, labels_test = joblib.load(open(args.test, mode='rb'))
    labels_test = turn_numpy(labels_test)
    train_loader = DataLoader(
        myDataset(stft,labels), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        myDataset(stft_test, labels_test),
        batch_size=100, shuffle=True, num_workers=2, drop_last=True
    )

    return train_loader, test_loader


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



def one_hot(x, K):
    #x is a array from np
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def Loss(model, dataset_loader):
    total_loss = 0
    size = 0
    for stft, y in dataset_loader:
    
        with torch.no_grad():
            logits = model(stft)
        size = size+1
        entroy = nn.BCEWithLogitsLoss()
        y = y.type_as(logits)
        loss = entroy(logits, y).cpu().detach().numpy()
        total_loss += loss
    return total_loss  / size

def accuracy(model, dataset_loader):
    total_correct = 0
    for stft, y in dataset_loader:

        target_class = np.argmax(y.numpy(), axis=1)
        with torch.no_grad():
            logits = model(stft)       
        predicted_class = np.argmax(logits.cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

def confusion_matrix(model, dataset_loader):
    targets = []
    outputs = []

    for stft,y in dataset_loader:
        
        target_class = np.argmax(y, axis=1)
        targets = np.append(targets,target_class)
        with torch.no_grad():
            logits = model(stft)         
        predicted_class = np.argmax(logits.cpu().detach().numpy(), axis=1)
        outputs = np.append(outputs,predicted_class)
        
    Confusion_matrix=sk_confusion_matrix(targets.tolist(), outputs.tolist())
    print('Confusion_matrix:')
    print(Confusion_matrix)
    Se = Confusion_matrix[0][0]/(sum(Confusion_matrix[0]))
    Sq = (Confusion_matrix[1][1]+Confusion_matrix[2][2]+Confusion_matrix[3][3])/(sum(Confusion_matrix[1])+sum(Confusion_matrix[2])+sum(Confusion_matrix[3]))
    return Se, Sq, (Se+Sq)/2
        
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

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    mixed_y = lam * y + (1 - lam) * y[index,:]
    
    cat_x = torch.cat((x,mixed_x),0)
    y = y.type_as(mixed_y)
    cat_y = torch.cat((y,mixed_y),0)

    return cat_x, cat_y

if __name__ == '__main__':


    writer = SummaryWriter(comment=args.comment)
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    
    use_cuda = torch.cuda.is_available()
    batch_size = args.batch_size
    net = BiResNet(64)
    
    if use_cuda:
    # data parallel
        n_gpu = torch.cuda.device_count()
        batch_size *= n_gpu

        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')
    
    logger.info(net)
    logger.info('Number of parameters: {}'.format(count_parameters(net)))


    criterion = nn.BCEWithLogitsLoss()
    if args.optimizer=='adam':    
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_loader, test_loader = get_mnist_loaders(batch_size)
    
    for epoch in range(args.nepochs):
        
################################train##########################################  
        
        net.train()
    
        for batch_idx, (stft, labels) in enumerate(train_loader): 
            if use_cuda:
                stft, labels = stft.cuda(), labels.cuda()
            if args.mixup:
                stft, labels = mixup_data(stft, labels, args.alpha, use_cuda)
            optimizer.zero_grad()
            stft, labels = Variable(stft), Variable(labels)
            outputs = net(stft)
            labels = labels.type_as(outputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
        train_acc = accuracy(net, train_loader)
        train_loss = Loss(net, train_loader)
        
#################################test##########################################
            
        net.eval()        
            
        val_acc = accuracy(net, test_loader)
        val_loss = Loss(net, test_loader)
        scheduler.step()
###############################################################################
            
        writer.add_scalar('train/loss',train_loss,epoch)
        writer.add_scalar('test/loss',val_loss,epoch)
        writer.add_scalar('train/acc',train_acc,epoch)
        writer.add_scalar('test/acc',val_acc,epoch)
        test_Se,test_Sq,test_Score = confusion_matrix(net, test_loader)
        train_Se,train_Sq,train_Score = confusion_matrix(net, train_loader)
 
        logger.info(
                "Epoch {:04d}  |  "
                "Train Acc {:.4f} | Test Acc {:.4f} | Train Loss {:.4f} | Test Loss {:.4f} | train Se {:.4f} | train Sq {:.4f} | train Score {:.4f} | test Se {:.4f} | test Sq {:.4f} | test Score {:.4f}".format(
                    epoch, train_acc, val_acc, train_loss , val_loss, train_Se, train_Sq, train_Score, test_Se,test_Sq,test_Score
                )
            )
