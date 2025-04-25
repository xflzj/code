import os
import sys
import time
import logging
import argparse
import numpy as np
import pretrainedmodels
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from kd_losses import *
from utils_data import SubsetImageNet
from SAM import SAM
parser=argparse.ArgumentParser(description='train kd by logits')
parser.add_argument('--save_root',type=str , default='')

parser.add_argument('--img_s',type=str , default='')
#########################
parser.add_argument('--note', type=str, default='', help='note for this run')
# parser.add_argument('--t_name', type=str, default='resnet152', help='name of teacher')    # resnet152/densenet201
# parser.add_argument('--s_name', type=str, default='resnet152', help='name of student')    # resnet152/densenet201
parser.add_argument('--arch', type=str, default='resnet152', help='the network')    # resnet152/densenet201

parser.add_argument('--T', type=float, default=20, help='temperature for ST')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=24, help='number of total epochs to run')
#####################################
# training hyper parametersq
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')

parser.add_argument('--batch_size', type=int, default=45, help='The size of batch')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--cuda', type=int, default=1)#

#others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--lambda_kd', type=float, default=1, help='trade-off parameter for kd loss')

kwargs = {'num_workers': 4, 'pin_memory': True}

args,unparsed=parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    logging.info('----------- Network Initialization --------------')

    ### student
    snet = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    snet = nn.Sequential(Normalize(mean=mean, std=std), snet)
    snet = snet.cuda()
    logging.info('Student: %s', snet)
    logging.info('Student param size = %fMB', count_parameters_in_MB(snet))
    ### teacher
    tnet=pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    tnet = nn.Sequential(Normalize(mean=mean, std=std), tnet)
    tnet = tnet.cuda()
    tnet.eval()

    for param in tnet.parameters():
        param.requires_grad = False

    logging.info('Teacher: %s', tnet)
    logging.info('Teacher param size = %fMB', count_parameters_in_MB(tnet))
    logging.info('-----------------------------------------------')

    criterionKD = SoftTarget(args.T)
    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()
    
    #base_opt  = torch.optim.SGD(snet.parameters(),
    #                           lr=args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay,
    #                            nesterov=True)
    
    base_opt = torch.optim.SGD
    optimizer = SAM(snet.parameters(), base_opt, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,nesterov=True)
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
      # 数据集处理操作
    s_data_set = SubsetImageNet(root=args.img_s, transform=transform_test) 
    s_train_loader = torch.utils.data.DataLoader(s_data_set, batch_size=args.batch_size, shuffle=True, **kwargs)  # 批处理操作
    test_loader = torch.utils.data.DataLoader(s_data_set, batch_size=args.batch_size, shuffle=False, **kwargs) 

    nets = {'snet': snet, 'tnet': tnet}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        train(s_train_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        logging.info('Testing the models......')
        test_top1, test_top5,kd_loss = test(test_loader, nets, criterions, epoch)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        # save model
        is_best = False
        if kd_loss < best_loss:
            best_loss = kd_loss
            is_best = True
        logging.info('Saving models......')
        save_checkpoint({
            'epoch': epoch,
            'snet': snet.state_dict(),
            'tnet': tnet.state_dict(),
            'prec@1': test_top1,
            'prec@5': test_top5,
        }, is_best, args.save_root)

def train(s_train_loader, nets, optimizer, criterions, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionKD  = criterions['criterionKD']

    snet.train()

    end = time.time()
    for i, (img, target,idx) in enumerate(s_train_loader, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        out_s = snet(img)
        out_t = tnet(img)

        kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd
        loss =  kd_loss

        prec1, prec5 = accuracy(out_s, target, topk=(1,5))
        kd_losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        
        loss.backward()
        optimizer.first_step(zero_grad=True)
        loss = criterionKD(snet(img), tnet(img).detach()) * args.lambda_kd
        loss.backward()
        optimizer.second_step(zero_grad=True)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'Time:{batch_time.val:.4f} '
                       'Data:{data_time.val:.4f}  '
                       'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                       'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                       epoch, i, len(s_train_loader), batch_time=batch_time, data_time=data_time,
                       cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
            logging.info(log_str)

def test(test_loader, nets, criterions, epoch):
    cls_losses = AverageMeter()
    kd_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']

    snet.eval()

    end = time.time()
    for i, (img, target,idx) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            out_s = snet(img)
            out_t = tnet(img)

        cls_loss = criterionCls(out_s, target)

        kd_loss  = criterionKD(out_s, out_t.detach()) * args.lambda_kd


        prec1, prec5 = accuracy(out_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
    logging.info('Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg, kd_losses.avg


def adjust_lr_init(optimizer, epoch):
    scale   = 0.1
    lr_list = [args.lr*scale] * 30
    lr_list += [args.lr*scale*scale] * 10
    lr_list += [args.lr*scale*scale*scale] * 10

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optimizer, epoch):
    scale   = 0.1
    lr_list = [args.lr] * 20
    lr_list += [args.lr*scale] * 4

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    
    main()









