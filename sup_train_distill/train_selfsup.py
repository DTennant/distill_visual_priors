import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision as tv
import numpy as np
import gc

from autoaug import AutoAugment, Cutout
import models.MobileNet as Mov
import models.ResNet as ResNet
import distiller

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet-1k Training')
parser.add_argument('--data_path', type=str, help='path to dataset')
parser.add_argument('--net_type', default='resnet', type=str, help='networktype: resnet, mobilenet')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('--input-res', type=int, default=224)
parser.add_argument('--brainpp', action='store_true', help='On brainpp or not')
parser.add_argument('--train_json', type=str, help='path to train nori json')
parser.add_argument('--val_json', type=str, help='path to val nori json')
parser.add_argument('--trainval', action='store_true', help='how to train')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--no_distill_epoch', default=5, type=int, help='epoch before distill')
parser.add_argument('--save_path', default='', type=str)

parser.add_argument('--mixup', action='store_true')
parser.add_argument('--no_tea', action='store_true', help='use teacher or not') 

parser.add_argument('--autoaug', action='store_true', help='use auto aug or not')
parser.add_argument('--cutout', action='store_true', help='use cutout or not')
parser.add_argument('--label_smooth', action='store_true')

best_err1 = 100
best_err5 = 100

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    if args.trainval:
        traindir = os.path.join(args.data_path, 'trainval')
        valdir = os.path.join(args.data_path, 'test/labeled')
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_trm = [transforms.RandomResizedCrop(args.input_res),
                 transforms.RandomHorizontalFlip()]
    if args.autoaug:
        train_trm.append(AutoAugment())
    if args.cutout:
        train_trm.append(Cutout(length=args.input_res // 4))
    train_trm.append(transforms.ToTensor())
    train_trm.append(normalize)
    train_trm = transforms.Compose(train_trm)

    train_dataset = datasets.ImageFolder(traindir, transform=train_trm)

    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([
                                           transforms.Resize(int(args.input_res / 0.875)),
                                           transforms.CenterCrop(args.input_res),
                                           transforms.ToTensor(),
                                           normalize])
                                       )

    num_classes = len(train_dataset.classes)
    if args.brainpp:
        assert args.train_json is not None
        assert args.val_json is not None
        from nori_util import get_img, get_sample_list_from_json, get_num_class_from_json
        train_samples = get_sample_list_from_json(args.train_json)
        train_dataset.samples = train_samples
        train_dataset.loader = get_img
        val_samples = get_sample_list_from_json(args.val_json)
        val_dataset.samples = val_samples
        val_dataset.loader = get_img

        num_classes = get_num_class_from_json(args.train_json)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
                                             
    if args.net_type == 'mobilenet':
        t_net = ResNet.resnet50(pretrained=True)
        s_net = Mov.MobileNet()
    elif args.net_type == 'resnet':
        t_net = ResNet.resnet152(pretrained=True)
        s_net = ResNet.resnet50(pretrained=False)
    elif args.net_type == 'self_sup':
        t_net = ResNet.resnet50(pretrained=False, num_classes=num_classes)
        s_net = ResNet.resnet50(pretrained=False, num_classes=num_classes)
        # load from pre-trained, before DistributedDataParallel constructor
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                args.start_epoch = 0
                msg = t_net.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                msg = s_net.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                print("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))
            
    elif args.net_type == 'infomin_self':
        from collections import OrderedDict
        t_net = ResNet.resnet50(pretrained=False, num_classes=num_classes)
        s_net = ResNet.resnet50(pretrained=False, num_classes=num_classes)
        if args.pretrained:
            ckpt = torch.load(args.pretrained, map_location='cpu')
            state_dict = ckpt['model']
            encoder_state_dict = OrderedDict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                if 'encoder' in k:
                    k = k.replace('encoder.', '')
                    encoder_state_dict[k] = v

            msg = t_net.load_state_dict(encoder_state_dict, strict=False)
            print(set(msg.missing_keys))
            msg = s_net.load_state_dict(encoder_state_dict, strict=False)
            print(set(msg.missing_keys))
    elif args.net_type == 'r50':
        s_net = ResNet.resnet50(pretrained=False, num_classes=num_classes)
    elif args.net_type == 'r50_inpretrain':
        s_net = ResNet.resnet50(pretrained=True)
        fc_dim = s_net.fc.weight.shape[1]
        s_net.fc = torch.nn.Linear(fc_dim, num_classes)
    else:
        print('undefined network type !!!')
        raise

    if not args.no_tea:
        d_net = distiller.Distiller(t_net, s_net)

    if not args.no_tea:
        print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
    print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

    if not args.no_tea:
        t_net = torch.nn.DataParallel(t_net).cuda()
        s_net = torch.nn.DataParallel(s_net).cuda()
        d_net = torch.nn.DataParallel(d_net).cuda()
    else:
        s_net = torch.nn.DataParallel(s_net).cuda()

    # define loss function (criterion) and optimizer
    if not args.label_smooth:
        criterion_CE = nn.CrossEntropyLoss().cuda()
    else:
        criterion_CE = CrossEntropyLabelSmooth(num_classes=num_classes)
    if not args.no_tea:
        optimizer = torch.optim.SGD(list(s_net.parameters()) + list(d_net.module.Connectors.parameters()), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.SGD(list(s_net.parameters()), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    cudnn.benchmark = True

    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        if not args.no_tea:
            if args.mixup:
                train_with_distill_mixup(train_loader, d_net, optimizer, criterion_CE, epoch, args.no_distill_epoch)
            else:
                train_with_distill(train_loader, d_net, optimizer, criterion_CE, epoch, args.no_distill_epoch)
        else:
            if args.mixup:
                train_mixup(train_loader, s_net, optimizer, criterion_CE, epoch)
            else:
                train(train_loader, s_net, optimizer, criterion_CE, epoch)
            
        # evaluate on validation set
        err1, err5 = validate(val_loader, s_net, criterion_CE, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5
        print ('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': s_net.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        gc.collect()

    print ('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
 

def validate(val_loader, model, criterion_CE, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)

        # for PyTorch 0.4.x, volatile=True is replaced by with torch.no.grad(), so uncomment the followings:
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            loss = criterion_CE(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test (on val set): [Epoch {0}/{1}][Batch {2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'
          .format(epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg

def train(train_loader, s_net, optimizer, criterion_CE, epoch):
    s_net.train()

    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.cuda(async=True)
        batch_size = inputs.shape[0]
        outputs = s_net(inputs)

        loss = criterion_CE(outputs, targets)

        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))

        train_loss.update(loss.item(), batch_size)
        top1.update(err1.item(), batch_size)
        top5.update(err5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Train : [Epoch %d/%d][Batch %d/%d]\t Loss %.3f, Top 1-error %.3f, Top 5-error %.3f' %
                  (epoch, args.epochs, i, len(train_loader), train_loss.avg, top1.avg, top5.avg))

def train_with_distill(train_loader, d_net, optimizer, criterion_CE, epoch, no_d_epoch):
    d_net.train()
    d_net.module.s_net.train()
    d_net.module.t_net.train()

    train_loss = AverageMeter()
    ce_loss = AverageMeter()
    dis_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.cuda(async=True)
        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)

        loss_CE = criterion_CE(outputs, targets)
        if epoch < no_d_epoch:
            loss = loss_CE
        else:
            loss = loss_CE + loss_distill.sum() / batch_size / 10000

        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))

        train_loss.update(loss.item(), batch_size)
        ce_loss.update(loss_CE.item(), batch_size)
        dis_loss.update(loss_distill.sum().item() / batch_size / 10000, batch_size)
        top1.update(err1.item(), batch_size)
        top5.update(err5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Train with distillation: [Epoch %d/%d][Batch %d/%d]\t Loss %.3f, loss_ce %.3f, Top 1-error %.3f, Top 5-error %.3f' %
                  (epoch, args.epochs, i, len(train_loader), train_loss.avg, ce_loss.avg, top1.avg, top5.avg))

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup(train_loader, s_net, optimizer, criterion_CE, epoch):
    s_net.train()

    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.cuda(async=True)
        batch_size = inputs.shape[0]
        mixed_input, targets_a, targets_b, lam = mixup_data(inputs, targets)
        outputs = s_net(mixed_input)

        loss = mixup_criterion(criterion_CE, outputs, targets_a, targets_b, lam)

        err1_a, err5_a = accuracy(outputs.data, targets_a, topk=(1, 5))
        err1_b, err5_b = accuracy(outputs.data, targets_b, topk=(1, 5))
        err1 = lam * err1_a + (1 - lam) * err1_b
        err5 = lam * err5_a + (1 - lam) * err5_b

        train_loss.update(loss.item(), batch_size)
        top1.update(err1.item(), batch_size)
        top5.update(err5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Train : [Epoch %d/%d][Batch %d/%d]\t Loss %.3f, Top 1-error %.3f, Top 5-error %.3f' %
                  (epoch, args.epochs, i, len(train_loader), train_loss.avg, top1.avg, top5.avg))

def train_with_distill_mixup(train_loader, d_net, optimizer, criterion_CE, epoch, no_d_epoch):
    d_net.train()
    d_net.module.s_net.train()
    d_net.module.t_net.train()

    train_loss = AverageMeter()
    ce_loss = AverageMeter()
    dis_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.cuda(async=True)
        batch_size = inputs.shape[0]
        mixed_input, targets_a, targets_b, lam = mixup_data(inputs, targets)
        outputs, loss_distill = d_net(mixed_input)

        # loss_CE = criterion_CE(outputs, targets)
        loss_CE = mixup_criterion(criterion_CE, outputs, targets_a, targets_b, lam)
        if epoch < no_d_epoch:
            loss = loss_CE
        else:
            loss = loss_CE + loss_distill.sum() / batch_size / 10000

        err1_a, err5_a = accuracy(outputs.data, targets_a, topk=(1, 5))
        err1_b, err5_b = accuracy(outputs.data, targets_b, topk=(1, 5))
        err1 = lam * err1_a + (1 - lam) * err1_b
        err5 = lam * err5_a + (1 - lam) * err5_b

        train_loss.update(loss.item(), batch_size)
        ce_loss.update(loss_CE.item(), batch_size)
        dis_loss.update(loss_distill.sum().item() / batch_size / 10000, batch_size)
        top1.update(err1.item(), batch_size)
        top5.update(err5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Train with distillation: [Epoch %d/%d][Batch %d/%d]\t Loss %.3f, loss_ce %.3f, Top 1-error %.3f, Top 5-error %.3f' %
                  (epoch, args.epochs, i, len(train_loader), train_loss.avg, ce_loss.avg, top1.avg, top5.avg))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    save_path = args.save_path
    if save_path == "":
        save_path = args.net_type
    directory = "runs/%s/"%(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(save_path) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

if __name__ == '__main__':
    main()
