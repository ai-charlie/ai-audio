import argparse
from logging import raiseExceptions
import sys
import os
import os.path
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
# use fp16totrain
from torch.cuda.amp import autocast

# import from local file
from datasets import get_dataloader
from losses import AngleLoss, CenterLoss, FocalLoss, TripletLoss
from models import Classifier
from utils import AverageMeter, Summary, ProgressMeter, save_checkpoint, accuracy, Logger
from metrics import SphereProduct, AddMarginProduct, ArcMarginProduct

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

best_acc1 = 0


def main(args):
    '''
    main function
    '''
    print(
        f"torch.__version__: {torch.__version__}, torchvision.__version__: { torchvision.__version__}"
    )
    # empty gpu cache
    torch.cuda.empty_cache()

    # logs
    os.makedirs("logs", exist_ok=True)
    train_time = time.time()
    sys.stdout = Logger(os.path.join('logs',
                                     'log_' + str(train_time) + '.txt'))
    # set random seed to get the same result.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # GPU distributed setting in here.
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch]()

    # define loss function (criterion),
    num_ftrs = model.fc.in_features
    if args.loss == "FocalLoss":
        criterion = FocalLoss(gamma=2).cuda(args.gpu)
    elif args.loss == "CrossEntryLoss":
        model.fc = nn.Linear(in_features=num_ftrs,
                             out_features=args.num_classes)
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.loss == "AngleLoss":
        # math: y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        # <https://arxiv.org/abs/1502.03167>
        model.fc = nn.Linear(in_features=num_ftrs, out_features=128)
        criterion = AngleLoss().cuda(args.gpu)
    elif args.loss == "CenterLoss":
        try:
            model = Classifier(backbone).cuda()
        except:
            from collections import OrderedDict
            classifier = nn.Sequential(
                OrderedDict([('fc1', nn.Linear(num_ftrs, 512)),
                             ('relu1', nn.ReLU()),
                             ('dropout1', nn.Dropout(0.5)),
                             ('fc2', nn.Linear(512, 128)),
                             ('output', nn.Softmax(dim=1))]))
            model.fc = classifier
        criterion = CenterLoss(num_classes=args.num_classes).cuda(args.gpu)
    else:
        raise ValueError()

    # optimizer, and learning rate scheduler
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params=[{
            "params": backbone.parameters()
        }, {
            "params": classifier.parameters()
        }],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=[{
            "params": backbone.parameters()
        }, {
            "params": classifier.parameters()
        }],
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        raise ValueError()

    # count model parameters

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_parameters = count_parameters(model)
    print("Number of parameters: %s" % num_parameters)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading
    print("PROJECT PATH: ", os.getcwd())
    args.data = os.path.join(os.getcwd(), args.data)
    # traindir = os.path.join(args.data, 'datasets')
    # traindir = os.path.join(args.data, 'lfw')
    traindir = args.data
    train_dataset = get_dataloader(traindir, local_rank=args.rank)

    dataset_len = len(train_dataset)
    train_dataset_len = int(dataset_len * 0.8)
    # X_train, X_test, y_train, y_test = train_test_split(df['data'],df['label'],test_size=0.2,stratify=df['label'])
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_dataset_len, dataset_len - train_dataset_len])
    print(
        dataset_len,
        train_dataset_len,
        dataset_len - train_dataset_len,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, seed=args.seed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, seed=args.seed, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
    )

    if args.evaluate:
        print("evaluate")
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader) + (
        args.distributed and
        (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(
            val_loader.dataset)):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(
                len(val_loader.sampler) * args.world_size,
                len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    '''
    argparse 模块是标准库中最大的模块之一，拥有大量的配置选项。 
    https://python3-cookbook.readthedocs.io/zh_CN/latest/c13/p03_parsing_command_line_options.html
    '''

    parser = argparse.ArgumentParser(
        description='PyTorch Audio Emotion Regcognition')
    parser.add_argument('data', default='data', type=str, metavar='DIR')
    parser.add_argument('--num-classes', default=10575,
                        type=int, metavar='NUM')
    parser.add_argument('-a',
                        '--arch',
                        dest='arch',
                        default='resnet18',
                        choices=model_names)
    parser.add_argument('--loss', metavar='LOSS', default='CrossEntryLoss')
    parser.add_argument('--optimizer', metavar='Optimizer', default='sgd')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='NUM')
    parser.add_argument('--epochs', default=20, type=int, metavar='NUM')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='NUM')
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        dest='lr')
    parser.add_argument('--lr-step',
                        '--learning-rate-step',
                        default=30,
                        type=int,
                        dest='lr_step')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        dest='weight_decay')
    parser.add_argument('-p',
                        '--print-freq',
                        default=10,
                        type=int,
                        dest='print_freq')
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
    )
    parser.add_argument(
        '-e',
        '--evaluate',
        dest='evaluate',
        action='store_true',
    )
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--world-size', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument(
        '--dist-url', default='tcp://localhost:23456', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=2048, type=int)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument(
        '--multiprocessing-distributed',
        action='store_true',
    )
    # parse args
    args = parser.parse_args()

    main(args)
    # sys.exit(0)
