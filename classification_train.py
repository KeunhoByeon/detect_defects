import argparse
import os
import random
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from classification_model import Classifier
from dataloader import StainlessDefectsDataset
from logger import Logger
from utils import accuracy


def train(epoch, model, criterion, optimizer, train_loader, logger=None):
    model.train()

    total_confusion_mat, confusion_mat = [[0, 0], [0, 0]], [[0, 0], [0, 0]]
    num_progress, next_print = 0, args.print_freq
    for i, (inputs, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        output = model(inputs)
        loss = criterion(output, targets)
        acc = accuracy(output, targets)[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})
        logger.add_history('batch', {'loss': loss.item(), 'accuracy': acc})

        # Confusion Matrix
        _, preds = output.topk(1, 1, True, True)
        for t, p in zip(targets, preds):
            confusion_mat[int(t)][p[0]] += 1
            total_confusion_mat[int(t)][p[0]] += 1

        num_progress += len(inputs)
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, confusion_mat=confusion_mat, time=time.strftime('%Y-%m-%d %H:%M:%S'))
            confusion_mat = [[0, 0], [0, 0]]
            next_print += args.print_freq

        del output, loss, acc

    if logger is not None:
        logger(history_key='total', epoch=epoch, confusion_mat=total_confusion_mat)


def val(epoch, model, criterion, val_loader, logger=None):
    model.eval()

    with torch.no_grad():
        confusion_mat = [[0, 0], [0, 0]]
        for i, (inputs, targets) in tqdm(enumerate(val_loader), leave=False, desc='Validation {}'.format(epoch), total=len(val_loader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)
            loss = criterion(output, targets)
            acc = accuracy(output, targets)[0]

            logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})

            # Confusion Matrix
            _, preds = output.topk(1, 1, True, True)
            for t, p in zip(targets, preds):
                confusion_mat[int(t)][p[0]] += 1

            del output, loss, acc

        if logger is not None:
            logger('*Validation', history_key='total', confusion_mat=confusion_mat, time=time.strftime('%Y-%m-%d %H:%M:%S'))


def run(args):
    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Model
    model = Classifier(args.model, num_classes=2, pretrained=True)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Dataset
    train_dataset = StainlessDefectsDataset(args.data, 'train', input_size=args.input_size, data_type='classification')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_dataset = StainlessDefectsDataset(args.data, 'test', input_size=args.input_size, data_type='classification')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'confusion_mat', 'time'])
    logger(str(args))

    save_dir = os.path.join(args.result, 'models')
    os.makedirs(save_dir, exist_ok=True)

    # Run training
    val('preval', model, criterion, val_loader, logger=logger)
    print('Training...')
    for epoch in range(args.start_epoch, args.epochs):
        train(epoch, model, criterion, optimizer, train_loader, logger=logger)
        if epoch % args.val_freq == 0:
            val(epoch, model, criterion, val_loader, logger=logger)
            torch.save(model.state_dict(), os.path.join(save_dir, '{}.pth'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='resnet101', help='u2net or u2netp')
    parser.add_argument('--pretrained', default=False, action='store_true', help='Load pretrained model.')
    # Data Arguments
    parser.add_argument('--data', default='~/data/KolektorSDD2', help='path to dataset')
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    # Training Arguments
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation freq')
    parser.add_argument('--print_freq', default=100, type=int, help='print and save frequency')
    parser.add_argument('--result', default='results', help='path to results')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('--debug', default=False, action='store_true', help='debug validation')
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    os.makedirs(args.result, exist_ok=True)

    run(args)
