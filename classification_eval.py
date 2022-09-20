import argparse
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from classification_model import Classifier
from dataloader import StainlessDefectsDataset
from logger import Logger
from utils import accuracy


def eval(model, criterion, test_loader, logger=None):
    model.eval()

    with torch.no_grad():
        confusion_mat = [[0, 0], [0, 0]]
        for i, (img_paths, inputs, targets) in tqdm(enumerate(test_loader), leave=False, desc='Testing', total=len(test_loader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)
            loss = criterion(output, targets)
            acc = accuracy(output, targets)[0]

            # Confusion Matrix
            _, preds = output.topk(1, 1, True, True)
            for img_path, t, p in zip(img_paths, targets, preds):
                confusion_mat[int(t)][p[0]] += 1
                logger.write_log('{},{},{}'.format(img_path, t, p[0]))  # img paths, GT, pred

            logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})

            del output, loss, acc

        if logger is not None:
            logger('*Test', history_key='total', confusion_mat=confusion_mat, time=time.strftime('%Y%m%d%H%M%S'))


def run(args):
    # Model
    model = Classifier(args.model, num_classes=2, pretrained=True)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Dataset
    test_dataset = StainlessDefectsDataset(args.data, 'test', input_size=args.input_size, data_type='classification')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=1, dataset_size=len(test_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'confusion_mat', 'time'])
    logger(str(args))

    # Run training
    eval(model, criterion, test_loader, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='resnet101', help='u2net or u2netp')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    # Data Arguments
    parser.add_argument('--data', default='~/data/KolektorSDD2', help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    # Evaluating Arguments
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')
    # Debugging Arguments
    parser.add_argument('--result', default='results_eval', help='path to results')
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    os.makedirs(args.result, exist_ok=True)

    run(args)
