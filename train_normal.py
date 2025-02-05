"""
Author: Benny
Date: Nov 2019
"""
import argparse
import csv
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset
import hydra
import omegaconf
import torch.nn.functional as F

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def write_header_csv(path: str):
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Similarity", "Test Similarity"])


def write_stats_csv(path: str, epoch: int, train_sim: float, test_sim: float):
    with open(path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_sim, test_sim])


@hydra.main(config_path='config', config_name='normal', version_base=None)
def main(args):
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    root = hydra.utils.to_absolute_path('data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='trainval', normal_channel=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerNorm')(
        args)
    try:
        classifier = torch.compile(classifier, backend="eager")
    except:
        pass
    finally:
        classifier = classifier.cuda()

    csv_path = "./normal_prediction.csv"

    if not os.path.exists(csv_path):
        write_header_csv(csv_path)

    criterion = torch.nn.CosineSimilarity(dim=-1)

    best_acc = 0
    global_epoch = 0
    best_cosine_similarity = 0

    try:
        checkpoint = torch.load('best_model_norm.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_cosine_similarity = checkpoint['test_acc']
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        train_cosine_similarity = 0
        train_points_seen = 0

        '''learning one epoch'''
        for i, (points, _, _) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points = points.data.numpy()
            # points = provider.rotate_point_cloud_with_normal(points)

            target = torch.Tensor(points[:, :, 3:])

            points = points[:, :, :3]

            points = provider.random_scale_point_cloud(points)
            points = provider.shift_point_cloud(points)

            points = torch.Tensor(points)

            points, target = points.float().cuda(), target.float().cuda()
            optimizer.zero_grad()

            pred_normals = classifier(points)

            cosine_similarity = F.cosine_similarity(pred_normals, target, dim=-1)
            train_cosine_similarity += cosine_similarity.sum().item()
            train_points_seen += points.size(0) * points.size(1)
            # seg_pred = seg_pred.contiguous().view(-1, num_part)
            # target = target.view(-1, 1)[:, 0]
            # pred_choice = seg_pred.data.max(1)[1]

            # correct = pred_choice.eq(target.data).cpu().sum()
            # mean_correct.append(correct.item() / (args.batch_size * args.num_point))
            loss = (1. - criterion(pred_normals, target)).mean()
            loss.backward()
            optimizer.step()

        train_acc = train_cosine_similarity / train_points_seen
        logger.info('Train cosine similarity is: %.5f' % train_acc)

        with torch.no_grad():
            test_metrics = {}
            total_mse = 0
            total_cosine_similarity = 0
            total_seen = 0

            classifier = classifier.eval()

            for batch_id, (points, _, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                 smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points = points.data.numpy()

                target = torch.Tensor(points[:, :, 3:])

                points = points[:, :, :3]
                points = torch.Tensor(points)

                points, target = points.float().cuda(), target.float().cuda()

                pred_normals = classifier(points)

                # Calculate MSE
                mse = F.mse_loss(pred_normals, target, reduction='sum').item()
                total_mse += mse

                # Calculate cosine similarity
                cosine_similarity = F.cosine_similarity(pred_normals, target, dim=-1)
                total_cosine_similarity += cosine_similarity.sum().item()

                total_seen += cur_batch_size * NUM_POINT

            test_metrics['mse'] = total_mse / total_seen
            test_metrics['cosine_similarity'] = total_cosine_similarity / total_seen

        logger.info('Epoch %d - Test MSE: %f  Average Cosine Similarity: %f' % (
            epoch + 1, test_metrics['mse'], test_metrics['cosine_similarity']))

        write_stats_csv(csv_path, epoch + 1, train_acc, test_metrics['cosine_similarity'])

        if test_metrics['cosine_similarity'] >= best_cosine_similarity:
            best_cosine_similarity = test_metrics['cosine_similarity']
            logger.info('Save model...')
            savepath = 'best_model_norm.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_acc,
                'test_acc': test_metrics['cosine_similarity'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        if test_metrics['cosine_similarity'] > best_acc:
            best_acc = test_metrics['cosine_similarity']

        logger.info('Best accuracy is: %.5f' % best_acc)
        global_epoch += 1


if __name__ == '__main__':
    main()
