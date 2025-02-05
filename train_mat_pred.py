"""
Author: Benny
Date: Nov 2019
"""
import argparse
import csv
import os
import platform
import time
from typing import List, Literal

import torch
import datetime
import logging
import sys
import importlib
import shutil

from torch.utils.data import DataLoader

import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset, MaterialDataset
import hydra
import omegaconf
import torch.nn.functional as F

from timer import Timer


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


@hydra.main(config_path='config', config_name='mat_pred', version_base=None)
def main(args):
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # root = hydra.utils.to_absolute_path('data/material_pred/')
    if platform.system() == "Windows":
        root = Path(r"E:\\FYP Dataset\\131072_64\\")
    else:
        root = Path(r"/mnt/e/FYP Dataset/131072_64/")

    train_dataset = MaterialDataset(root=root / "train/Outdoor/Single", npoints=args.num_point, num_samples_per_ds=3,
                                    dataset_type="clean")
    test_dataset = MaterialDataset(root=root / "test/Outdoor/Single", npoints=args.num_point, num_samples_per_ds=3,
                                   dataset_type="clean")
    trainDataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=10)

    testDataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = 3 + (3 + 3) * 64
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerMat')(
        args)
    try:
        classifier = torch.compile(classifier, backend="eager")
    except:
        pass

    classifier = classifier.cuda()
    pytorch_total_params = sum(p.numel() for p in classifier.parameters())
    print(pytorch_total_params)

    csv_path = "./normal_prediction.csv"

    if not os.path.exists(csv_path):
        write_header_csv(csv_path)

    criterion = torch.nn.CosineSimilarity(dim=-1)

    best_acc = 4
    global_epoch = 0
    best_total_loss = 4

    try:
        checkpoint = torch.load('best_model_mat_pred.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_total_loss = checkpoint['best_total_loss']
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

    def calc_loss(pred_albedo, pred_metallic, pred_norm, pred_occ, target_albedo, target_metallic, target_norm,
                  target_occ,
                  total_included: List[Literal['albedo', 'metallic', 'normals', 'occlusion']]):
        loss_norm = (1. - criterion(pred_norm, target_norm).mean())
        loss_albedo = F.mse_loss(pred_albedo, target_albedo)
        loss_metallic = F.mse_loss(pred_metallic, target_metallic)
        loss_occ = F.mse_loss(pred_occ, target_occ)

        total_loss = torch.Tensor([0]).float().cuda()

        for included in total_included:
            if included == 'albedo':
                total_loss += loss_albedo
            elif included == 'metallic':
                total_loss += loss_metallic
            elif included == 'normals':
                total_loss += loss_norm
            elif included == 'occlusion':
                total_loss += loss_occ

        return total_loss, loss_albedo.item(), loss_metallic.item(), loss_norm.item(), loss_occ.item()

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

        train_loss = 0
        train_points_seen = 0

        '''learning one epoch'''
        for i, (data, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            # points = provider.rotate_point_cloud_with_normal(points)

            # target = torch.Tensor(points[:, :, 3:])
            #
            # points = points[:, :, :3]
            #
            # points = provider.random_scale_point_cloud(points)
            # points = provider.shift_point_cloud(points)
            # Slow part
            data = torch.Tensor(data).float().cuda()
            target_albedo, target_metallic, target_norm, target_occ = (torch.Tensor(target[..., :3]).float().cuda(),
                                                                       torch.Tensor(target[..., 4:6]).float().cuda(),
                                                                       torch.Tensor(target[..., 6:9]).float().cuda(),
                                                                       torch.Tensor(target[..., 10:11]).float().cuda())

            # points, target = points.float().cuda(), target.float().cuda()
            optimizer.zero_grad()

            pred_albedo, pred_metallic, pred_occ, pred_normals = classifier(data, target_norm)

            # cosine_similarity = F.cosine_similarity(pred_normals, target, dim=-1)
            # train_cosine_similarity += cosine_similarity.sum().item()
            # train_points_seen += points.size(0) * points.size(1)
            # seg_pred = seg_pred.contiguous().view(-1, num_part)
            # target = target.view(-1, 1)[:, 0]
            # pred_choice = seg_pred.data.max(1)[1]

            # correct = pred_choice.eq(target.data).cpu().sum()
            # mean_correct.append(correct.item() / (args.batch_size * args.num_point))
            total_loss, _, _, _, _ = calc_loss(pred_albedo, pred_metallic, pred_normals, pred_occ, target_albedo,
                                               target_metallic,
                                               target_norm, target_occ, ['albedo', 'metallic', 'occlusion'])

            train_loss += total_loss.item()
            train_points_seen += 1

            total_loss.backward()
            optimizer.step()

        train_acc = train_loss / train_points_seen
        logger.info('Train loss is: %.5f' % train_acc)

        with torch.no_grad():
            test_metrics = {}
            test_loss = 0
            total_albedo_loss = 0
            total_metallic_loss = 0
            total_occ_loss = 0
            total_normal_loss = 0
            # total_cosine_similarity = 0
            total_seen = 0

            classifier = classifier.eval()

            for batch_id, (data, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                 smoothing=0.9):
                data = torch.Tensor(data).float().cuda()
                target_albedo, target_metallic, target_norm, target_occ = (torch.Tensor(target[..., :3]).float().cuda(),
                                                                           torch.Tensor(
                                                                               target[..., 4:6]).float().cuda(),
                                                                           torch.Tensor(
                                                                               target[..., 6:9]).float().cuda(),
                                                                           torch.Tensor(
                                                                               target[..., 10:11]).float().cuda())

                pred_albedo, pred_metallic, pred_occ, pred_normals = classifier(data, target_norm)

                total_loss, loss_albedo, loss_metallic, loss_norm, loss_occ = calc_loss(pred_albedo, pred_metallic,
                                                                                        pred_normals,
                                                                                        pred_occ, target_albedo,
                                                                                        target_metallic,
                                                                                        target_norm, target_occ,
                                                                                        ['albedo', 'metallic',
                                                                                         'occlusion'])

                total_albedo_loss += loss_albedo
                total_metallic_loss += loss_metallic
                total_normal_loss += loss_norm
                total_occ_loss += loss_occ

                test_loss += total_loss.item()

                total_seen += 1

                # # Calculate MSE
                # mse = F.mse_loss(pred_normals, target, reduction='sum').item()
                # total_mse += mse
                #
                # # Calculate cosine similarity
                # cosine_similarity = F.cosine_similarity(pred_normals, target, dim=-1)
                # total_cosine_similarity += cosine_similarity.sum().item()
                #
                # total_seen += cur_batch_size * NUM_POINT

            test_metrics['albedo_loss'] = total_albedo_loss / total_seen
            test_metrics['metallic_loss'] = total_metallic_loss / total_seen
            test_metrics['normal_loss'] = total_normal_loss / total_seen
            test_metrics['occ_loss'] = total_occ_loss / total_seen
            test_metrics['total_loss'] = test_loss / total_seen
            # test_metrics['cosine_similarity'] = total_cosine_similarity / total_seen

        logger.info('Epoch %d - Albedo Loss: %f, Metallic Loss: %f, Occlusion Loss: %f, Total Loss: %f' % (
            epoch + 1, test_metrics['albedo_loss'], test_metrics['metallic_loss'], test_metrics['occ_loss'],
            test_metrics['total_loss']))

        write_stats_csv(csv_path, epoch + 1, train_acc,
                        test_metrics['total_loss'])

        if test_metrics['total_loss'] <= best_total_loss:
            best_total_loss = test_metrics['total_loss']
            logger.info('Save model...')
            savepath = 'best_model_mat_pred.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_acc,
                'best_total_loss': best_total_loss,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        if test_metrics['total_loss'] < best_acc:
            best_acc = test_metrics['total_loss']

        logger.info('Best loss is: %.5f' % best_acc)
        global_epoch += 1


if __name__ == '__main__':
    main()
