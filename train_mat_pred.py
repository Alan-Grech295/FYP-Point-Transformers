"""
Author: Benny
Date: Nov 2019
"""
import csv
import importlib
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import List, Literal, Tuple

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from tqdm import tqdm

import utils
from dataset import MaterialDataset
from renderer import Renderer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def write_header_csv(path: str, *attribute_names):
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Loss", "Test Loss"] + list(attribute_names))


def write_stats_csv(path: str, epoch: int, train_sim: float, test_sim: float, *args):
    with open(path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_sim, test_sim] + list(args))


def check_nan(**kwargs):
    for k, v in kwargs.items():
        if torch.isnan(v).any():
            print(f"{k} is NaN")


# https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39
def load_next_batch(data_iter: _BaseDataLoaderIter):
    next_batch = data_iter.__next__()  # start loading the first batch
    return [_.cuda(non_blocking=True) for _ in
            next_batch]  # with pin_memory=True and non_blocking=True, this will copy data to GPU non blockingly


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
        root = Path(r"E:\\FYP Dataset\\32768_64\\")
    else:
        root = Path(r"/home/alan/Dataset/32768_64/")
    train_dataset = MaterialDataset(root=root / "train/Outdoor/", npoints=args.num_point, num_samples_per_ds=2,
                                    dataset_type="rendered")
    test_dataset = MaterialDataset(root=root / "test/Outdoor/", npoints=args.num_point, num_samples_per_ds=2,
                                   dataset_type="rendered")
    trainDataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=12, pin_memory=True)

    testDataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=12, pin_memory=True)

    '''MODEL LOADING'''
    args.input_dim = 3 + (3 + 3) * 64
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    renderer = Renderer(chunk_size=64)

    def calc_loss(pred_albedo, pred_metallic, pred_smoothness, pred_env_map, normals, env_vis, target_view_dirs,
                  target_radiance, target_env_map, target_albedo, target_metallic, target_smoothness, exposure: float,
                  weights: List[Tuple[Literal["albedo", "metallic", "smoothness"], float]]):
        indices = np.random.choice(N, args.render_sample_size, replace=False)

        radiance = renderer(normals[:, indices, ...], pred_albedo[:, indices, ...], pred_metallic[:, indices, ...],
                            pred_smoothness[:, indices, ...], target_view_dirs[:, indices, ...],
                            env_vis[:, indices, ...],
                            pred_env_map, exposure=exposure)

        render_loss = F.mse_loss(radiance * 255, target_radiance[:, indices, ...] * 255)
        albedo_loss = F.mse_loss(pred_albedo * 255, target_albedo * 255)
        metallic_loss = F.mse_loss(pred_metallic * 255, target_metallic * 255)
        smoothness_loss = F.mse_loss(pred_smoothness * 255, target_smoothness * 255)
        env_map_loss = torch.tensor([0], device=target_radiance.device)
        total_loss = render_loss + env_map_loss

        for name, weight in weights:
            if name == "albedo":
                total_loss += albedo_loss * weight
            elif name == "metallic":
                total_loss += metallic_loss * weight
            elif name == "smoothness":
                total_loss += smoothness_loss * weight

        return total_loss, render_loss.item(), env_map_loss.item(), albedo_loss.item(), metallic_loss.item(), smoothness_loss.item()

    global_epoch = 0
    best_total_loss = 1e10

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerMat')(
        args)
    uncompiled_model = classifier
    try:
        classifier = torch.compile(classifier, backend="eager")
    except:
        pass

    classifier = classifier.cuda()

    optimizer_state = None

    try:
        checkpoint = torch.load(f'best_model_mat_pred.pth')
        uncompiled_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint['optimizer_state_dict']
        start_epoch = checkpoint['epoch']
        best_total_loss = checkpoint['best_total_loss']
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    pytorch_trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(pytorch_trainable_params)

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

    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    csv_path = f"./material_prediction_trial.csv"
    # if not os.path.exists(csv_path):
    write_header_csv(csv_path,
                     "Train Albedo Loss", "Test Albedo Loss",
                     "Train Metallic Loss", "Test Metallic Loss",
                     "Train Smoothness Loss", "Test Smoothness Loss",
                     "Train Render Loss", "Test Render Loss")

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
        train_albedo_loss = 0
        train_metallic_loss = 0
        train_smoothness_loss = 0
        train_render_loss = 0
        train_points_seen = 0

        train_iter = iter(trainDataLoader)
        next_batch = load_next_batch(train_iter)

        '''learning one epoch'''
        for i in tqdm(range(len(trainDataLoader)),
                      total=len(trainDataLoader),
                      smoothing=0.9):
            data, target, view_radiances, env_map, exposure = next_batch
            if i + 1 != len(trainDataLoader):
                next_batch = load_next_batch(train_iter)

            xyz = torch.Tensor(data[..., :3]).float()
            normals = torch.Tensor(data[..., 3:6]).float()
            target_albedo = torch.Tensor(target[..., :3]).float()
            target_metallic = torch.Tensor(target[..., 4:5]).float()
            target_smoothness = torch.Tensor(target[..., 5:6]).float()
            env_visibility = torch.Tensor(target[..., 6:]).float()

            exposure = torch.Tensor(exposure).float()

            B, N = data.shape[0], data.shape[1]
            view_radiances = view_radiances.reshape(B, N, 3, -1, 3)
            target_view_dirs = torch.Tensor(view_radiances[..., 0, :, :]).float().cuda()
            target_view_rads = torch.Tensor(view_radiances[..., 1, :, :]).float().cuda()

            target_env_map = torch.Tensor(env_map).float().cuda()

            optimizer.zero_grad()

            pred_albedo, pred_metallic, pred_smoothness = classifier(xyz, normals, target_view_dirs,
                                                                     target_view_rads)

            total_loss, render_loss, env_map_loss, albedo_loss, metallic_loss, smoothness_loss = calc_loss(
                pred_albedo, pred_metallic, pred_smoothness, target_env_map,
                normals,
                env_visibility, target_view_dirs, target_view_rads,
                target_env_map, target_albedo, target_metallic, target_smoothness,
                exposure,
                weights=[("albedo", 0.5), ("metallic", 0.5), ("smoothness", 0.5)])

            train_loss += total_loss.item()
            train_render_loss += render_loss
            train_albedo_loss += albedo_loss
            train_metallic_loss += metallic_loss
            train_smoothness_loss += smoothness_loss
            train_points_seen += 1

            total_loss.backward()

            optimizer.step()

        utils.print_gradient_norm_named(classifier)

        train_loss /= train_points_seen
        train_render_loss /= train_points_seen
        train_albedo_loss /= train_points_seen
        train_metallic_loss /= train_points_seen
        train_smoothness_loss /= train_points_seen
        logger.info('Train loss is: %.5f' % train_loss)

        with torch.no_grad():
            test_metrics = {}
            test_loss = 0
            total_render_loss = 0
            total_env_map_loss = 0
            total_albedo_loss = 0
            total_metallic_loss = 0
            total_smoothness_loss = 0
            total_seen = 0

            classifier = classifier.eval()

            test_iter = iter(testDataLoader)
            next_batch = load_next_batch(test_iter)

            for batch_id in tqdm(range(len(testDataLoader)),
                                 total=len(testDataLoader),
                                 smoothing=0.9):
                data, target, view_radiances, env_map, exposure = next_batch
                if batch_id + 1 != len(testDataLoader):
                    next_batch = load_next_batch(test_iter)

                xyz = torch.Tensor(data[..., :3]).float()
                normals = torch.Tensor(data[..., 3:6]).float()
                target_albedo = torch.Tensor(target[..., :3]).float()
                target_metallic = torch.Tensor(target[..., 4:5]).float()
                target_smoothness = torch.Tensor(target[..., 5:6]).float()
                env_visibility = torch.Tensor(target[..., 6:]).float()

                exposure = torch.Tensor(exposure).float()

                B, N = data.shape[0], data.shape[1]
                view_radiances = view_radiances.reshape(B, N, 3, -1, 3)
                target_view_dirs = torch.Tensor(view_radiances[..., 0, :, :]).float().cuda()
                target_view_rads = torch.Tensor(view_radiances[..., 1, :, :]).float().cuda()
                target_view_hdr_rads = torch.Tensor(view_radiances[..., 2, :, :]).float().cuda()

                target_env_map = torch.Tensor(env_map).float().cuda()
                # min_env_map, max_env_map = torch.min(target_env_map), torch.max(target_env_map)

                optimizer.zero_grad()

                pred_albedo, pred_metallic, pred_smoothness = classifier(xyz, normals,
                                                                         target_view_dirs,
                                                                         target_view_rads)

                total_loss, render_loss, env_map_loss, albedo_loss, metallic_loss, smoothness_loss = calc_loss(
                    pred_albedo, pred_metallic, pred_smoothness,
                    target_env_map, normals,
                    env_visibility, target_view_dirs,
                    target_view_rads, target_env_map, target_albedo, target_metallic, target_smoothness, exposure,
                    weights=[("albedo", 0.5), ("metallic", 0.5), ("smoothness", 0.5)])

                total_render_loss += render_loss
                total_env_map_loss += env_map_loss
                total_albedo_loss += albedo_loss
                total_metallic_loss += metallic_loss
                total_smoothness_loss += smoothness_loss

                test_loss += total_loss.item()

                total_seen += 1

            test_metrics['render_loss'] = total_render_loss / total_seen
            test_metrics['env_map_loss'] = total_env_map_loss / total_seen
            test_metrics['albedo_loss'] = total_albedo_loss / total_seen
            test_metrics['metallic_loss'] = total_metallic_loss / total_seen
            test_metrics['smoothness_loss'] = total_smoothness_loss / total_seen
            test_metrics['total_loss'] = test_loss / total_seen
            # test_metrics['cosine_similarity'] = total_cosine_similarity / total_seen

        logger.info(
            'Epoch %d - Render Loss: %f, Environment Map Loss: %f, Albedo Loss: %f, Metallic Loss: %f, \
            Smoothness Loss: %f, Total Loss: %f' % (
                global_epoch + 1, test_metrics['render_loss'], test_metrics['env_map_loss'],
                test_metrics['albedo_loss'], test_metrics['metallic_loss'], test_metrics['smoothness_loss'],
                test_metrics['total_loss']))

        write_stats_csv(csv_path, global_epoch + 1, train_loss, test_metrics['total_loss'],
                        train_albedo_loss, test_metrics['albedo_loss'],
                        train_metallic_loss, test_metrics['metallic_loss'],
                        train_smoothness_loss, test_metrics['smoothness_loss'],
                        train_render_loss, test_metrics['render_loss'])

        if test_metrics['total_loss'] <= best_total_loss:
            best_total_loss = test_metrics['total_loss']
            logger.info('Save model...')
            savepath = f'best_model_mat_pred.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': global_epoch,
                'train_loss': train_loss,
                'best_total_loss': best_total_loss,
                'model_state_dict': uncompiled_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        logger.info('Best loss is: %.5f' % best_total_loss)
        global_epoch += 1


if __name__ == '__main__':
    main()
