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
from typing import List, Literal, Tuple, Union

import hydra
import omegaconf
import torch
import torch.nn.functional as F
from scipy import stats
from skimage.color import rgb2lab, deltaE_ciede2000
from torch import nn, autograd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MaterialDataset
from timer import Timer
from trial_manager import TrialManager
import utils

# torch.autograd.set_detect_anomaly(True)

# TODO: REMOVE
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def check_nan(**kwargs):
    for k, v in kwargs.items():
        if torch.isnan(v).any():
            print(f"{k} is NaN")


def calculate_weight_stats(model):
    """
    Calculate min, max, mean, and standard deviation of all trainable weights in a PyTorch model.

    Args:
        model (torch.nn.Module): The model whose weights need to be analyzed.

    Returns:
        dict: A dictionary containing the statistics.
    """
    weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])

    stats = {
        "min": torch.min(weights).item(),
        "max": torch.max(weights).item(),
        "mean": torch.mean(weights).item(),
        "std_dev": torch.std(weights).item()
    }

    return stats


def check_negative(val):
    assert torch.all(val >= 0), "Invalid"


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, color1, color2):
        """
        Computes the perceptual color difference (ΔE 2000) between two colors in CIELAB space.

        Args:
            color1: Tensor of shape (..., 3) in RGB format, values in [0, 1]
            color2: Tensor of shape (..., 3) in RGB format, values in [0, 1]

        Returns:
            Loss value representing the ΔE 2000 color difference.
        """
        # Convert RGB to LAB
        # lab1 = self.rgb_to_lab(color1)
        # lab2 = self.rgb_to_lab(color2)

        # lab1 = self.normalize_lab(lab1)
        # lab2 = self.normalize_lab(lab2)

        # sklab1 = self.rgb_to_lab_scikit(color1)
        # sklab2 = self.rgb_to_lab_scikit(color2)

        # delta_e_sk = torch.tensor(deltaE_ciede2000(sklab1, sklab2), dtype=torch.float32, device=color1.device)

        # delta_e = self.delta_e_2000(lab1, lab2)

        # assert torch.allclose(delta_e_sk, delta_e, atol=1e-2)

        # return (delta_e / 100).mean()

        # l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        # l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        # loss = 0.6 * F.mse_loss(l1, l2) + 0.2 * F.mse_loss(a1, a2) + 0.2 * F.mse_loss(b1, b2)
        return F.mse_loss(color1 * 255, color2 * 255)

    # ChatGPT generated, converted from Scikit
    def delta_e_2000(self, lab1, lab2, kL=1, kC=1, kH=1):
        """
        Compute the CIEDE2000 color difference between two LAB color tensors.

        Args:
            lab1: Tensor of shape (..., 3) representing LAB colors.
            lab2: Tensor of shape (..., 3) representing LAB colors.
            kL: Lightness scale factor (default=1).
            kC: Chroma scale factor (default=1).
            kH: Hue scale factor (default=1).

        Returns:
            deltaE: Tensor of shape (...) representing the color difference.
        """
        assert lab1.shape[-1] == 3 and lab2.shape[-1] == 3, "Inputs must have shape (..., 3)"

        def _cart2polar_2pi(x, y):
            """convert cartesian coordinates to polar (uses non-standard theta range!)

            NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than usual ``(-pi, +pi)``
            """
            r, t = torch.sqrt(x ** 2 + y ** 2), torch.arctan2(y, x)
            t += torch.where(t < 0, 2 * torch.pi, torch.tensor(0.0, device=t.device))
            return r, t

        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        check_negative(a1 ** 2 + b1 ** 2 + 1e-6)
        check_negative(a2 ** 2 + b2 ** 2 + 1e-6)
        CBar = 0.5 * (torch.sqrt(a1 ** 2 + b1 ** 2 + 1e-6) + torch.sqrt(a2 ** 2 + b2 ** 2 + 1e-6))
        c7 = CBar ** 7
        G = 0.5 * (1 - torch.sqrt(c7 / (c7 + 25 ** 7) + 1e-6))
        scale = 1 + G
        C1, h1 = _cart2polar_2pi(a1 * scale, b1)
        C2, h2 = _cart2polar_2pi(a2 * scale, b2)

        check_nan(CBar=CBar, G=G, C1=C1, C2=C2, h1=h1, h2=h2)

        # lightness term
        Lbar = 0.5 * (L1 + L2)
        tmp = (Lbar - 50) ** 2
        SL = 1 + 0.015 * tmp / torch.sqrt(20 + tmp)
        L_term = (L2 - L1) / (kL * SL)
        check_negative(20 + tmp)

        check_nan(Lbar=Lbar, tmp=tmp, SL=SL, L_term=L_term)

        # chroma term
        Cbar = 0.5 * (C1 + C2)  # new coordinates
        SC = 1 + 0.045 * Cbar
        C_term = (C2 - C1) / (kC * SC)

        check_nan(Cbar=Cbar, SC=SC, C_term=C_term)

        # hue term
        h_diff = h2 - h1
        h_sum = h1 + h2
        CC = C1 * C2

        check_nan(h_diff=h_diff, h_sum=h_sum, CC=CC)

        dH = h_diff.detach().clone()
        dH[h_diff > torch.pi] -= 2 * torch.pi
        dH[h_diff < -torch.pi] += 2 * torch.pi
        dH[CC == 0.] = 0.  # if r == 0, dtheta == 0
        dH_term = 2 * torch.sqrt(CC + 1e-6) * torch.sin(dH / 2)
        check_negative(CC + 1e-6)

        check_nan(dH=dH, dH_term=dH_term)

        Hbar = h_sum.detach().clone()
        mask = torch.logical_and(CC != 0., torch.abs(h_diff) > torch.pi)
        Hbar[mask * (h_sum < 2 * torch.pi)] += 2 * torch.pi
        Hbar[mask * (h_sum >= 2 * torch.pi)] -= 2 * torch.pi
        Hbar[CC == 0.] *= 2
        Hbar *= 0.5

        check_nan(Hbar=Hbar)

        T = (1 -
             0.17 * torch.cos(Hbar - torch.deg2rad(torch.tensor(30.0, device=Hbar.device))) +
             0.24 * torch.cos(2 * Hbar) +
             0.32 * torch.cos(3 * Hbar + torch.deg2rad(torch.tensor(6.0, device=Hbar.device))) -
             0.20 * torch.cos(4 * Hbar - torch.deg2rad(torch.tensor(63.0, device=Hbar.device)))
             )
        SH = 1 + 0.015 * Cbar * T
        H_term = dH_term / (kH * SH)
        check_nan(SH=SH, H_term=H_term)

        c7 = Cbar ** 7
        Rc = 2 * torch.sqrt(c7 / (c7 + 25 ** 7) + 1e-6)
        check_negative(c7 / (c7 + 25 ** 7) + 1e-6)
        dtheta = torch.deg2rad(torch.tensor(30.0, device=Hbar.device)) * torch.exp(
            -((torch.rad2deg(Hbar) - 275) / 25) ** 2)
        R_term = -torch.sin(2 * dtheta) * Rc * C_term * H_term

        check_nan(Rc=Rc, dtheta=dtheta, R_term=R_term)

        # put it all together
        dE2 = L_term ** 2
        dE2 += C_term ** 2
        dE2 += H_term ** 2
        dE2 += R_term
        ans = torch.sqrt(torch.clamp(dE2, min=0))
        check_negative(torch.clamp(dE2, min=0))

        check_nan(dE2=dE2)

        return ans

    def normalize_lab(self, lab: torch.Tensor):
        """
        Normalize a LAB tensor from standard LAB ranges to [0,1].

        Args:
            lab: Tensor of shape (b, n, 3) in LAB color space.

        Returns:
            Normalized LAB tensor with values in [0,1].
        """
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

        L = L / 100  # Normalize L to [0,1]
        a = (a + 128) / 255  # Normalize a to [0,1]
        b = (b + 128) / 255  # Normalize b to [0,1]

        return torch.stack([L, a, b], dim=-1)

    def rgb_to_lab_scikit(self, rgb):
        """ Converts RGB tensor (0-1 range) to LAB color space. """
        # Convert to 0-255 range for compatibility with skimage
        rgb_np = rgb.cpu().detach()
        lab_np = rgb2lab(rgb_np, channel_axis=-1)  # Convert to LAB using skimage
        return lab_np
        # return torch.tensor(lab_np, dtype=torch.float32, device=rgb.device, requires_grad=True)

    # ChatGPT generated, with values adapted from Scikit image
    def rgb_to_lab(self, rgb: torch.Tensor):
        """
        Convert an RGB tensor (0-1 range) to LAB color space.

        Args:
            rgb: Tensor of shape (batch, n, 3) with RGB values in [0,1].

        Returns:
            lab: Tensor of shape (batch, n, 3) in LAB color space.
        """
        assert rgb.shape[-1] == 3, "Input must have shape (batch, n, 3)"
        assert not torch.isnan(
            rgb).any(), f"Input must not have nan values ({torch.isnan(rgb).sum().item()} / {rgb.numel()})"
        assert torch.all(rgb >= 0) and torch.all(rgb <= 1), f"Input must have values between 0 and 1"

        check_nan(input_rgb=rgb)

        # Convert sRGB to linear RGB
        mask = rgb > 0.04045
        rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        check_nan(rgb_linear=rgb_linear)

        # RGB to XYZ transformation matrix
        M = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], dtype=torch.float32, device=rgb.device)

        # Convert RGB to XYZ (batch-wise matrix multiplication)
        xyz = torch.einsum('...ij,jk->...ik', rgb_linear, M.T)

        check_nan(xyz=xyz)

        # Normalize XYZ by reference white point (D65)
        xyz_ref_white = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32, device=rgb.device)
        xyz = xyz / xyz_ref_white

        check_nan(xyz_ref_white=xyz_ref_white, xyz_norm=xyz)

        # Nonlinear transformation for LAB
        epsilon = 0.008856
        kappa = 903.3
        mask = xyz > epsilon
        xyz_f = torch.where(mask, xyz ** (1 / 3), (kappa * xyz + 16) / 116)

        check_nan(xyz_f=xyz_f)

        # Compute L, a, b
        L = (116 * xyz_f[..., 1]) - 16
        a = 500 * (xyz_f[..., 0] - xyz_f[..., 1])
        b = 200 * (xyz_f[..., 1] - xyz_f[..., 2])

        return torch.stack([L, a, b], dim=-1)


@hydra.main(config_path='config', config_name='mat_pred', version_base=None)
def main(args):
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    omegaconf.OmegaConf.set_struct(args, False)

    cl = ColorLoss()
    # color1 = torch.tensor([[1,1,1]], dtype=torch.float32)  # RGB [0-1] range
    # color2 = torch.tensor([[1,1,1]], dtype=torch.float32)
    # print(cl(color1, color2))

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # root = hydra.utils.to_absolute_path('data/material_pred/')
    if platform.system() == "Windows":
        root = Path(r"E:\\FYP Dataset\\32768_64\\")
    else:
        root = Path(r"/home/alan/Dataset/32768_64/")
    train_dataset = MaterialDataset(root=root / "train/Outdoor/DirLight", npoints=args.num_point, num_samples_per_ds=2,
                                    dataset_type="clean")
    test_dataset = MaterialDataset(root=root / "test/Outdoor/DirLight", npoints=args.num_point, num_samples_per_ds=2,
                                   dataset_type="clean")
    trainDataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=8, persistent_workers=True)

    testDataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=8, persistent_workers=True)

    '''MODEL LOADING'''
    args.input_dim = 3 + (3 + 3) * 64
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    # classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerMat')(
    #     args)
    # try:
    #     classifier = torch.compile(classifier, backend="eager")
    # except:
    #     pass
    #
    # classifier = classifier.cuda()

    criterion = torch.nn.CosineSimilarity(dim=-1)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    TrialManager.add_trials("default")

    def calc_loss(pred_albedo, pred_metallic, pred_hdr, target_albedo, target_metallic, target_hdr,
                  total_included: List[
                      Union[Literal['albedo', 'metallic', 'smoothness', 'normals', 'occlusion', 'hdr'], Tuple[
                          str, float]]]):

        def diff_loss(target: torch.Tensor, pred: torch.Tensor):
            # target_diffs = target.unsqueeze(1) - target.unsqueeze(2)
            # pred_diffs = pred.unsqueeze(1) - pred.unsqueeze(2)
            return F.mse_loss(target, pred)

        # loss_norm = (1. - criterion(pred_norm, target_norm).mean())
        # loss_albedo = F.mse_loss(pred_albedo * 255, target_albedo * 255)  # cl(pred_albedo, target_albedo)
        loss_albedo = cl(pred_albedo, target_albedo)
        loss_metallic = diff_loss(pred_metallic[..., 0] * 255, target_metallic[..., 0] * 255)
        loss_smooth = diff_loss(pred_metallic[..., 1] * 255, target_metallic[..., 1] * 255)

        # loss_occ = F.mse_loss(pred_occ * 255, target_occ * 255)

        def remap_hdr(hdr, mean=None, std=None):
            transformed = torch.log(hdr + 1e-4)
            mean = mean if mean is not None else torch.mean(transformed)
            std = std if std is not None else torch.std(transformed)
            norm_dist = torch.distributions.Normal(mean, std)
            transformed = norm_dist.cdf(transformed)
            return transformed, mean, std

        def get_hdr_mult(hdr, min=1., max=10., k=3.):
            return torch.exp(-hdr * k) * (1. - hdr) * (max - min) + min

        target_hdr, mean, std = remap_hdr(target_hdr)
        pred_hdr, _, _ = remap_hdr(pred_hdr, mean=mean, std=std)

        # loss_hdr_mult = get_hdr_mult(target_hdr, min=1, max=10, k=3)
        # loss_hdr = torch.mean(F.mse_loss(pred_hdr * 255, target_hdr * 255, reduction='none') * loss_hdr_mult)
        loss_hdr = F.mse_loss(pred_hdr * 255, target_hdr * 255)

        total_loss = torch.Tensor([0]).float().cuda()

        for included in total_included:
            if type(included) is tuple:
                loss_name, weight = included
            else:
                loss_name, weight = included, 1

            if loss_name == 'albedo':
                total_loss += loss_albedo * weight
            elif loss_name == 'metallic':
                total_loss += loss_metallic * weight
            elif loss_name == 'smoothness':
                total_loss += loss_smooth * weight
            elif loss_name == 'normals':
                # total_loss += loss_norm
                pass
            elif loss_name == 'occlusion':
                # total_loss += loss_occ * weight
                pass
            elif loss_name == 'hdr':
                total_loss += loss_hdr * weight

        return total_loss, loss_albedo.item(), loss_metallic.item(), loss_smooth.item(), loss_hdr.item()

    while TrialManager.next_trial():
        print(f"Starting trial run {TrialManager().trial_name}...")

        global_epoch = 0
        best_total_loss = 999999999999999999999

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
            checkpoint = torch.load(f'best_model_mat_pred_{TrialManager().trial_name}.pth')
            uncompiled_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            start_epoch = checkpoint['epoch']
            best_total_loss = checkpoint['best_total_loss']
            logger.info('Use pretrain model')
        except:
            logger.info('No existing model, starting training from scratch...')
            start_epoch = 0

        pytorch_total_params = sum(p.numel() for p in classifier.parameters())
        pytorch_trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        print(pytorch_total_params, pytorch_trainable_params)

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

        csv_path = f"./material_prediction_trial_{TrialManager().trial_name}.csv"
        # if not os.path.exists(csv_path):
        write_header_csv(csv_path)

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
            for i, (data, target) in tqdm(enumerate(trainDataLoader),
                                          total=len(trainDataLoader),
                                          smoothing=0.9):
                # points = provider.rotate_point_cloud_with_normal(points)

                # target = torch.Tensor(points[:, :, 3:])
                #
                # points = points[:, :, :3]
                #
                # points = provider.random_scale_point_cloud(points)
                # points = provider.shift_point_cloud(points)
                # Slow part
                # with autograd.detect_anomaly():
                data = torch.Tensor(data).float().cuda()
                target_albedo, target_metallic, target_norm, target_occ, target_hdr = (
                    torch.Tensor(target[..., :3]).float().cuda(),
                    torch.Tensor(
                        target[..., 4:6]).float().cuda(),
                    torch.Tensor(
                        target[..., 6:9]).float().cuda(),
                    torch.Tensor(
                        target[..., 10:11]).float().cuda(),
                    torch.Tensor(
                        target[..., 11:]).float().cuda()
                )

                # target_light = torch.Tensor(target_light[:, ::2, ::2, ::2, :]).float().cuda().reshape(
                #     target_light.shape[0], -1, target_light.shape[-1])[..., 0].unsqueeze(-1)
                # light_min_bounds = torch.Tensor(light_min_bounds).float().cuda()
                # light_max_bounds = torch.Tensor(light_max_bounds).float().cuda()

                # points, target = points.float().cuda(), target.float().cuda()
                optimizer.zero_grad()

                pred_albedo, pred_metallic, pred_occ, pred_hdr, radiance_indices = classifier(data, target_norm)

                radiance_indices = radiance_indices.unsqueeze(-1) * 3

                # Create indices for the 3 values (radiance) per viewpoint
                offsets = torch.arange(3, device=radiance_indices.device).view(1, 1, 3)  # (1, 1, 1, 6)
                radiance_indices = (radiance_indices + offsets).view(radiance_indices.shape[0],
                                                                     radiance_indices.shape[1], -1)

                target_hdr = torch.gather(target_hdr, -1, radiance_indices)

                # cosine_similarity = F.cosine_similarity(pred_normals, target, dim=-1)
                # train_cosine_similarity += cosine_similarity.sum().item()
                # train_points_seen += points.size(0) * points.size(1)
                # seg_pred = seg_pred.contiguous().view(-1, num_part)
                # target = target.view(-1, 1)[:, 0]
                # pred_choice = seg_pred.data.max(1)[1]

                # correct = pred_choice.eq(target.data).cpu().sum()
                # mean_correct.append(correct.item() / (args.batch_size * args.num_point))
                total_loss, _, _, _, _ = calc_loss(pred_albedo, pred_metallic, pred_hdr,
                                                   target_albedo, target_metallic, target_hdr,
                                                   [('albedo', 0), ('metallic', 2), ('smoothness', 2), ('hdr', 1)])

                train_loss += total_loss.item()
                train_points_seen += 1

                total_loss.backward()

                optimizer.step()

            utils.print_gradient_norm_named(classifier)

            # model_stats = calculate_weight_stats(classifier)
            # print(
            #     f"Model stats - min: {model_stats['min']}, max: {model_stats['max']}, mean: {model_stats['mean']}, \
            #     std. dev.: {model_stats['std_dev']}")
            # logger.info(f"Positional Encoding Gamma: {classifier.pe_gamma.item()}")
            train_acc = train_loss / train_points_seen
            logger.info('Train loss is: %.5f' % train_acc)

            with torch.no_grad():
                test_metrics = {}
                test_loss = 0
                total_albedo_loss = 0
                total_metallic_loss = 0
                total_smoothness_loss = 0
                total_hdr_loss = 0
                total_lighting_loss = 0
                total_normal_loss = 0
                # total_cosine_similarity = 0
                total_seen = 0

                classifier = classifier.eval()

                for batch_id, (data, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                     smoothing=0.9):
                    data = torch.Tensor(data).float().cuda()
                    target_albedo, target_metallic, target_norm, target_occ, target_hdr = (
                        torch.Tensor(target[..., :3]).float().cuda(),
                        torch.Tensor(
                            target[..., 4:6]).float().cuda(),
                        torch.Tensor(
                            target[..., 6:9]).float().cuda(),
                        torch.Tensor(
                            target[..., 10:11]).float().cuda(),
                        torch.Tensor(
                            target[..., 11:]).float().cuda()
                    )

                    # target_light = torch.Tensor(target_light[:, ::2, ::2, ::2, :]).float().cuda().reshape(
                    #     target_light.shape[0], -1, target_light.shape[-1])[..., 0].unsqueeze(-1)
                    # light_min_bounds = torch.Tensor(light_min_bounds).float().cuda()
                    # light_max_bounds = torch.Tensor(light_max_bounds).float().cuda()

                    pred_albedo, pred_metallic, pred_occ, pred_hdr, radiance_indices = classifier(data, target_norm)

                    radiance_indices = radiance_indices.unsqueeze(-1) * 3

                    # Create indices for the 3 values (radiance) per viewpoint
                    offsets = torch.arange(3, device=radiance_indices.device).view(1, 1, 3)  # (1, 1, 1, 6)
                    radiance_indices = (radiance_indices + offsets).view(radiance_indices.shape[0],
                                                                         radiance_indices.shape[1], -1)

                    target_hdr = torch.gather(target_hdr, -1, radiance_indices)

                    total_loss, loss_albedo, loss_metallic, loss_smoothness, loss_hdr = calc_loss(pred_albedo,
                                                                                                  pred_metallic,
                                                                                                  pred_hdr,
                                                                                                  target_albedo,
                                                                                                  target_metallic,
                                                                                                  target_hdr,
                                                                                                  [('albedo', 0),
                                                                                                   ('metallic', 2),
                                                                                                   ('smoothness', 2),
                                                                                                   ('hdr', 1)])

                    total_albedo_loss += loss_albedo
                    total_metallic_loss += loss_metallic
                    total_smoothness_loss += loss_smoothness
                    # total_lighting_loss += loss_lighting
                    total_hdr_loss += loss_hdr

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
                test_metrics['smoothness_loss'] = total_smoothness_loss / total_seen
                test_metrics['hdr_loss'] = total_hdr_loss / total_seen
                test_metrics['lighting_loss'] = total_lighting_loss / total_seen
                test_metrics['total_loss'] = test_loss / total_seen
                # test_metrics['cosine_similarity'] = total_cosine_similarity / total_seen

            logger.info(
                'Epoch %d - Albedo Loss: %f, Metallic Loss: %f, Smoothness Loss: %f, HDR Loss: %f, Lighting Loss: %f, Total Loss: %f' % (
                    global_epoch + 1, test_metrics['albedo_loss'], test_metrics['metallic_loss'],
                    test_metrics['smoothness_loss'],
                    test_metrics['hdr_loss'], test_metrics['lighting_loss'],
                    test_metrics['total_loss']))

            write_stats_csv(csv_path, global_epoch + 1, train_acc,
                            test_metrics['total_loss'])

            if test_metrics['total_loss'] <= best_total_loss:
                best_total_loss = test_metrics['total_loss']
                logger.info('Save model...')
                savepath = f'best_model_mat_pred_{TrialManager().trial_name}.pth'
                logger.info('Saving at %s' % savepath)
                state = {
                    'epoch': global_epoch,
                    'train_acc': train_acc,
                    'best_total_loss': best_total_loss,
                    'model_state_dict': uncompiled_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                logger.info('Saving model....')

            # if test_metrics['total_loss'] < best_acc:
            #     best_acc = test_metrics['total_loss']

            logger.info('Best loss is: %.5f' % best_total_loss)
            global_epoch += 1


if __name__ == '__main__':
    main()
