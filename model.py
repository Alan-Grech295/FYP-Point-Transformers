import time

import fpsample
import numpy as np
import torch
import torch.nn as nn

from models.Hengshuang.model import TransitionUp
from pointnet_util import farthest_point_sample, index_points, square_distance, pc_normalize
import torch.nn.functional as F


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]

    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


def sample_and_group_new(npoint, nsample, xyz, points):
    # fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    # try:
    #     import torch_fpsample
    #     _, fps_idx = torch_fpsample.sample(xyz, npoint)
    # except:
    #     import fpsample
    np_xyz: np.ndarray = xyz.cpu().numpy()
    fps_idxs = []
    for i in range(np_xyz.shape[0]):
        result = fpsample.bucket_fps_kdline_sampling(np_xyz[i], npoint, h=9)
        fps_idxs.append(torch.from_numpy(result.astype(np.int64)).to(xyz.device))

    fps_idx = torch.stack(fps_idxs, dim=0)

    new_xyz = index_points(xyz, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    return new_xyz, grouped_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    # def __init__(self, channels):
    #     super().__init__()
    #     self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
    #     self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
    #     self.v_conv = nn.Conv1d(channels, channels, 1)
    #     self.q_conv.weight = self.k_conv.weight
    #     self.trans_conv = nn.Conv1d(channels, channels, 1)
    #     self.after_norm = nn.BatchNorm1d(channels)
    #     self.act = nn.ReLU()
    #     self.softmax = nn.Softmax(dim=-1)
    #
    # def forward(self, x):
    #     x_q = self.q_conv(x).permute(0, 2, 1)  # (b, n, c//4)
    #     x_k = self.k_conv(x).permute(0, 2, 1)  # (b, n, c//4)
    #     x_v = self.v_conv(x).permute(0, 2, 1)  # (b, n, c)
    #
    #     energy = torch.bmm(x_q, x_k.transpose(1, 2))  # (b, n, n)
    #     energy = energy / (x_q.shape[-1] ** 0.5)  # Scaling for stability
    #     attention = self.softmax(energy)
    #
    #     x_r = torch.bmm(attention, x_v).permute(0, 2, 1)  # (b, c, n)
    #     x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
    #     x = x + x_r
    #     return x


class MultiHeadAttention(nn.Module):
    def __init__(self, channels=256, heads=4):
        super().__init__()
        assert channels % heads == 0, "Channels must be divisible by the number of heads"

        self.head_dim = channels // heads
        self.num_heads = heads
        self.heads = nn.ModuleList([SA_Layer(self.head_dim) for _ in range(heads)])
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, N = x.shape

        x_split = x.view(B, self.num_heads, self.head_dim, N)
        x_out = [self.heads[i](x_split[:, i, :, :]) for i in range(self.num_heads)]

        x = torch.cat(x_out, dim=1).permute(0, 2, 1)
        return self.proj(x).permute(0, 2, 1)


class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class MultiStackedAttention(nn.Module):
    def __init__(self, channels=256, heads=4, layers=4):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        self.sa = nn.ModuleList([MultiHeadAttention(channels, heads=heads) for _ in range(layers)])

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        x = self.input(x)  # B, D, N

        # results = []
        for sa in self.sa:
            x = sa(x)
            # results.append(x)

        # x = torch.cat(results, dim=1)

        return x


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x


class PointTransformerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.npoints = cfg.num_point
        output_channels = 3  # Normal vector
        d_points = cfg.input_dim
        channels = 256
        self.conv1 = nn.Conv1d(d_points, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

        self.sa1 = SA_Layer(256)
        self.sa2 = SA_Layer(256)
        self.sa3 = SA_Layer(256)
        self.sa4 = SA_Layer(256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.normal_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(64),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.convs1 = nn.Conv1d(1024 * 2, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 3, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        # self.linear1 = nn.Linear(1024, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=0.5)
        # self.linear3 = nn.Linear(256, output_channels)

        # Per-point output layer for normal prediction
        # self.fc_normals = nn.Sequential(
        #     nn.Conv1d(channels, 512, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 256, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Conv1d(256, output_channels, kernel_size=1)  # Outputs 3 values per point (x, y, z)
        # )

    def forward(self, x):
        xyz = x[..., :3]
        # x = x.permute(0, 2, 1)
        # batch_size, _, _ = x.size()
        # x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        # x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        #
        # x = x.permute(0, 2, 1)
        # new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        # feature_0 = self.gather_local_0(new_feature)
        # feature = feature_0.permute(0, 2, 1)
        # new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        # feature_1 = self.gather_local_1(new_feature)
        #
        # x = self.pt_last(feature_1)
        # x = torch.cat([x, feature_1], dim=1)
        #
        # x = self.conv_fuse(x).permute(0, 2, 1)
        # # 12 256 2048
        #
        # normals = self.fc_normals(x)
        # normals = F.normalize(normals, p=2, dim=1)

        x = x.permute(0, 2, 1)

        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)

        x_max = torch.max(x, 2, keepdim=True)[0].repeat(1, 1, N)
        x_avg = torch.mean(x, 2, keepdim=True).repeat(1, 1, N)

        global_features: torch.Tensor = self.global_pool(torch.cat((x_max, x_avg), dim=1)).squeeze(-1)
        global_features = global_features.unsqueeze(1).expand(-1, 1024, -1)

        #
        x = torch.cat((global_features, x), dim=1)

        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        normals = F.normalize(x, p=2, dim=1)

        return normals.permute(0, 2, 1)


def normalize_pc(xyz):
    centroid = xyz.mean(dim=1, keepdim=True)  # Compute mean per batch
    pc_centered = xyz - centroid  # Subtract mean to center at origin
    max_dist = torch.norm(pc_centered, dim=2, keepdim=True).max(dim=1, keepdim=True)[0]  # Find max distance per batch
    pc_normalized = pc_centered / (max_dist + 1e-8)  # Avoid division by zero
    return pc_normalized


class PointTransformerMat(nn.Module):
    class Prediction(nn.Module):
        def __init__(self, channels, out_channels):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(channels, channels // 2, kernel_size=1),
                nn.BatchNorm1d(channels // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(channels // 2, channels // 4, kernel_size=1),
                nn.BatchNorm1d(channels // 4),
                nn.ReLU(),
                nn.Conv1d(channels // 4, out_channels, kernel_size=1),
            )

        def forward(self, x):
            return self.layers(x)

    def __init__(self, cfg):
        super().__init__()
        self.npoints = cfg.num_point
        output_channels = 3  # Normal vector
        d_points = cfg.input_dim
        self.group_size = cfg.model.group_size
        internal_channels = 1024

        self.ie_input_size = 192
        self.ie_output_size = 128 - 3

        self.input_embedding = nn.Sequential(
            nn.Conv1d(self.ie_input_size, self.ie_input_size, kernel_size=1),
            nn.BatchNorm1d(self.ie_input_size),
            nn.ReLU(),
            nn.Conv1d(self.ie_input_size, self.ie_output_size, kernel_size=1),
            nn.BatchNorm1d(self.ie_output_size),
            nn.ReLU(),
        )

        self.conv_p3 = nn.Conv1d(2048, 256, kernel_size=1, bias=False)
        self.conv_p4 = nn.Conv1d(4096, 1024, kernel_size=1)
        # self.conv_up = nn.Conv1d(64, self.npoints, kernel_size=1)

        self.transition_up = nn.Sequential(nn.Conv1d(32, 128, kernel_size=1),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Conv1d(128, 256, kernel_size=1),
                                           nn.BatchNorm1d(256),
                                           nn.ReLU(),
                                           nn.Conv1d(256, self.npoints, kernel_size=1),
                                           nn.BatchNorm1d(self.npoints),
                                           nn.ReLU(),
                                           )

        self.down_feature1 = nn.Sequential(nn.Conv1d(32768, 16384, kernel_size=1),
                                           nn.BatchNorm1d(16384),
                                           nn.ReLU(),
                                           # nn.Conv1d(32768, 16384, kernel_size=1),
                                           # nn.BatchNorm1d(16384),
                                           # nn.ReLU(),
                                           nn.Conv1d(16384, 8192, kernel_size=1),
                                           nn.BatchNorm1d(8192),
                                           nn.ReLU(),
                                           )

        self.down_feature2 = nn.Sequential(nn.Conv1d(8192, 4096, kernel_size=1),
                                           nn.BatchNorm1d(4096),
                                           nn.ReLU(),
                                           # nn.Conv1d(8192, 4096, kernel_size=1),
                                           # nn.BatchNorm1d(4096),
                                           # nn.ReLU(),
                                           nn.Conv1d(4096, 1024, kernel_size=1),
                                           nn.BatchNorm1d(1024),
                                           nn.ReLU(),
                                           )

        # self.down_feature3 = nn.Sequential(nn.Conv1d(16384, 8192, kernel_size=1),
        #                                    nn.BatchNorm1d(8192),
        #                                    nn.ReLU(),
        #                                    nn.Conv1d(8192, 4096, kernel_size=1),
        #                                    nn.BatchNorm1d(4096),
        #                                    nn.ReLU(),
        #                                    nn.Conv1d(4096, 2048, kernel_size=1),
        #                                    nn.BatchNorm1d(2048),
        #                                    nn.ReLU(),
        #                                    )

        self.point_attention = MultiHeadAttention(128)

        self.grp_atn_1 = MultiStackedAttention(self.group_size * 128, heads=8, layers=2)
        self.grp_atn_2 = MultiStackedAttention(self.group_size * 512, heads=8)
        # self.grp_atn_1 = nn.MultiheadAttention(self.group_size * 128, num_heads=4)
        self.downscale = nn.Sequential(nn.Conv1d(self.group_size * 2048, self.group_size * 512, kernel_size=1),
                                       nn.BatchNorm1d(self.group_size * 512),
                                       nn.ReLU(), )
        # self.grp_atn_2 = nn.MultiheadAttention(self.group_size * 512, num_heads=4)

        self.conv1 = nn.Conv1d(d_points, internal_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(internal_channels, internal_channels, kernel_size=1, bias=False)

        # self.msa = MultiHeadAttention(internal_channels)

        # self.conv_fuse = nn.Sequential(nn.Conv1d(internal_channels, internal_channels, kernel_size=1, bias=False),
        #                                nn.BatchNorm1d(internal_channels),
        #                                nn.LeakyReLU(negative_slope=0.2))
        #
        # self.normal_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                                  nn.BatchNorm1d(64),
        #                                  nn.LeakyReLU(negative_slope=0.2))
        #
        # self.global_pool = nn.AdaptiveMaxPool1d(1)

        # self.convs1 = nn.Conv1d(1024 * 2, 512, 1)
        # self.dp1 = nn.Dropout(0.5)
        # self.convs2 = nn.Conv1d(512, 256, 1)
        # self.convs3 = nn.Conv1d(256, 3, 1)
        # self.bns1 = nn.BatchNorm1d(512)
        # self.bns2 = nn.BatchNorm1d(256)
        #
        self.relu = nn.ReLU()

        # self.albedo_metallic_head = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 7),  # Albedo (R, G, B), Metallic (R, G, B, A)
        #     nn.Sigmoid()  # To ensure values are between 0 and 1
        # )

        self.albedo_metallic_occlusion_head = PointTransformerMat.Prediction(internal_channels, 6)
        self.sig = nn.Sigmoid()

        # self.normals_head = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3),  # Normals (X, Y, Z)
        # )
        self.normals_head = PointTransformerMat.Prediction(internal_channels, 3)

    def round_down_to_power_of_2(self, n):
        if n == 0:
            return 0  # Edge case: if n is 0, return 0
        n |= (n >> 1)
        n |= (n >> 2)
        n |= (n >> 4)
        n |= (n >> 8)
        n |= (n >> 16)
        return n - (n >> 1)  # Subtract half to get closest lower power of

    def rotation_matrix_from_normals(self, normals, up=(0, 1, 0)):
        def normalize(v):
            """ Normalizes a tensor along the last dimension, avoiding division by zero. """
            return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

        up = torch.tensor(up, dtype=torch.float32, device=normals.device).expand_as(normals)

        # Compute rotation axis (cross product)
        axis = torch.cross(normals, up, dim=-1)
        axis = normalize(axis)  # Normalize axis

        # Compute rotation angle (dot product and arccos)
        dot = torch.sum(normals * up, dim=-1, keepdim=True).clamp(-1, 1)
        angle = torch.acos(dot)  # Angle in radians

        # Rodrigues' formula components
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Skew-symmetric matrix of axis
        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        K = torch.stack([
            torch.zeros_like(x), -z, y,
            z, torch.zeros_like(x), -x,
            -y, x, torch.zeros_like(x)
        ], dim=-1).view(*axis.shape[:-1], 3, 3)

        # Compute rotation matrix: R = I + sin(angle) * K + (1 - cos(angle)) * K^2
        I = torch.eye(3, device=normals.device).expand(*axis.shape[:-1], 3, 3)
        R = I + sin_a.unsqueeze(-1) * K + (1 - cos_a.unsqueeze(-1)) * (K @ K)

        return R  # Shape: (B, N, 3, 3)

    def rotate_directions(self, new_x, normals):
        """ Rotates only the view directions in an interleaved (dir, rad) tensor. """
        B, N, k6 = new_x.shape
        k = k6 // 6  # Number of selected viewpoints

        view_dir_indices = torch.tensor([i for i in range(k6) if ((i // 3) % 2) == 0], device=new_x.device)
        rad_indices = view_dir_indices + 3

        # Extract interleaved view directions and radiances
        view_dirs = new_x[..., view_dir_indices].contiguous().view(B, N, k, 3)  # (B, N, k, 3)
        radiances = new_x[..., rad_indices].contiguous().view(B, N, k, 3)  # (B, N, k, 3)

        # Compute rotation matrix
        R = self.rotation_matrix_from_normals(normals, up=(0, 0, 1))  # Shape (B, N, 3, 3)

        # Rotate view directions: R @ view_dirs
        rotated_dirs = torch.einsum("b n i j, b n k j -> b n k i", R, view_dirs)

        # Re-interleave rotated view directions and original radiances
        rotated_x = torch.empty_like(new_x)  # Same shape (B, N, k*6)

        # Fill in rotated view directions
        rotated_x[..., view_dir_indices] = rotated_dirs.reshape(B, N, -1)
        # Keep radiances unchanged
        rotated_x[..., rad_indices] = radiances.reshape(B, N, -1)

        return rotated_x

    def forward(self, x, normals):
        torch.set_printoptions(sci_mode=False, precision=10)
        B, N, C = x.shape

        xyz = x[..., :3]
        xyz = normalize_pc(xyz)
        new_x = x[..., 3:]

        # Gets the dot products of the view directions to the normal
        dir_indices = [i for i in range(new_x.shape[-1]) if ((i // 3) % 2) == 0]
        num_viewpoints = new_x.shape[-1] // 6
        dirs = new_x[..., dir_indices].view(B, N, num_viewpoints, 3)
        norm = torch.norm(dirs, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm))
        dot_products = torch.sum(dirs * normals.unsqueeze(2), dim=-1)

        # Gathers k-closest directions to the normal
        k = 32
        _, top_k_i = torch.topk(dot_products, k, dim=-1)

        # Chooses the view directions
        top_k_i = top_k_i.unsqueeze(-1) * 6

        # Create indices for the 6 values (direction + radiance) per viewpoint
        offsets = torch.arange(6, device=x.device).view(1, 1, 6)  # (1, 1, 1, 6)
        top_k_i = (top_k_i + offsets).view(B, N, -1)

        new_x = torch.gather(new_x, 2, top_k_i)
        new_x = self.rotate_directions(new_x, normals)

        new_x = new_x.permute(0, 2, 1)

        # new_x = self.conv_p2(self.point_attention(self.conv_p1(new_x))).permute(0, 2, 1)
        new_x = self.input_embedding(new_x).permute(0, 2, 1)
        new_x = torch.cat((xyz, new_x), dim=-1)

        n_point = self.round_down_to_power_of_2(N // self.group_size)
        new_xyz, new_points = sample_and_group_new(n_point, self.group_size, xyz, new_x)

        new_points = new_points.view(B, -1, new_points.shape[-1] * new_points.shape[-2])

        # new_points = new_points.permute(1, 0, 2)
        new_points = new_points.permute(0, 2, 1)
        new_points = self.grp_atn_1(new_points).permute(0, 2, 1)
        # new_points = new_points.permute(1, 0, 2)

        # new_points = self.relu(self.conv_p3(new_points).permute(0, 2, 1))

        n_point //= self.group_size
        new_xyz, new_points = sample_and_group_new(n_point, self.group_size, new_xyz, new_points)

        new_points = new_points.view(B, -1, new_points.shape[-1] * new_points.shape[-2])

        new_points = self.downscale(new_points.permute(0, 2, 1)).permute(0, 2, 1)

        # new_points = self.relu(self.conv_p4(self.grp_atn_2(new_points.permute(0, 2, 1))))
        # new_points = new_points.permute(1, 0, 2)
        new_points = new_points.permute(0, 2, 1)
        new_points = self.grp_atn_2(new_points).permute(0, 2, 1)
        # new_points = new_points.permute(1, 0, 2)

        # new_points = self.down_feature1(new_points)

        # new_points = self.down_feature1(new_points.permute(0, 2, 1)).permute(0, 2, 1)
        new_points = self.transition_up(new_points).permute(0, 2, 1)
        new_points = self.down_feature2(new_points)

        # new_points = self.conv_up(new_points.permute(0, 2, 1)).permute(0, 2, 1)

        albedo_metallic_occ = self.sig(self.albedo_metallic_occlusion_head(new_points)).permute(0, 2, 1)
        normals = F.normalize(self.normals_head(new_points), p=2, dim=-1).permute(0, 2, 1)

        return albedo_metallic_occ[..., :3], albedo_metallic_occ[..., 3:5], albedo_metallic_occ[..., 5:], normals
        # x = x.permute(0, 2, 1)
        #
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        #
        # x = self.msa(x)
        #
        # x = self.conv_fuse(x)
        #
        # # x = x.permute(0, 2, 1)
        #
        # albedo_metallic_occ = self.sig(self.albedo_metallic_occlusion_head(x)).permute(0, 2, 1)
        # normals = F.normalize(self.normals_head(x), p=2, dim=-1).permute(0, 2, 1)
        #
        # return albedo_metallic_occ[..., :3], albedo_metallic_occ[..., 3:5], albedo_metallic_occ[..., 5:], normals
