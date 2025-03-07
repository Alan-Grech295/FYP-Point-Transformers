import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import fpsample
import numpy as np
import torch
import torch.nn as nn
# from learnable_fourier_positional_encoding import LearnableFourierPositionalEncoding

from models.Hengshuang.model import TransitionUp
from pointnet_util import farthest_point_sample, index_points, square_distance, pc_normalize
import torch.nn.functional as F

from trial_manager import TrialManager


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
    np_xyz: np.ndarray = xyz.cpu().detach().numpy()
    fps_idxs = []
    for i in range(np_xyz.shape[0]):
        result = fpsample.bucket_fps_kdline_sampling(np_xyz[i], npoint, h=9)
        fps_idxs.append(torch.from_numpy(result.astype(np.int64)).to(xyz.device))

    fps_idx = torch.stack(fps_idxs, dim=0)

    new_xyz = index_points(xyz, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    return new_xyz, grouped_points, idx


def reconstruct_points(grouped_points, idx, original_num_points):
    """
    Reconstruct the original points tensor using grouped points and indices.

    Args:
        grouped_points (torch.Tensor): Grouped points of shape (B, npoint, nsample, C).
        idx (torch.Tensor): Indices of grouped points of shape (B, npoint, nsample).
        original_num_points (int): Original number of points (N) in the input.

    Returns:
        torch.Tensor: Reconstructed points tensor of shape (B, N, C).
    """
    B, npoint, nsample, C = grouped_points.shape
    device = grouped_points.device

    # Flatten indices and grouped points
    flat_idx = idx.reshape(B, -1)  # (B, npoint*nsample)
    flat_grouped = grouped_points.reshape(B, -1, C)  # (B, npoint*nsample, C)

    # Initialize tensors for reconstruction and counts
    reconstructed = torch.zeros((B, original_num_points, C), device=device)
    counts = torch.zeros((B, original_num_points, 1), device=device)

    # Accumulate grouped points into reconstructed and count occurrences
    reconstructed.scatter_add_(
        dim=1,
        index=flat_idx.unsqueeze(-1).expand(-1, -1, C),
        src=flat_grouped
    )
    counts.scatter_add_(
        dim=1,
        index=flat_idx.unsqueeze(-1),
        src=torch.ones((B, npoint * nsample, 1), device=device)
    )

    # Avoid division by zero (set counts to 1 where zero)
    counts = counts.clamp(min=1)

    # Average the accumulated points
    reconstructed = reconstructed / counts

    return reconstructed


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


"""
# Taken from https://github.com/BreaGG/Attention_Is_All_You_Need_From_Scratch/blob/main/transformer_model.py
# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.nn.functional.softmax(scores, dim=-1) + 1e-9
        output = torch.matmul(attn, V)
        return output, attn


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.W_q = nn.Linear(d_model, d_model // 2)
        self.W_k = nn.Linear(d_model, d_model // 2)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V):
        batch_size = Q.shape[0]
        Q = self.W_q(Q).view(batch_size, -1, self.heads, self.d_k // 2).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.heads, self.d_k // 2).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        scores, attn = self.attention(Q, K, V)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        output = self.fc(scores)
        return output


# Feedforward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual connection and layer normalization
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward network with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, d_model, layers, heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(heads, d_model)
        self.encoder_attention = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        # Self-attention with residual connection and layer normalization
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Attention over encoder output
        attn_output = self.encoder_attention(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(attn_output))

        # Feedforward network with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        for layer in self.layers:
            x = layer(x, enc_output)
        return x
#"""


# -------------------------------------

class SA_Layer(nn.Module):
    def __init__(self, channels, div=4):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // div, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // div, 1, bias=False)
        # self.q_conv.weight = self.k_conv.weight
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
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
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
    def __init__(self, channels=256, heads=4, div=4):
        super().__init__()
        assert channels % heads == 0, "Channels must be divisible by the number of heads"

        self.head_dim = channels // heads
        self.num_heads = heads
        self.heads = nn.ModuleList([SA_Layer(self.head_dim, div=div) for _ in range(heads)])
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, N = x.shape

        x_split = x.view(B, self.num_heads, self.head_dim, N)
        x_out = [self.heads[i](x_split[:, i, :, :]) for i in range(self.num_heads)]

        x = torch.cat(x_out, dim=1).permute(0, 2, 1)
        return self.proj(x).permute(0, 2, 1)


# """

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
    def __init__(self, channels=256, heads=4, layers=4, div=4, dropout=0.2, skip=False):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        self.skip = skip

        # self.sa = nn.ModuleList([MultiHeadAttention(heads=heads, d_model=channels) for _ in range(layers)])
        # self.sa = nn.ModuleList([MultiHeadAttention(channels, heads=heads, div=div) for _ in range(layers)])
        self.sa = nn.ModuleList([nn.MultiheadAttention(embed_dim=channels, num_heads=heads) for _ in range(layers)])
        self.linear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, channels),
                nn.Dropout(dropout)
            ) for _ in range(layers)
        ])
        self.norm = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        x = x.permute(0, 2, 1)

        x = self.input(x)  # B, D, N
        x = x.permute(2, 0, 1)

        for sa_layer, linear_layer, norm_layer in zip(self.sa, self.linear, self.norm):
            residual = x.permute(1, 2, 0)
            x_attn, _ = sa_layer(x, x, x)
            x = norm_layer(residual + self.dropout(x_attn).permute(1, 2, 0)).permute(2, 0, 1)
            x = linear_layer(x) + x  # Add skip connection

        # x = x.permute(0, 2, 1)
        x = x.permute(2, 0, 1)

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


# class PositionalEncoding3D(nn.Module):
#     def __init__(self, d_model: int, max_freq: int = 10000):
#         """
#         Implements the positional encoding from the original Transformer paper but adapted for 3D coordinates.
#
#         :param d_model: The number of dimensions for the encoding.
#         :param max_freq: The scaling factor for frequency encoding.
#         """
#         super().__init__()
#         self.d_model = d_model
#         self.max_freq = max_freq
#         self.encoding_dim = d_model // 3  # Divide across X, Y, and Z
#
#     def forward(self, coords: torch.Tensor):
#         """
#         Apply sinusoidal positional encoding to 3D coordinates.
#
#         :param coords: (B, N, 3) tensor containing 3D point coordinates.
#         :return: (B, N, d_model) tensor with positionally encoded coordinates.
#         """
#         B, N, _ = coords.shape  # (batch, num_points, 3)
#         device = coords.device
#
#         # Create frequency bands
#         i = torch.arange(0, self.encoding_dim // 2, device=device, dtype=torch.float32)
#         div_term = torch.pow(self.max_freq, -2 * i / self.encoding_dim)  # Shape: (encoding_dim // 2,)
#
#         # Expand for broadcasting
#         coords = coords.unsqueeze(-1)  # Shape: (B, N, 3, 1)
#         div_term = div_term.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, encoding_dim // 2)
#
#         # Compute the positional encodings
#         encoded_coords = torch.cat([
#             torch.sin(coords * div_term),
#             torch.cos(coords * div_term)
#         ], dim=-1)  # Shape: (B, N, 3, encoding_dim)
#
#         # Reshape to (B, N, d_model)
#         return encoded_coords.view(B, N, self.d_model)

class Fixed3DPositionalEncoding(nn.Module):
    def __init__(self, feature_size, max_freq=10):
        """
        Fixed 3D Positional Encoding using sinusoidal functions.
        :param feature_size: Number of output features per position (must be divisible by 6).
        :param max_freq: Maximum frequency for sine and cosine functions.
        """
        super().__init__()
        assert feature_size % 6 == 0, "Feature size must be divisible by 6 (to split across x, y, z)."
        self.feature_size = feature_size
        self.max_freq = max_freq  # Determines the range of frequencies used.

    def forward(self, xyz):
        """
        :param xyz: Tensor of shape (B, N, 3) representing 3D coordinates.
        :return: Positional encoding of shape (B, N, feature_size).
        """
        B, N, _ = xyz.shape  # B = batch, N = number of points, 3 = (x, y, z)

        # Generate frequency bands
        num_bands = self.feature_size // 6  # Each coordinate gets an equal portion of features
        frequencies = torch.linspace(1.0, self.max_freq, num_bands, device=xyz.device)  # Shape: (num_bands,)
        frequencies = frequencies[None, None, :]  # Expand for broadcasting (1, 1, num_bands)

        # Expand input coordinates
        xyz = xyz.unsqueeze(-1)  # Shape: (B, N, 3, 1)

        # Compute sinusoidal embeddings
        sin_components = torch.sin(xyz * frequencies)  # Shape: (B, N, 3, num_bands)
        cos_components = torch.cos(xyz * frequencies)

        # Concatenate along the last axis and reshape
        positional_encoding = torch.cat([sin_components, cos_components], dim=-1)  # Shape: (B, N, 3, 2 * num_bands)
        positional_encoding = positional_encoding.view(B, N, self.feature_size)  # Flatten last two dimensions

        return positional_encoding


class Permute(nn.Module):
    def __init__(self, *permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(*self.permutation)


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

        self._write_index = 0

        self.npoints = cfg.num_point
        output_channels = 3  # Normal vector
        d_points = cfg.input_dim
        self.group_size = cfg.model.group_size
        internal_channels = 1024

        self.ie_input_size = 155
        self.ie_output_size = 128 - 3

        self.feature_size = self.ie_output_size + 3

        self.input_embedding = nn.Sequential(
            nn.Linear(self.ie_input_size, self.ie_input_size),
            Permute(1, 2, 0),
            nn.BatchNorm1d(self.ie_input_size),
            Permute(2, 0, 1),
            # nn.LayerNorm(self.ie_input_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.ie_input_size, self.ie_output_size),
            Permute(1, 2, 0),
            nn.BatchNorm1d(self.ie_output_size),
            Permute(2, 0, 1),
            # nn.LayerNorm(self.ie_output_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # self.ie_transformer = MultiStackedAttention(6, heads=1, layers=4, div=1)

        self.grp_atn_1 = MultiStackedAttention(self.feature_size, heads=4, layers=4, div=1)

        # self.cluster_embedding = nn.Sequential(
        #     nn.Linear(self.feature_size + 3, self.feature_size + 3),
        #     # nn.BatchNorm1d(self.ie_input_size),
        #     nn.LayerNorm(self.feature_size + 3),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_size + 3, self.feature_size),
        #     # nn.BatchNorm1d(self.ie_output_size),
        #     nn.LayerNorm(self.feature_size),
        #     nn.ReLU(),
        # )

        # self.grp_atn_2 = MultiStackedAttention(self.feature_size, heads=4, layers=4, div=1)

        # self.max_global_pooling = nn.AdaptiveMaxPool1d(self.feature_size // 2)
        #
        # self.global_point_output_size = 16 - 3
        # self.global_point_feature_size = self.global_point_output_size + 3
        # self.global_input_embedding = nn.Sequential(
        #     nn.Linear(self.ie_output_size, self.ie_output_size),
        #     Permute(1, 2, 0),
        #     nn.BatchNorm1d(self.ie_output_size),
        #     Permute(2, 0, 1),
        #     # nn.LayerNorm(self.ie_output_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(self.ie_output_size, self.global_point_output_size),
        #     Permute(1, 2, 0),
        #     nn.BatchNorm1d(self.global_point_output_size),
        #     Permute(2, 0, 1),
        #     # nn.LayerNorm(self.global_point_output_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        # )
        # self.global_transformer = MultiStackedAttention(self.global_point_feature_size, heads=2, layers=4, div=1)
        # self.global_upscale = nn.Linear(self.global_point_feature_size, self.feature_size // 2)
        # self.avg_global_pooling = nn.AdaptiveAvgPool1d(self.feature_size)

        # self.conv_p3 = nn.Conv1d(2048, 256, kernel_size=1, bias=False)
        # self.conv_p4 = nn.Conv1d(4096, 1024, kernel_size=1)
        # self.conv_up = nn.Conv1d(64, self.npoints, kernel_size=1)

        # self.transition_up = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1),
        #                                    nn.BatchNorm1d(256),
        #                                    nn.LeakyReLU(0.2),
        #                                    nn.Conv1d(256, 512, kernel_size=1),
        #                                    nn.BatchNorm1d(512),
        #                                    nn.LeakyReLU(0.2),
        #                                    nn.Conv1d(512, self.npoints, kernel_size=1),
        #                                    nn.BatchNorm1d(self.npoints),
        #                                    nn.LeakyReLU(0.2),
        #                                    )

        # self.down_feature1 = nn.Sequential(nn.Conv1d(32768, 16384, kernel_size=1),
        #                                    nn.BatchNorm1d(16384),
        #                                    nn.ReLU(),
        #                                    # nn.Conv1d(32768, 16384, kernel_size=1),
        #                                    # nn.BatchNorm1d(16384),
        #                                    # nn.ReLU(),
        #                                    nn.Conv1d(16384, 8192, kernel_size=1),
        #                                    nn.BatchNorm1d(8192),
        #                                    nn.ReLU(),
        #                                    )

        self.down_feature2 = nn.Sequential(nn.Conv1d(self.feature_size * 2, 2048, kernel_size=1),
                                           nn.BatchNorm1d(2048),
                                           nn.LeakyReLU(0.2),
                                           # nn.Conv1d(8192, 4096, kernel_size=1),
                                           # nn.BatchNorm1d(4096),
                                           # nn.ReLU(),
                                           nn.Conv1d(2048, 1024, kernel_size=1),
                                           nn.BatchNorm1d(1024),
                                           nn.LeakyReLU(0.2),
                                           )

        # self.conv1 = nn.Conv1d(d_points, internal_channels, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(internal_channels, internal_channels, kernel_size=1, bias=False)

        # self.relu = nn.ReLU()

        self.albedo_metallic_occ_head = PointTransformerMat.Prediction(internal_channels, 5)
        self.occ_head = PointTransformerMat.Prediction(internal_channels, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    type_to_size = {
        "f": 4,
        "f2": 8,
        "f3": 12,
        "f4": 16,
        "i": 4
    }

    type_to_num_elements = {
        "f": 1,
        "f2": 2,
        "f3": 3,
        "f4": 4,
        "i": 1
    }

    type_to_dtype = {
        "f": "f4",
        "f2": "f4",
        "f3": "f4",
        "f4": "f4",
        "i": "i4"
    }

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

        # Convert sRGB to linear RGB
        mask = rgb > 0.04045
        rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # RGB to XYZ transformation matrix
        M = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], dtype=torch.float32, device=rgb.device)

        # Convert RGB to XYZ (batch-wise matrix multiplication)
        xyz = torch.einsum('...ij,jk->...ik', rgb_linear, M.T)

        # Normalize XYZ by reference white point (D65)
        xyz_ref_white = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32, device=rgb.device)
        xyz = xyz / xyz_ref_white

        # Nonlinear transformation for LAB
        epsilon = 0.008856
        kappa = 903.3
        mask = xyz > epsilon
        xyz_f = torch.where(mask, xyz ** (1 / 3), (kappa * xyz + 16) / 116)

        # Compute L, a, b
        L = (116 * xyz_f[..., 1]) - 16
        a = 500 * (xyz_f[..., 0] - xyz_f[..., 1])
        b = 200 * (xyz_f[..., 1] - xyz_f[..., 2])

        return torch.stack([L, a, b], dim=-1)

    def normalize_lab(self, lab: torch.Tensor) -> torch.Tensor:
        """
        Normalize LAB values to the range [-1,1].

        Args:
            lab: Tensor of shape (..., 3) with LAB values.

        Returns:
            Normalized LAB tensor with values in [-1,1].
        """
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

        L = (L - 50) / 50  # L in [-1,1]
        a = a / 128  # a in [-1,1]
        b = b / 128  # b in [-1,1]

        return torch.stack([L, a, b], dim=-1)

    def __get_header_dtype(self, meta: dict) -> np.dtype:
        fields = []
        for k, v in meta["header"].items():
            np_type = ("<" if meta["isLittleEndian"] else ">") + self.type_to_dtype[v]
            count = self.type_to_num_elements[v]
            if count == 1:
                fields.append((k, np_type))
            else:
                for i in range(count):
                    fields.append((f"{k}_{i}", np_type))

        return np.dtype(fields)

    def write_test_data(self, xyz: torch.Tensor, normals: torch.Tensor, view_dirs: torch.Tensor, out_dir):
        num_rows = xyz.shape[1]
        meta = {
            "header": {
                "Position": "f3",
                "Object Index": "i",
                "Albedo": "f4",
                "Metallic": "f4",
                "Normal": "f4",
                "Occlusion": "f3",
                **{f"{label} {i + 1}": "f3" for i in range(view_dirs.shape[-1] // 6) for label in
                   ("View Direction", "Radiance")},
                "Correct Viewpoint Ratio": "f"
            },
            "numViewpoints": view_dirs.shape[-1] // 6,
            "numPoints": num_rows,
            "numExtraData": 4,
            "isLittleEndian": True
        }

        num_models = xyz.shape[0]
        dtype = self.__get_header_dtype(meta)

        out_dir = Path(out_dir)
        for i in range(num_models):
            scene_index = self._write_index + i
            with open(out_dir / f"test_{scene_index}_cleaned.meta.json", "w") as f:
                json.dump(meta, f)

            data: torch.Tensor = torch.cat((xyz[i],
                                            torch.zeros((num_rows, 9), device=xyz.device),
                                            normals[i],
                                            torch.zeros((num_rows, 4), device=xyz.device),
                                            view_dirs[i],
                                            torch.zeros((num_rows, 1), device=xyz.device)), dim=1)

            np_data: np.ndarray = np.core.records.fromarrays(data.detach().cpu().numpy().T, dtype=dtype)

            with open(out_dir / f"test_{scene_index}_cleaned.data", "wb") as f:
                f.write(np_data.tobytes())

        self._write_index += num_models

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

    def get_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device)
        luminance = torch.sqrt(torch.sum(weights * (rgb ** 2), dim=-1, keepdim=True))
        return luminance

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
        k = 24
        _, top_k_i = torch.topk(dot_products, k, dim=-1)

        # Chooses the view directions
        top_k_i = top_k_i.unsqueeze(-1) * 6

        # Create indices for the 6 values (direction + radiance) per viewpoint
        offsets = torch.arange(6, device=x.device).view(1, 1, 6)  # (1, 1, 1, 6)
        top_k_i = (top_k_i + offsets).view(B, N, -1)

        new_x = torch.gather(new_x, -1, top_k_i)
        new_x = self.rotate_directions(new_x, normals)

        # Luminance calculations
        rgb_indices = range(1, k * 2, 2)
        rgb = new_x.reshape(B, N, -1, 3)[:, :, rgb_indices, :]
        lum = self.get_luminance(rgb).squeeze(-1)

        max_lum_i = torch.argmax(lum, dim=-1, keepdim=True)
        max_lum = torch.gather(lum, index=max_lum_i, dim=-1)
        max_lum_rgb = torch.gather(rgb, index=max_lum_i.unsqueeze(-1).expand(-1, -1, -1, 3), dim=-2).squeeze(-2)
        min_lum_i = torch.argmin(lum, dim=-1, keepdim=True)
        min_lum = torch.gather(lum, index=min_lum_i, dim=-1)
        min_lum_rgb = torch.gather(rgb, index=min_lum_i.unsqueeze(-1).expand(-1, -1, -1, 3), dim=-2).squeeze(-2)
        avg_lum = torch.mean(lum, dim=-1, keepdim=True)
        var_lum = torch.var(lum, dim=-1, keepdim=True)
        med_lum, _ = torch.median(lum, dim=-1, keepdim=True)

        new_x = torch.cat([new_x, max_lum, max_lum_rgb, min_lum, min_lum_rgb, med_lum, avg_lum, var_lum], dim=-1)

        new_x = self.input_embedding(new_x)  # .permute(0, 2, 1)

        # global_point_features = self.global_input_embedding(new_x)
        # global_point_features = torch.cat((xyz, global_point_features), dim=-1)
        # global_point_features = torch.cat(
        #     (torch.zeros(B, 1, self.global_point_feature_size, device=global_point_features.device),
        #      global_point_features),
        #     dim=1)
        # global_point_features = self.global_transformer(global_point_features).permute(2, 1, 0)
        # global_feature = self.global_upscale(global_point_features[:, 0, :].unsqueeze(1)).expand(-1, N, -1)

        # new_x = new_x + self.pe_gamma * positional_encoding
        new_x = torch.cat((xyz, new_x), dim=-1)

        # global_max_pool = self.max_global_pooling(new_x)
        # global_feature = torch.cat((global_feature, global_max_pool), dim=-1)
        #
        # new_x = torch.cat((new_x, global_feature), dim=-1)

        n_point = self.round_down_to_power_of_2(N // self.group_size) * 2
        new_xyz, new_points, indices = sample_and_group_new(n_point, self.group_size, xyz, new_x)
        # centroids = torch.mean(new_points[..., :3], dim=2)

        # distances = torch.norm(xyz.unsqueeze(2) - centroids.unsqueeze(1), dim=-1, p=2)  # (b, m, n)
        # closest_centroid_idx = torch.argmin(distances, dim=-1)  # (b, m)
        # closest_centroid_idx = closest_centroid_idx.unsqueeze(-1).expand(-1, -1, self.feature_size)

        new_points = new_points.view(B * n_point, self.group_size, -1)
        new_points = torch.cat(
            [new_points, torch.zeros((B * n_point, 1, self.feature_size), device=new_points.device)],
            dim=1)

        # new_points = new_points.permute(1, 0, 2)
        # new_points = new_points.permute(0, 2, 1)
        new_points = self.grp_atn_1(new_points)

        cluster_features = new_points[:, -1, :]
        new_points = new_points[:, :-1, :]

        point_cluster_features = reconstruct_points(new_points.reshape(B, n_point, self.group_size, -1), indices, N)

        # cluster_features = new_points[:, -1, :].reshape(B, n_point, -1)
        # point_cluster_features = torch.gather(cluster_features, dim=1, index=closest_centroid_idx)

        new_points = torch.cat([new_x, point_cluster_features], dim=-1)

        """

        clusters = torch.cat([centroids, cluster_features], dim=-1)
        clusters = self.cluster_embedding(clusters)
        n_point = self.round_down_to_power_of_2(clusters.shape[1] // self.group_size) * 2
        _, new_clusters, indices = sample_and_group_new(n_point, self.group_size, centroids, clusters)

        centroids = torch.mean(new_clusters[..., :3], dim=2)

        distances = torch.norm(xyz.unsqueeze(2) - centroids.unsqueeze(1), dim=-1, p=2)  # (b, m, n)
        closest_centroid_idx = torch.argmin(distances, dim=-1)  # (b, m)
        closest_centroid_idx = closest_centroid_idx.unsqueeze(-1).expand(-1, -1, self.feature_size)

        new_clusters = new_clusters.view(B * n_point, self.group_size, -1)
        new_clusters = torch.cat(
            [new_clusters, torch.zeros((B * n_point, 1, self.feature_size), device=new_points.device)],
            dim=1)

        new_clusters = self.grp_atn_2(new_clusters)

        cluster_features = new_clusters[:, -1, :].reshape(B, n_point, -1)
        point_cluster_features = torch.gather(cluster_features, dim=1, index=closest_centroid_idx)

        new_points = torch.cat([new_points, point_cluster_features], dim=-1)

        # """
        new_points = new_points.permute(0, 2, 1)

        new_points = self.down_feature2(new_points)

        albedo_metallic_occ = self.sig(self.albedo_metallic_occ_head(new_points)).permute(0, 2, 1)
        albedo = albedo_metallic_occ[..., :3]
        metallic = albedo_metallic_occ[..., 3:5]
        # occlusion = albedo_metallic_occ[..., 5].unsqueeze(-1)
        occlusion = self.sig(self.occ_head(new_points)).permute(0, 2, 1)

        return albedo, metallic, occlusion
