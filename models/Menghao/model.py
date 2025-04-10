import json
from pathlib import Path

import fpsample
import numpy as np
import skimage.color
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

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


def knn(xyz, points, centroids, nsample):
    dists = square_distance(centroids, xyz)
    idx = dists.argsort()[:, :, :nsample]

    grouped_points = index_points(points, idx)

    return grouped_points


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


def reconstruct_points(grouped_points, idx, original_num_points, return_transformed_points=False):
    """
    Reconstruct the original points tensor using grouped points and indices.

    Args:
        grouped_points (torch.Tensor): Grouped points of shape (B, npoint, nsample, C).
        idx (torch.Tensor): Indices of grouped points of shape (B, npoint, nsample).
        original_num_points (int): Original number of points (N) in the input.

    Returns:
        torch.Tensor: Reconstructed points tensor of shape (B, N, C).
        torch.Tensor: Boolean tensor of shape (B, N, C) where true means the point was transformed
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

    if return_transformed_points:
        transformed_points = counts > 0

    # Avoid division by zero (set counts to 1 where zero)
    counts = counts.clamp(min=1)

    # Average the accumulated points
    reconstructed = reconstructed / counts

    if return_transformed_points:
        return reconstructed, transformed_points

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
    def __init__(self, channels=256, heads=4, layers=4, dropout=0.2):
        super().__init__()
        self.channels = channels
        self.layers = layers

        # Using ModuleList to hold layers for attention and feed-forward networks
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        # Using ModuleList for LayerNorms (applied *before* attention/FFN in Pre-Norm)
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()

        self.dropout = nn.Dropout(dropout)

        for _ in range(layers):
            # LayerNorm applied *before* the attention mechanism
            self.norm1_layers.append(nn.LayerNorm(channels))
            # Multi-Head Attention layer
            self.attn_layers.append(nn.MultiheadAttention(embed_dim=channels,
                                                          num_heads=heads,
                                                          dropout=dropout,  # Dropout within MHA
                                                          batch_first=True))  # Expects (Batch, Seq, Feat)

            # LayerNorm applied *before* the Feed-Forward Network
            self.norm2_layers.append(nn.LayerNorm(channels))
            # Feed-Forward Network (Simple Linear -> Dropout as in original)
            # Consider replacing with a more standard FFN (Linear->ReLU/GELU->Dropout->Linear->Dropout)
            # if performance is insufficient.
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(channels, channels),
                nn.Dropout(dropout)
            ))

        # Final LayerNorm after all layers (optional, but common)
        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, SequenceLength, Channels)
        Returns:
            torch.Tensor: Output tensor of shape (Batch, SequenceLength, Channels)
        """
        # Input x is expected to be (Batch, SequenceLength, Channels)
        # because batch_first=True in MHA and LayerNorm operates on the last dim.

        for i in range(self.layers):
            # --- Pre-Normalization 1 ---
            x_norm1 = self.norm1_layers[i](x)

            # --- Multi-Head Self-Attention ---
            # MHA expects query, key, value. For self-attention, they are the same.
            # Returns attn_output, attn_weights (we discard weights)
            attn_output, _ = self.attn_layers[i](x_norm1, x_norm1, x_norm1)

            # --- Residual Connection 1 ---
            # Dropout is often applied *before* the residual connection in Transformers
            x = x + self.dropout(attn_output)

            # --- Pre-Normalization 2 ---
            x_norm2 = self.norm2_layers[i](x)

            # --- Feed-Forward Network ---
            ffn_output = self.ffn_layers[i](x_norm2)

            # --- Residual Connection 2 ---
            x = x + ffn_output  # Dropout is already included in the ffn_layer

        # Apply final normalization
        x = self.final_norm(x)

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


def resize_pc(xyz, min, max):
    min = min.unsqueeze(1).expand(-1, xyz.shape[1], -1)
    max = max.unsqueeze(1).expand(-1, xyz.shape[1], -1)
    return ((xyz - min) / (max - min)) * 2 - 1


def rgb_to_lab(rgb: torch.Tensor):
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


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """
    Convert a LAB tensor to RGB color space (0-1 range).

    Args:
        lab: Tensor of shape (batch, n, 3) with LAB values.

    Returns:
        rgb: Tensor of shape (batch, n, 3) in RGB [0,1] range.
    """
    assert lab.shape[-1] == 3, "Input must have shape (..., 3)"
    assert not torch.isnan(lab).any(), f"Input contains NaN values ({torch.isnan(lab).sum().item()})"

    device = lab.device

    # Extract components
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Compute intermediate values
    f_y = (L + 16) / 116
    f_x = a / 500 + f_y
    f_z = f_y - b / 200

    # Nonlinear transformation parameters
    epsilon = 0.008856
    kappa = 903.3
    epsilon_cbrt = epsilon ** (1 / 3)  # ~0.20689655

    # Convert to normalized XYZ
    xyz_normalized = torch.stack([f_x, f_y, f_z], dim=-1)
    mask = xyz_normalized > epsilon_cbrt
    xyz_normalized = torch.where(mask, xyz_normalized ** 3, (116 * xyz_normalized - 16) / kappa)

    # Denormalize using reference white (D65)
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=device)
    xyz = xyz_normalized * xyz_ref_white

    # XYZ to RGB transformation matrix (inverse of RGB-to-XYZ matrix)
    M_inv = torch.tensor([
        [3.24096994, -1.53738318, -0.49861076],
        [-0.96924364, 1.8759675, 0.04155506],
        [0.05563008, -0.20397696, 1.05697151]
    ], dtype=torch.float32, device=device)

    # Convert to linear RGB
    rgb_linear = torch.einsum('...ij,jk->...ik', xyz, M_inv.T)

    # Apply inverse gamma correction
    mask_gamma = rgb_linear > 0.0031308
    rgb = torch.where(
        mask_gamma,
        (rgb_linear ** (1 / 2.4)) * 1.055 - 0.055,
        rgb_linear * 12.92
    )

    # Clamp to valid [0, 1] range
    rgb = torch.clamp(rgb, 0.0, 1.0)

    # Final validation
    assert torch.all(rgb >= 0) and torch.all(rgb <= 1), \
        f"Invalid RGB values detected (min: {rgb.min().item()}, max: {rgb.max().item()})"

    return rgb


def normalize_lab(lab: torch.Tensor) -> torch.Tensor:
    l, a, b = lab[..., 0].unsqueeze(-1), lab[..., 1].unsqueeze(-1), lab[..., 2].unsqueeze(-1)

    l = l / 100
    a = (a + 128) / 256
    b = (b + 128) / 256

    return torch.cat((l, a, b), dim=-1)


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


# Adapted from https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    # Ensure input has at least one batch dimension
    orig_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)  # Flatten all dimensions except the last (RGB channels)

    cmax, cmax_idx = torch.max(rgb, dim=-1, keepdim=True)  # Max across RGB channels
    cmin = torch.min(rgb, dim=-1, keepdim=True)[0]  # Min across RGB channels
    delta = cmax - cmin

    # Initialize HSL components
    hsl_h = torch.empty_like(cmax)
    cmax_idx[delta == 0] = 3  # Set undefined hues to 0

    # Hue calculation
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]  # Red is max
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]  # Green is max
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]  # Blue is max
    hsl_h[cmax_idx == 3] = 0.  # Zero hue when no chroma
    hsl_h /= 6.

    # Lightness calculation
    hsl_l = (cmax + cmin) / 2.

    # Saturation calculation
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0

    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)

    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (-hsl_l * 2. + 2.))[hsl_l_l0_5]

    # Reshape back to the original shape
    hsl = torch.cat([hsl_h, hsl_s, hsl_l], dim=-1).reshape(orig_shape)
    return hsl


# Adapted from https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    # Split channels, preserving all leading dimensions
    hsl_h, hsl_s, hsl_l = hsl[..., 0:1], hsl[..., 1:2], hsl[..., 2:3]

    # Compute intermediate values
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.

    # Compute region index
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(*idx.shape[:-1], 3)  # Expand to 3 channels

    # Initialize output tensor
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)

    # Assign RGB values based on hue region
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=-1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=-1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=-1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=-1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=-1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=-1)[idx == 5]

    # Add lightness adjustment
    rgb += _m

    return torch.clamp(rgb, 0, 1)


class IntrinsicsEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dirs, rgb, normals, hdr):
        B, N = rgb.shape[0], rgb.shape[1]
        lab = rgb_to_lab(rgb)
        device = rgb.device
        lum = lab[..., 0] / 100
        total_mean_lum = torch.mean(torch.max(lum, dim=-1)[0].reshape(B, -1), dim=-1, keepdim=True).unsqueeze(
            dim=-1).expand(-1, N, -1)
        lum_max = torch.max(lum, dim=-1, keepdim=True)[0]
        max_lum = torch.max(lum_max, dim=1, keepdim=True)[0].expand(-1, N, -1)
        # occ = lum_max / max_lum
        occ = torch.where(lum_max > total_mean_lum, torch.ones_like(lum_max, device=device),
                          torch.zeros_like(lum_max, device=device))

        # Albedo
        hsl = rgb2hsl_torch(rgb)
        norm_hsl = torch.linalg.norm(hsl[..., 1:3], dim=-1)
        max_norm = torch.argmax(norm_hsl, dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, 3)
        avg_hsl = torch.gather(hsl, 2, max_norm).squeeze(2)

        h, s, l = avg_hsl[..., 0], avg_hsl[..., 1], avg_hsl[..., 2]
        hue_diff = torch.abs(h.unsqueeze(2) - h.unsqueeze(1))

        similar_hue_mask = hue_diff <= 0.02
        selection = torch.linalg.norm(avg_hsl[..., 1:3], dim=-1, keepdim=True)

        masked_selection = similar_hue_mask * selection

        max_indices = torch.argmax(masked_selection, dim=-1)

        batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)
        selected_colors = avg_hsl[batch_indices, max_indices]

        albedo = hsl2rgb_torch(selected_colors)

        # Occlusion
        # masked_l = torch.where(similar_hue_mask, l, torch.nan)
        # avg_lum = torch.nanmean(masked_l, dim=-1, keepdim=True)
        max_lum = torch.max(hsl[..., 2], dim=2, keepdim=True)[0]
        avg_lum = torch.mean(l, dim=-1, keepdim=True).unsqueeze(-1).expand(-1, N, -1)

        occ = torch.where(max_lum < avg_lum, torch.tensor(0), torch.tensor(1))
        # occ = max_lum

        # Metallic
        b, n = torch.nonzero(occ.squeeze(-1))[0]
        hdr_lum = torch.sqrt(0.299 * hdr[..., 0] ** 2 + 0.587 * hdr[..., 1] ** 2 + 0.114 * hdr[..., 2] ** 2)
        lum_values = hdr_lum[b, n, :]
        lum_dirs = dirs[b, n, ...]
        scaled_lum_dirs = lum_dirs * lum_values.unsqueeze(-1).expand(-1, 3)

        min_lum = torch.mean(lum_values).cpu()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        lum_dirs = lum_dirs.cpu() * min_lum
        scaled_lum_dirs = scaled_lum_dirs.cpu()

        ax.scatter(scaled_lum_dirs[..., 0], scaled_lum_dirs[..., 1], scaled_lum_dirs[..., 2])
        ax.scatter(lum_dirs[..., 0], lum_dirs[..., 1], lum_dirs[..., 2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

        smoothness = torch.max(lum, dim=-1, keepdim=True)[0] - torch.min(lum, dim=-1, keepdim=True)[0]

        max_lum_index = torch.argmax(lum, dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, 3)
        max_lum_hsl = torch.gather(hsl, 2, max_lum_index).squeeze(2)
        diff = max_lum_hsl[..., 0:2] - selected_colors[..., 0:2]
        metallic = torch.linalg.norm(diff, dim=-1, keepdim=True)

        met_smth = torch.cat((metallic, smoothness), dim=-1)

        return albedo, met_smth, occ


class Permute(nn.Module):
    def __init__(self, *permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(*self.permutation)


def round_up_to_power_of_2(n):
    if n <= 1:
        return 1  # Edge case: if n is 0, return 0
    n -= 1
    n |= (n >> 1)
    n |= (n >> 2)
    n |= (n >> 4)
    n |= (n >> 8)
    n |= (n >> 16)
    return n + 1


class GlobalTransformer(nn.Module):
    def __init__(self, channels, group_size, heads=4, layers=4):
        super().__init__()
        self.channels = channels
        self.group_size = group_size
        self.local_attention = MultiStackedAttention(channels, heads=heads, layers=layers)
        self.global_attention = MultiStackedAttention(channels, heads=heads, layers=layers)

    def forward(self, xyz, point_data):
        B, N, _ = point_data.shape
        n_point = round_up_to_power_of_2(N // self.group_size)
        new_xyz, new_points, indices = sample_and_group_new(n_point, self.group_size, xyz, point_data)
        centroids = torch.mean(new_points[..., :3], dim=2)

        distances = torch.norm(xyz.unsqueeze(2) - centroids.unsqueeze(1), dim=-1, p=2)  # (b, m, n)
        closest_centroid_idx = torch.argmin(distances, dim=-1)  # (b, m)
        closest_centroid_idx = closest_centroid_idx.unsqueeze(-1).expand(-1, -1, self.channels)

        new_points = new_points.view(B * n_point, self.group_size, -1)
        new_points = torch.cat(
            [new_points, torch.zeros((B * n_point, 1, self.channels), device=new_points.device)],
            dim=1)

        new_points = self.local_attention(new_points)

        cluster_features = new_points[:, -1, :].reshape(B, n_point, -1)
        new_points = new_points[:, :-1, :]

        point_features, transformed_points = reconstruct_points(new_points.reshape(B, n_point, self.group_size, -1),
                                                                indices, N, return_transformed_points=True)
        new_points = point_data + point_features  # torch.where(transformed_points, point_features, point_data)

        # Transform cluster features
        cluster_features = self.global_attention(cluster_features)

        point_cluster_features = torch.gather(cluster_features, dim=1, index=closest_centroid_idx)
        return torch.cat((new_points, point_cluster_features), dim=-1)


class PointTransformerMat(nn.Module):
    class Prediction(nn.Module):
        def __init__(self, channels, out_channels, dp=0.5):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(channels, channels // 2),
                # Permute(1, 2, 0),
                # nn.BatchNorm1d(channels // 2),
                nn.LayerNorm(channels // 2),
                nn.ReLU(),
                nn.Dropout(dp),
                # Permute(2, 0, 1),
                nn.Linear(channels // 2, channels // 4),
                # nn.BatchNorm1d(channels // 4),
                nn.ReLU(),
                nn.Linear(channels // 4, out_channels),
            )

        def forward(self, x):
            return self.layers(x)

    def __init__(self, cfg):
        super().__init__()

        self._write_index = 0

        self.npoints = cfg.num_point
        self.group_size = cfg.model.group_size
        internal_channels = 1024
        self.k = 30

        self.ie_input_size = self.k * 6
        self.ie_output_size = 256 - 3

        self.feature_size = self.ie_output_size + 3

        self.input_embedding = nn.Sequential(
            nn.Linear(self.ie_input_size, self.ie_input_size),
            # Permute(1, 2, 0),
            # nn.BatchNorm1d(self.ie_input_size),
            # Permute(2, 0, 1),
            nn.LayerNorm(self.ie_input_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.ie_input_size, self.ie_output_size),
            # Permute(1, 2, 0),
            # nn.BatchNorm1d(self.ie_output_size),
            # Permute(2, 0, 1),
            nn.LayerNorm(self.ie_output_size),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )

        self.skip = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            # Permute(1, 2, 0),
            # nn.BatchNorm1d(self.feature_size),
            # Permute(2, 0, 1),
            nn.LayerNorm(self.feature_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_size, internal_channels),
            # Permute(1, 2, 0),
            # nn.BatchNorm1d(internal_channels),
            # Permute(2, 0, 1),
            nn.LayerNorm(internal_channels),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )

        self.hdr_ie = nn.Sequential(
            nn.Linear(self.ie_input_size, self.ie_input_size),
            # Permute(1, 2, 0),
            # nn.BatchNorm1d(self.ie_input_size),
            # Permute(2, 0, 1),
            nn.LayerNorm(self.ie_input_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.ie_input_size, self.feature_size),
            # Permute(1, 2, 0),
            # nn.BatchNorm1d(self.ie_output_size),
            # Permute(2, 0, 1),
            nn.LayerNorm(self.feature_size),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )
        self.hdr_transformer = GlobalTransformer(self.feature_size, self.group_size, heads=4, layers=4)
        self.hdr_ff = nn.Linear(self.feature_size * 2, 3 * self.k)

        self.intrinsics_estimator = IntrinsicsEstimator()

        self.transformer = GlobalTransformer(self.feature_size, self.group_size, heads=4, layers=4)

        self.met_ie = nn.Sequential(
            nn.Linear(6, self.feature_size),
            # Permute(1, 2, 0),
            # nn.BatchNorm1d(self.ie_input_size),
            # Permute(2, 0, 1),
            nn.LayerNorm(self.feature_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_size, self.feature_size),
            # Permute(1, 2, 0),
            # nn.BatchNorm1d(self.ie_output_size),
            # Permute(2, 0, 1),
            nn.LayerNorm(self.feature_size),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )
        self.met_transformer = MultiStackedAttention(6, heads=1, layers=4)

        # self.grp_atn_1 = MultiStackedAttention(self.feature_size, heads=4, layers=4)

        self.down_feature2 = nn.Sequential(nn.Conv1d(self.feature_size * 2, 2048, kernel_size=1),
                                           Permute(0, 2, 1),
                                           nn.LayerNorm(2048),
                                           Permute(0, 2, 1),

                                           # nn.BatchNorm1d(2048),
                                           nn.LeakyReLU(0.2),
                                           # nn.Conv1d(8192, 4096, kernel_size=1),
                                           # nn.BatchNorm1d(4096),
                                           # nn.ReLU(),
                                           nn.Conv1d(2048, 1024, kernel_size=1),
                                           # nn.BatchNorm1d(1024),
                                           Permute(0, 2, 1),
                                           nn.LayerNorm(1024),
                                           Permute(0, 2, 1),
                                           nn.LeakyReLU(0.2),
                                           )

        self.albedo_metallic_occ_head = PointTransformerMat.Prediction(internal_channels, 5)
        # self.metallic_head = PointTransformerMat.Prediction(internal_channels, 2)
        # self.occ_head = PointTransformerMat.Prediction(internal_channels, 1)
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

    def test_rgb_to_lab(self):
        # Define the number of steps for each channel
        steps = 255

        # Create a linspace for each channel (R, G, B) ranging from 0 to 1
        r = torch.linspace(0, 1, steps)
        g = torch.linspace(0, 1, steps)
        b = torch.linspace(0, 1, steps)

        # Create a grid of all possible combinations of R, G, B
        grid_r, grid_g, grid_b = torch.meshgrid(r, g, b)

        # Stack the grids along the last dimension to get the final tensor
        rgb = torch.stack((grid_r, grid_g, grid_b), dim=-1)

        test_lab = rgb_to_lab(rgb)
        lab = skimage.color.rgb2lab(rgb)

        loss = F.mse_loss(test_lab, torch.tensor(lab))

        assert loss.item() < 0.001

        lab_rgb = lab_to_rgb(test_lab)

        loss = F.mse_loss(lab_rgb * 255, rgb * 255)

        assert loss.item() < 0.001

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
        rotated_x = torch.empty_like(new_x)  # Same shape (B, N, k*9)

        # Fill in rotated view directions
        rotated_x[..., view_dir_indices] = rotated_dirs.reshape(B, N, -1)
        # Keep radiances unchanged
        rotated_x[..., rad_indices] = radiances.reshape(B, N, -1)

        return rotated_x

    def get_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device)
        luminance = torch.sqrt(torch.sum(weights * (rgb ** 2), dim=-1, keepdim=True))
        return luminance

    def forward(self, x, normals, light_min_bounds=None, light_max_bounds=None):
        torch.set_printoptions(sci_mode=False, precision=10)
        B, N, C = x.shape

        device = x.device

        xyz = x[..., :3]
        if light_min_bounds is not None and light_max_bounds is not None:
            xyz = resize_pc(xyz, light_min_bounds, light_max_bounds)
        else:
            xyz = normalize_pc(xyz)
        new_x = x[..., 3:]

        # Gets the dot products of the view directions to the normal
        dir_indices = [i for i in range(new_x.shape[-1]) if ((i // 3) % 2) == 0]
        dirs = new_x[..., dir_indices].view(B, N, -1, 3)
        norm = torch.norm(dirs, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm))
        dot_products = torch.einsum('bndi,bni->bnd', dirs, normals)
        # dot_products = torch.sum(dirs * normals.unsqueeze(2), dim=-1)

        # Gathers k-closest directions to the normal
        _, top_k_i = torch.topk(dot_products, self.k, dim=-1)
        radiance_indices = top_k_i

        # Chooses the view directions
        top_k_i = top_k_i.unsqueeze(-1) * 6

        # Create indices for the 6 values (direction + radiance) per viewpoint
        offsets = torch.arange(6, device=x.device).view(1, 1, 6)  # (1, 1, 1, 6)
        top_k_i = (top_k_i + offsets).view(B, N, -1)

        new_x = torch.gather(new_x, -1, top_k_i)
        new_x = self.rotate_directions(new_x, normals)
        # new_x = torch.cat((xyz, normals, new_x), dim=-1)

        rgb_indices = [i for i in range(0, new_x.shape[-1]) if ((i // 3) % 2) == 1]
        # hdr_radiances = torch.zeros(B, N, self.k * 3, device=device)

        """ HDR Prediction """
        new_hdr = self.hdr_ie(new_x)
        hdr_radiances = self.sig(self.hdr_ff(self.hdr_transformer(xyz, new_hdr)))
        updated_new_x = new_x.clone()
        updated_new_x[..., rgb_indices] = hdr_radiances

        # updated_new_x = updated_new_x.reshape(B * N, -1, 6)
        # # new_x = self.met_ie(updated_new_x).reshape(B * N, -1, self.feature_size)
        # # new_x = torch.cat((torch.zeros(B * N, 1, new_x.shape[-1], device=device), new_x), dim=1)
        # new_x = self.met_transformer(updated_new_x)
        # new_points = new_x.reshape(B, N, -1)

        new_x = self.input_embedding(updated_new_x)  # .permute(0, 2, 1)

        new_x = torch.cat((xyz, new_x), dim=-1)
        skip = self.skip(new_x)

        new_points = self.transformer(xyz, new_x)

        new_points = new_points.permute(0, 2, 1)

        new_points = self.down_feature2(new_points)
        new_points = new_points + skip.permute(0, 2, 1)

        new_points = new_points.permute(0, 2, 1)

        albedo_metallic = self.sig(self.albedo_metallic_occ_head(new_points))
        albedo = albedo_metallic[..., :3]
        metallic = albedo_metallic[..., 3:5]
        # occlusion = albedo_metallic_occ[..., 5].unsqueeze(-1)
        # occlusion = self.sig(self.occ_head(new_points))
        occlusion = torch.zeros(B, N, 1, device=device)

        # occlusion = self.intrinsics_estimator(rgb, lab)

        return albedo, metallic, occlusion, hdr_radiances, radiance_indices
