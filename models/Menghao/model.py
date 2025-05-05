import fpsample
import numpy as np
import torch
import torch.nn as nn

from pointnet_util import index_points, square_distance


def sample_and_group(npoint, nsample, xyz, points):
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


def normalize_pc(xyz):
    centroid = xyz.mean(dim=1, keepdim=True)  # Compute mean per batch
    pc_centered = xyz - centroid  # Subtract mean to center at origin
    max_dist = torch.norm(pc_centered, dim=2, keepdim=True).max(dim=1, keepdim=True)[0]  # Find max distance per batch
    pc_normalized = pc_centered / (max_dist + 1e-8)  # Avoid division by zero
    return pc_normalized


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
        new_xyz, new_points, indices = sample_and_group(n_point, self.group_size, xyz, point_data)
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
        cluster_features = torch.cat(
            (cluster_features,
             torch.zeros((cluster_features.shape[0], 1, cluster_features.shape[-1]), device=new_points.device)), dim=1)
        cluster_features = self.global_attention(cluster_features)
        global_token = cluster_features[:, -1, :].unsqueeze(1)

        point_cluster_features = torch.gather(cluster_features, dim=1, index=closest_centroid_idx)
        return torch.cat((new_points, point_cluster_features), dim=-1), global_token


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
        self.env_map_width = cfg.model.env_map_width
        self.env_map_height = cfg.model.env_map_height
        internal_channels = 1024
        self.k = 30

        self.ie_input_size = 195
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

        self.transformer = GlobalTransformer(self.feature_size, self.group_size, heads=4, layers=4)

        self.down_feature1 = nn.Sequential(nn.Conv1d(self.feature_size * 2, self.feature_size, kernel_size=1),
                                           Permute(0, 2, 1),
                                           nn.LayerNorm(self.feature_size),
                                           Permute(0, 2, 1),

                                           # nn.BatchNorm1d(2048),
                                           nn.ReLU(),
                                           )

        self.down_feature2 = nn.Sequential(nn.Conv1d(self.feature_size, 2048, kernel_size=1),
                                           Permute(0, 2, 1),
                                           nn.LayerNorm(2048),
                                           Permute(0, 2, 1),

                                           # nn.BatchNorm1d(2048),
                                           nn.ReLU(),
                                           # nn.Conv1d(8192, 4096, kernel_size=1),
                                           # nn.BatchNorm1d(4096),
                                           # nn.ReLU(),
                                           nn.Conv1d(2048, 1024, kernel_size=1),
                                           # nn.BatchNorm1d(1024),
                                           Permute(0, 2, 1),
                                           nn.LayerNorm(1024),
                                           Permute(0, 2, 1),
                                           nn.ReLU(),
                                           )

        self.albedo_metallic_occ_head = PointTransformerMat.Prediction(internal_channels, 5)
        self.env_map_head = PointTransformerMat.Prediction(self.feature_size,
                                                           self.env_map_width * self.env_map_height * 3)
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

    def forward(self, xyz, normals, view_dirs, radiances):
        torch.set_printoptions(sci_mode=False, precision=10)
        B, N, C = xyz.shape

        device = xyz.device

        xyz = normalize_pc(xyz)
        # new_x = x[..., 3:]

        # Gets the dot products of the view directions to the normal
        # dir_indices = [i for i in range(new_x.shape[-1]) if ((i // 3) % 2) == 0]
        # dirs = new_x[..., dir_indices].view(B, N, -1, 3)
        # norm = torch.norm(dirs, dim=-1)
        # assert torch.allclose(norm, torch.ones_like(norm))
        # dot_products = torch.einsum('bndi,bni->bnd', dirs, normals)
        # dot_products = torch.sum(dirs * normals.unsqueeze(2), dim=-1)

        # Gathers k-closest directions to the normal
        # _, top_k_i = torch.topk(dot_products, self.k, dim=-1)
        # radiance_indices = top_k_i

        # Chooses the view directions
        # top_k_i = top_k_i.unsqueeze(-1) * 6
        #
        # # Create indices for the 6 values (direction + radiance) per viewpoint
        # offsets = torch.arange(6, device=x.device).view(1, 1, 6)  # (1, 1, 1, 6)
        # top_k_i = (top_k_i + offsets).view(B, N, -1)
        #
        # new_x = torch.gather(new_x, -1, top_k_i)

        # Might want to rotate the view directions
        # new_x = self.rotate_directions(new_x, normals)

        # new_x = torch.cat((xyz, normals, new_x), dim=-1)

        # rgb_indices = [i for i in range(0, new_x.shape[-1]) if ((i // 3) % 2) == 1]
        # hdr_radiances = torch.zeros(B, N, self.k * 3, device=device)

        """ HDR Prediction """
        # new_hdr = self.hdr_ie(new_x)
        # hdr_radiances = self.sig(self.hdr_ff(self.hdr_transformer(xyz, new_hdr)))
        # updated_new_x = new_x.clone()
        # updated_new_x[..., rgb_indices] = hdr_radiances

        # updated_new_x = updated_new_x.reshape(B * N, -1, 6)
        # # new_x = self.met_ie(updated_new_x).reshape(B * N, -1, self.feature_size)
        # # new_x = torch.cat((torch.zeros(B * N, 1, new_x.shape[-1], device=device), new_x), dim=1)
        # new_x = self.met_transformer(updated_new_x)
        # new_points = new_x.reshape(B, N, -1)

        view_dirs = view_dirs.reshape(B, N, -1)
        radiances = radiances.reshape(B, N, -1)
        new_x = torch.zeros(B, N, view_dirs.shape[-1] * 2, device=device)
        # Combining view directions and radiances
        view_dir_indices = [i for i in range(new_x.shape[-1]) if (i // 3) % 2 == 0]
        rad_indices = [i + 1 for i in view_dir_indices]
        new_x[..., view_dir_indices] = view_dirs
        new_x[..., rad_indices] = radiances
        new_x = torch.cat((new_x, normals), dim=-1)

        new_x = self.input_embedding(new_x)  # .permute(0, 2, 1)

        new_x = torch.cat((xyz, new_x), dim=-1)
        residual = new_x

        new_points, global_token = self.transformer(xyz, new_x)

        new_points = new_points.permute(0, 2, 1)

        new_points = self.down_feature1(new_points) + residual.permute(0, 2, 1)
        new_points = self.down_feature2(new_points)

        new_points = new_points.permute(0, 2, 1)

        albedo_metallic = self.sig(self.albedo_metallic_occ_head(new_points))
        albedo = albedo_metallic[..., :3]
        metallic = albedo_metallic[..., 3:4]
        smoothness = albedo_metallic[..., 4:5]

        # env_map = self.sig(self.env_map_head(global_token).reshape(B, self.env_map_width, self.env_map_height, 3))
        # occlusion = albedo_metallic_occ[..., 5].unsqueeze(-1)
        # occlusion = self.sig(self.occ_head(new_points))

        # occlusion = self.intrinsics_estimator(rgb, lab)

        return albedo, metallic, smoothness  # , env_map
