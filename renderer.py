import math
from typing import Tuple, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F

# --- Constants ---
PI = torch.tensor(math.pi)
EPS = 1e-6  # Small epsilon for numerical stability


# --- Helper Functions ---

def safe_acos(x):
    """Clamped acos for numerical stability."""
    return torch.acos(x.clamp(-1.0 + EPS, 1.0 - EPS))


def equirect_uv_to_dir(uv: torch.Tensor) -> torch.Tensor:
    """Converts UV coordinates (0-1) to normalized direction vectors."""
    # uv: (..., 2) tensor
    u, v = uv.split(1, dim=-1)

    phi = (u * 2 * PI) + PI / 2
    theta = v * PI

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.cos(theta)
    z = -torch.sin(theta) * torch.sin(phi)

    return torch.cat([x, y, z], dim=-1)  # (..., 3)


def get_envmap_sample_dirs_and_solid_angles(W, H, device):
    """Generates sample directions and solid angles for envmap pixels."""
    # Create UV grid for pixel centers
    v_coords = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
    u_coords = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
    uv_grid = torch.stack(torch.meshgrid(u_coords, v_coords, indexing='xy'), dim=-1)  # (W, H, 2)

    # Convert UVs to directions
    dirs = equirect_uv_to_dir(uv_grid)  # (W, H, 3)

    # Calculate solid angles (approximate for pixel centers)
    # d_omega = sin(theta) * d_theta * d_phi
    # d_theta = pi / H, d_phi = 2*pi / W
    # theta = v * pi
    theta = uv_grid[..., 1:2] * PI  # Keep last dim (W, H, 1)
    d_theta = PI / H
    d_phi = (2 * PI) / W
    solid_angles = torch.sin(theta) * d_theta * d_phi  # (W, H, 1)

    # Reshape for easier processing
    num_samples = W * H
    dirs = dirs.reshape(num_samples, 3)  # (M, 3) where M = W*H
    solid_angles = solid_angles.reshape(num_samples, 1)  # (M, 1)
    uv_grid = uv_grid.reshape(num_samples, 2)  # (M, 2)

    return dirs, solid_angles, uv_grid  # (M, 3), (M, 1), (M, 2)


def F_schlick(f0, VdotH):
    """Schlick Fresnel approximation."""
    # f0: (..., 3), base reflectivity at normal incidence
    # VdotH: (...)
    return f0 + (1.0 - f0) * torch.pow(1.0 - VdotH.clamp(0.0, 1.0), 5.0)


def D_ggx(NdotH, roughness):
    """GGX/Trowbridge-Reitz Normal Distribution Function (NDF)."""
    # NdotH: (...)
    # roughness: (...) in [0, 1]
    # TODO: Change alpha to roughness
    alpha = roughness * roughness  # Perceptual roughness to square roughness
    alpha2 = alpha * alpha
    denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0)
    return alpha2 / (PI * denom * denom + EPS)


def G_smith_ggx(NdotV, NdotL, roughness):
    """Smith Geometry term (Schlick-GGX approximation)."""
    # NdotV, NdotL: (...)
    # roughness: (...)
    alpha = roughness * roughness
    alpha2 = alpha * alpha

    def _g1(NdotW, k):
        return NdotW / (NdotW * (1.0 - k) + k + EPS)

    # Direct lighting approximation (k = alpha2 / 2) - adjust if needed for IBL
    # More common for IBL: k = alpha^2 / 2 -> alpha_g = (roughness + 1)^2 / 8
    # Alternative for IBL k = alpha / 2 -> alpha_g = roughness^2 / 2
    # Let's use the simpler direct lighting form for now, might need refinement
    k = alpha / 2.0

    return _g1(NdotV, k) * _g1(NdotL, k)


def cook_torrance_specular(NdotL, NdotV, F, D, G):
    """Cook-Torrance Specular BRDF calculation."""
    # All inputs: (...) or broadcastable
    numerator = F * D * G
    # Note: Original Cook-Torrance includes 4 * NdotL * NdotV in denominator.
    # This factor is often moved to the lighting integral or cancelled out
    # depending on the exact formulation. We follow common practice where
    # the BRDF itself doesn't include it, but the final integral light needs it.
    # denominator = (4.0 * NdotV * NdotL) + EPS
    # spec = numerator / denominator
    # Let's return numerator; the 4*NdotV*NdotL will be handled in the main loop implicitly
    # because the standard rendering equation integral includes NdotL, and we divide by pi later
    # Actually, standard definition needs the denominator here.
    denominator = (4.0 * NdotV * NdotL) + EPS
    spec = numerator / denominator
    return spec  # (..., 3)


def lambertian_diffuse(albedo):
    """Lambertian Diffuse BRDF (constant)."""
    return albedo / PI  # (..., 3)


# https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
def tonemap_aces_filmic(hdr_color, exposure: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Applies the ACES Filmic tonemapping curve (approx.).

    Args:
        hdr_color (torch.Tensor): HDR color tensor (..., 3). Values can be > 1.0.
        exposure (float): Multiplier for the HDR color before tonemapping.
                          Adjust this to control overall brightness.

    Returns:
        torch.Tensor: LDR color tensor (..., 3) in approx. [0, 1] range.
    """
    # Constants for the ACES approximation
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14

    # Apply exposure
    if exposure is float:
        x = hdr_color * exposure
    else:
        exposure = exposure.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, *hdr_color.shape[1:])
        x = hdr_color * exposure

    # Apply ACES curve component-wise
    # Formula: (x * (A * x + B)) / (x * (C * x + D) + E)
    numerator = x * (A * x + B)
    denominator = x * (C * x + D) + E
    ldr_color = numerator / denominator

    # Clamp potentially slightly out-of-range values
    return ldr_color.clamp(0.0, 1.0)


def tonemap_aces_filmic_better(hdr_color, exposure: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Applies the ACES Filmic tonemapping curve (approx.).

    Args:
        hdr_color (torch.Tensor): HDR color tensor (..., 3). Values can be > 1.0.
        exposure (float): Multiplier for the HDR color before tonemapping.
                          Adjust this to control overall brightness.

    Returns:
        torch.Tensor: LDR color tensor (..., 3) in approx. [0, 1] range.
    """
    def rrt_and_odt_fit(c):
        num = c * (c + 0.0245786) - 0.000090537
        denom = c * (0.983729 * c + 0.4329510) + 0.238081
        return num / denom

    # Apply exposure
    if exposure is float:
        x = hdr_color * exposure
    else:
        exposure = exposure.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, *hdr_color.shape[1:])
        x = hdr_color * exposure

    device = hdr_color.device

    aces_input_mat = torch.tensor(
        [[0.59719, 0.35458, 0.04823],
         [0.07600, 0.90834, 0.01566],
         [0.02840, 0.13383, 0.83777]], device=device).view(1, 1, 1, 3, 3).expand(*hdr_color.shape[:-1], -1, -1)

    aces_output_mat = torch.tensor(
        [[1.60475, -0.53108, -0.07367],
         [-0.10208,  1.10813, -0.00605],
         [-0.00327, -0.07276,  1.07602]], device=device).view(1, 1, 1, 3, 3).expand(*hdr_color.shape[:-1], -1, -1)

    ldr_color = torch.einsum("b n c i j, b n c j -> b n c i", aces_input_mat, x)

    ldr_color = rrt_and_odt_fit(ldr_color)

    ldr_color = torch.einsum("b n c i j, b n c j -> b n c i", aces_output_mat, ldr_color)

    # Clamp potentially slightly out-of-range values
    return ldr_color.clamp(0.0, 1.0)


def gamma_correction(ldr_color, gamma=2.2):
    """
    Applies gamma correction to an LDR color.

    Args:
        ldr_color (torch.Tensor): Linear LDR color tensor (..., 3) in [0, 1].
        gamma (float): The gamma value to apply (e.g., 2.2 for sRGB).

    Returns:
        torch.Tensor: Gamma-corrected LDR color tensor (..., 3) in [0, 1].
    """
    if gamma <= 0:
        raise ValueError("Gamma must be positive")
    # Ensure no negative values before exponentiation
    return ldr_color.clamp(min=EPS) ** (1.0 / gamma)


# https://learnopengl.com/PBR/Theory
class HDRRenderer(nn.Module):
    def __init__(self, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, points_normals: torch.Tensor,  # (B, N, 3) - Should be normalized
                points_albedo: torch.Tensor,  # (B, N, 3) - [0, 1]
                points_metallic: torch.Tensor,  # (B, N, 1) - [0, 1]
                points_smoothness: torch.Tensor,  # (B, N, 1) - [0, 1]
                viewing_dirs: torch.Tensor,  # (B, N, V, 3) - V is num views (64), should be normalized
                env_vis: torch.Tensor,
                env_map: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
            Renders points using PBR IBL from an environment map.

            Args:
                points_pos: Point positions (B, N, 3).
                points_normals: Point normals (B, N, 3).
                points_albedo: Point albedo (B, N, 3).
                points_metallic: Point metallic values (B, N, 1).
                points_smoothness: Point smoothness values (B, N, 1).
                viewing_dirs: Viewing directions for each point (B, N, V, 3).
                env_map: Equirectangular HDR environment map (H, W, 3).
                envmap_samples: Optional tuple of precomputed (sample_dirs, solid_angles, sample_uvs)
                                to avoid recomputing them every call.

            Returns:
                rendered_radiance: (B, N, V, 3) - HDR radiance for each point/view direction.
            """
        B, N, V, _ = viewing_dirs.shape
        _, W, H, _ = env_map.shape
        device = points_normals.device

        # --- 0. Ensure Inputs are Correct ---
        n = F.normalize(points_normals, p=2, dim=-1)  # (B, N, 3)
        v = F.normalize(viewing_dirs, p=2, dim=-1)  # (B, N, V, 3)
        roughness = 1.0 - points_smoothness  # (B, N, 1)
        albedo = points_albedo  # (B, N, 3)
        metallic = points_metallic  # (B, N, 1)

        # --- 1. Prepare Environment Map Samples ---
        # l_dirs_world: (M, 3) - Incoming light directions in world space
        # l_solid_angles: (M, 1) - Solid angle for each direction
        # l_uvs: (M, 2) - UV coordinates for sampling envmap
        l_dirs_world, l_solid_angles, l_uvs = get_envmap_sample_dirs_and_solid_angles(W, H, device)

        M = l_dirs_world.shape[0]

        # Sample environment map using grid_sample
        # grid_sample expects input in shape (B, C, Hin, Win) and grid in (B, Hout, Wout, 2) in [-1, 1]
        env_map_batch = env_map.permute(0, 3, 2, 1)  # (1, 3, H, W)
        # Convert UVs [0, 1] to grid coords [-1, 1]
        grid = l_uvs.unsqueeze(0).expand(B, -1, -1).unsqueeze(1) * 2.0 - 1.0  # (1, 1, M, 2)
        # Perform sampling
        Li = F.grid_sample(env_map_batch, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # Li shape: (1, 3, 1, M)
        Li = Li.squeeze(2).permute(0, 2, 1)  # (1, M, 3)

        # --- 2. Prepare Tensors (Expand only B, N, V dimensions initially) ---
        # These will be reused in the loop without the large M dimension yet
        n_exp_base = n.unsqueeze(2)  # (B, N, 1, 3)
        v_exp_base = v  # (B, N, V, 3)
        albedo_exp_base = albedo.unsqueeze(2)  # (B, N, 1, 3)
        metallic_exp_base = metallic.unsqueeze(2)  # (B, N, 1, 1)
        roughness_exp_base = roughness.unsqueeze(2)  # (B, N, 1, 1)

        # --- 2. Prepare Tensors for Broadcasting ---
        # Expand dimensions to match: (B, N, V, M, dim)
        # n_exp = n.unsqueeze(2).unsqueeze(3).expand(B, N, V, M, 3)  # (B, N, 1, 1, 3) -> (B, N, V, M, 3)
        # v_exp = v.unsqueeze(3).expand(B, N, V, M, 3)  # (B, N, V, 1, 3) -> (B, N, V, M, 3)
        # l_exp = l_dirs_world.view(1, 1, 1, M, 3).expand(B, N, V, M, 3)  # (1, 1, 1, M, 3) -> (B, N, V, M, 3)
        #
        # albedo_exp = albedo.unsqueeze(2).unsqueeze(3)  # (B, N, 1, 1, 3)
        # metallic_exp = metallic.unsqueeze(2).unsqueeze(3)  # (B, N, 1, 1, 1)
        # roughness_exp = roughness.unsqueeze(2).unsqueeze(3)  # (B, N, 1, 1, 1)

        # Li_exp = Li.unsqueeze(1).unsqueeze(1).expand(B, N, V, M, 3)  # (1, M, 3) -> (B, N, V, M, 3)
        # l_solid_angles_exp = l_solid_angles.view(1, 1, 1, M, 1).expand(B, N, V, M, 1)  # (M, 1) -> (B, N, V, M, 1)

        # Precompute Fresnel base
        f0_base_val = torch.full_like(albedo_exp_base, 0.04)  # (B, N, 1, 3)
        f0_base = torch.lerp(f0_base_val, albedo_exp_base, metallic_exp_base)  # (B, N, 1, 3)

        # --- Initialize Output ---
        total_rendered_radiance = torch.zeros(B, N, V, 3, device=device, dtype=torch.float32)

        for m_start in range(0, M, self.chunk_size):
            m_end = min(m_start + self.chunk_size, M)
            current_chunk_size = m_end - m_start
            if current_chunk_size == 0:
                continue

            # --- Get current chunk data ---
            l_chunk = l_dirs_world[m_start:m_end, :]  # (chunk_size, 3)
            l_solid_angles_chunk = l_solid_angles[m_start:m_end, :]  # (chunk_size, 1)
            Li_chunk = Li[:, m_start:m_end, :]  # (1, chunk_size, 3)
            env_vis_chunk = env_vis[:, :, m_start:m_end].unsqueeze(-1)  # (B, N, chunk_size, 1)

            # --- Expand tensors for the CURRENT CHUNK ---
            # Target shape: (B, N, V, chunk_size, dim)
            with torch.amp.autocast('cuda', enabled=True):
                n_exp = n_exp_base.unsqueeze(3).expand(B, N, V, current_chunk_size, 3)
                v_exp = v_exp_base.unsqueeze(3).expand(B, N, V, current_chunk_size, 3)
                l_exp = l_chunk.view(1, 1, 1, current_chunk_size, 3).expand(B, N, V, current_chunk_size, 3)

                albedo_exp = albedo_exp_base.unsqueeze(3).expand(B, N, V, current_chunk_size, 3)
                metallic_exp = metallic_exp_base.unsqueeze(3).expand(B, N, V, current_chunk_size, 1)
                roughness_exp = roughness_exp_base.unsqueeze(3).expand(B, N, V, current_chunk_size, 1)

                Li_exp = Li_chunk.unsqueeze(1).unsqueeze(1).expand(B, N, V, current_chunk_size, 3)
                l_solid_angles_exp = l_solid_angles_chunk.view(1, 1, 1, current_chunk_size, 1).expand(B, N, V,
                                                                                                      current_chunk_size,
                                                                                                      1)
                env_vis_exp = env_vis_chunk.unsqueeze(2).expand(B, N, V, current_chunk_size, 1)

                # --- Apply Visibility ---
                visible_Li_exp = Li_exp * env_vis_exp

                # --- Calculate PBR Components for the chunk ---
                h = F.normalize(v_exp + l_exp, p=2, dim=-1)
                NdotL = torch.sum(n_exp * l_exp, dim=-1, keepdim=True).clamp(min=EPS)
                NdotV = torch.sum(n_exp * v_exp, dim=-1, keepdim=True).clamp(min=EPS)
                NdotH = torch.sum(n_exp * h, dim=-1, keepdim=True).clamp(min=0.0)
                LdotH = torch.sum(l_exp * h, dim=-1, keepdim=True).clamp(min=0.0)
                VdotH = LdotH

                f0 = f0_base.unsqueeze(3).expand(B, N, V, current_chunk_size, 3)
                Fresnel = F_schlick(f0, VdotH)
                D = D_ggx(NdotH, roughness_exp)
                G = G_smith_ggx(NdotV, NdotL, roughness_exp)

                # --- Calculate BRDF for the chunk ---
                specular_term = cook_torrance_specular(NdotL, NdotV, Fresnel, D, G)
                kS = Fresnel
                diffuse_term = (1.0 - kS) * (1.0 - metallic_exp) * lambertian_diffuse(albedo_exp)
                brdf = diffuse_term + specular_term

                # --- Calculate Radiance Contribution for the chunk ---
                # Ensure calculations potentially done in float16 are cast back up for accumulation if needed
                # (though sum usually does this implicitly, explicit cast is safer if issues arise)
                radiance_samples_chunk = brdf * visible_Li_exp * NdotL * l_solid_angles_exp
                # <<< CHANGE: Accumulation type set explicitly to float32
                chunk_radiance = torch.sum(radiance_samples_chunk, dim=3, dtype=torch.float32)  # (B, N, V, 3)
            total_rendered_radiance += chunk_radiance

        return total_rendered_radiance


class Renderer(nn.Module):
    def __init__(self, chunk_size=64):
        super().__init__()
        self.hdr_renderer = HDRRenderer(chunk_size=chunk_size)

    def forward(self, points_normals: torch.Tensor,  # (B, N, 3) - Should be normalized
                points_albedo: torch.Tensor,  # (B, N, 3) - [0, 1]
                points_metallic: torch.Tensor,  # (B, N, 1) - [0, 1]
                points_smoothness: torch.Tensor,  # (B, N, 1) - [0, 1]
                viewing_dirs: torch.Tensor,  # (B, N, V, 3) - V is num views (64), should be normalized
                env_vis: torch.Tensor,
                env_map: torch.Tensor,
                exposure: Optional[float] = 1,
                return_hdr=False,
                return_exposure=False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Renders points using PBR IBL from an environment map.

        Args:
            points_pos: Point positions (B, N, 3).
            points_normals: Point normals (B, N, 3).
            points_albedo: Point albedo (B, N, 3).
            points_metallic: Point metallic values (B, N, 1).
            points_smoothness: Point smoothness values (B, N, 1).
            viewing_dirs: Viewing directions for each point (B, N, V, 3).
            env_map: Equirectangular HDR environment map (H, W, 3).
            envmap_samples: Optional tuple of precomputed (sample_dirs, solid_angles, sample_uvs)
                            to avoid recomputing them every call.

        Returns:
            rendered_radiance: (B, N, V, 3) - HDR radiance for each point/view direction.
        """
        hdr_radiance = self.hdr_renderer(points_normals, points_albedo, points_metallic, points_smoothness,
                                         viewing_dirs, env_vis, env_map)

        if exposure is None:
            # https://alienryderflex.com/hsp.html
            luminance = torch.sqrt(0.299 * hdr_radiance[..., 0] ** 2 +
                                   0.587 * hdr_radiance[..., 1] ** 2 +
                                   0.114 * hdr_radiance[..., 2] ** 2)

            # https://www-sciencedirect-com.ejournals.um.edu.mt/topics/computer-science/average-luminance
            log_lum = torch.log(luminance + EPS)
            avg_log_lum = torch.mean(log_lum.reshape(log_lum.shape[0], -1), dim=-1)
            log_avg_lum = torch.exp(avg_log_lum)
            exposure = 0.1 / log_avg_lum

        tone_mapped = tonemap_aces_filmic_better(hdr_radiance, exposure)

        ret = [tone_mapped]

        if return_hdr:
            ret.append(hdr_radiance)
        if return_exposure:
            ret.append(exposure)

        return ret[0] if len(ret) == 1 else tuple(ret)
