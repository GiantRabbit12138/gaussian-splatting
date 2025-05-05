import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tinycudann as tcnn

class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                bound: float = 1,
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == 'frequency':
        encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": multires
            }
        )

    elif encoding == 'sphere_harmonics':
        encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree
            },
            dtype=torch.float32
        )

    elif encoding == 'hashgrid':
        # 使用 tiny-cuda-nn 的哈希编码
        # 计算 per_level_scale
        max_res = float(desired_resolution * bound)
        per_level_scale = np.exp((np.log(max_res) - np.log(base_resolution)) / (num_levels-1))
        # per_level_scale = 2.0

        encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Linear"
            },
            dtype=torch.float32
        )
    
    elif encoding == 'tiledgrid':
        per_level_scale = 2.0
        encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Grid",
                "type": "Tiled",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Linear"
            }
        )

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.n_output_dims