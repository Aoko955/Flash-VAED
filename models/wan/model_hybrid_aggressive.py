import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.amp as amp

__all__ = [
    'WanVAE_',
    'WanVAE'
]

CACHE_T = 2

class CausalDepthwiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding
        assert len(self.padding) == 3
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, 0, dilation, groups=in_channels)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)

class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        shape = (dim, 1, 1) if channel_first and images else (dim, 1, 1, 1) if channel_first and not images else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        if self.channel_first and x.dim() == 5:
            b, c, t, h, w = x.shape
            x_reshaped = x.reshape(b, c, -1)
            norm_x = F.normalize(x_reshaped, dim=1) * self.scale
            return (norm_x.reshape(b, c, t, h, w) * self.gamma) + self.bias
        elif self.channel_first and x.dim() == 4:
            b_t, c, h, w = x.shape
            x_reshaped = x.reshape(b_t, c, -1)
            norm_x = F.normalize(x_reshaped, dim=1) * self.scale
            return (norm_x.reshape(b_t, c, h, w) * self.gamma) + self.bias
        else:
            return F.normalize(x, dim=-1) * self.scale * self.gamma + self.bias

class ResidualBlockFast(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalDepthwiseConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalDepthwiseConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalDepthwiseConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        res = x
        for layer in self.residual:
            if isinstance(layer, (CausalDepthwiseConv3d, CausalConv3d)) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = res[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                    cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(res.device), cache_x], dim=2)
                res = layer(res, feat_cache[idx] if feat_cache[idx] != 'Rep' else None)
                feat_cache[idx] = cache_x.detach() if feat_cache[idx] != 'Rep' else 'Rep'
                feat_idx[0] += 1
            else:
                res = layer(res)
        return h + res

class ResidualBlock2D(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=True), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=True), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        assert x.dim() == 4, f"ResidualBlock2D 需 4D 输入，实际为 {x.dim()}D"
        return self.residual(x) + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        assert x.dim() == 4, f"AttentionBlock expects a 4D tensor, but got {x.dim()}D"
        identity = x
        b_t, c, h, w = x.shape
        x_norm = self.norm(x)
        q, k, v = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(lambda tens: rearrange(tens, 'b c h w -> b (h w) c'), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.proj(out)
        return out + identity

class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, in_channels, out_channels, mode='3d'):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == '3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'),
                nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.time_conv = CausalConv3d(in_channels, in_channels * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        elif mode == '2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'),
                nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.time_conv = None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if self.mode == '2d':
            assert x.dim() == 4
            return self.resample(x)

        assert x.dim() == 5
        b, c, t, h, w = x.shape

        if self.time_conv is not None:
            if feat_cache is not None:
                idx = feat_idx[0]

                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                    x_time = x
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] != 'Rep':
                        cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(x.device), cache_x], dim=2)

                    x_time = self.time_conv(x, feat_cache[idx] if feat_cache[idx] != 'Rep' else None)

                    feat_cache[idx] = cache_x.detach()
                    feat_idx[0] += 1

                    x_time = x_time.reshape(b, 2, c, t, h, w)
                    x_time = torch.stack((x_time[:, 0], x_time[:, 1]), dim=3)
                    x_time = x_time.reshape(b, c, t * 2, h, w)
            else:

                x_time = x

            x = x_time

        t_new = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t_new, b=b)

        return x

class HybridDecoder3d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = CausalConv3d(16, 384, 3, padding=1)
        self.middle = nn.ModuleList([
            ResidualBlockFast(384, 384),
            AttentionBlock(384),
            ResidualBlockFast(384, 384)])

        self.upsamples = nn.ModuleList([
            ResidualBlockFast(384, 384),
            ResidualBlockFast(384, 384),
            ResidualBlockFast(384, 384),
            Resample(384, 192, mode='3d'),
            ResidualBlockFast(192, 384),
            ResidualBlockFast(384, 384),
            ResidualBlockFast(384, 384),
            Resample(384, 24, mode='3d'),
            ResidualBlock2D(24, 24),
            ResidualBlock2D(24, 24),
            ResidualBlock2D(24, 24),
            Resample(24, 12, mode='2d'),
            ResidualBlock2D(12, 12),
            ResidualBlock2D(12, 12),
            ResidualBlock2D(12, 12),
        ])

        self.head = nn.ModuleList([
            RMS_norm(12, images=False),
            nn.SiLU(),
            CausalConv3d(12, 3, 3, padding=1)
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b = x.shape[0]

        def _run_layer(layer, x_in):
            if isinstance(layer, (ResidualBlockFast, Resample)) and feat_cache is not None:
                return layer(x_in, feat_cache, feat_idx)
            elif isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_for_this_layer = feat_cache[idx]
                cache_to_save = x_in[:, :, -CACHE_T:, :, :].clone()
                if cache_to_save.shape[2] < 2 and cache_for_this_layer is not None and cache_for_this_layer != 'Rep':
                    cache_to_save = torch.cat([cache_for_this_layer[:, :, -1:, :, :].to(x_in.device), cache_to_save], dim=2)
                output = layer(x_in, cache_for_this_layer if cache_for_this_layer != 'Rep' else None)
                feat_cache[idx] = cache_to_save.detach() if cache_for_this_layer != 'Rep' else 'Rep'
                feat_idx[0] += 1
                return output
            else:
                return layer(x_in)

        x = _run_layer(self.conv1, x)
        x = _run_layer(self.middle[0], x)
        t_mid = x.shape[2]
        x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
        attn_out_2d = self.middle[1](x_2d)
        x = rearrange(attn_out_2d, '(b t) c h w -> b c t h w', b=b, t=t_mid)
        x = _run_layer(self.middle[2], x)

        is_2d_mode = False
        t_before_2d = x.shape[2]
        for i, layer in enumerate(self.upsamples):
            if i == 8 and not is_2d_mode:
                if x.dim() == 5:
                    t_before_2d = x.shape[2]
                    x = rearrange(x, 'b c t h w -> (b t) c h w')
                    is_2d_mode = True

            if is_2d_mode:
                x = layer(x)
            else:
                x = _run_layer(layer, x)

        x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=t_before_2d)
        x = self.head[0](x)
        x = self.head[1](x)
        x = _run_layer(self.head[2], x)

        return x

class Encoder3d(nn.Module):
    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout):
        super().__init__()
        self.model = nn.Identity()

    def forward(self, x, *args, **kwargs):
        return self.model(x), self.model(x)

def count_conv3d(model):
    return sum(1 for m in model.modules() if isinstance(m, (CausalConv3d, CausalDepthwiseConv3d)))

class WanVAE_(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = HybridDecoder3d()
        self.clear_cache()

    def forward(self, x, scale=None):
        scale = scale if scale is not None else [0., 1.]
        mu, log_var = self.encode(x, scale)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, scale)

    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out_chunks = []
        for i in range(iter_):
            self._enc_conv_idx = [0]
            start = 0 if i == 0 else 1 + 4 * (i-1)
            end = 1 if i == 0 else 1 + 4 * i
            chunk_mu, _ = self.encoder(x[:,:,start:end], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            out_chunks.append(chunk_mu)
        out = torch.cat(out_chunks, dim=2) if len(out_chunks) > 1 else out_chunks[0]
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu, log_var

    def decode(self, i, z, scale):
        if i == 0:
            self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        self._conv_idx = [0]
        out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        if i == 20:
            self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs, scale=[0., 1.])
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    model_args = {
        'dim': 96, 'z_dim': z_dim, 'dim_mult': [1, 2, 4, 4],
        'num_res_blocks': 2, 'attn_scales': [],
        'temperal_downsample': [False, True, True], 'dropout': 0.0}
    model_args.update(kwargs)
    model = WanVAE_(** model_args)
    if pretrained_path:
        logging.info(f'loading {pretrained_path}')
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    return model

class WanVAE:
    def __init__(self, vae_pth=None, device="cuda", z_dim=16, dtype=torch.float, **kwargs):
        self.dtype = dtype
        self.device = device
        mean = [-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]
        std = [2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]
        if vae_pth:
            self.model = _video_vae(pretrained_path=vae_pth, z_dim=z_dim, device=device, **kwargs).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        with amp.autocast('cuda', dtype=self.dtype):
            return [self.model.encode(u.unsqueeze(0), self.scale)[0].float().squeeze(0) for u in videos]

    def decode(self, zs):
        with amp.autocast('cuda', dtype=self.dtype):
            full_latent = torch.stack(zs) if isinstance(zs, list) else zs
            if full_latent.dim() == 4:
                full_latent = full_latent.unsqueeze(0)
            reconstructed_frames = []
            t_total = full_latent.shape[2]
            for i in range(t_total):
                frame = self.model.decode(i, full_latent, self.scale)
                reconstructed_frames.append(frame)
            reconstructed_video = torch.cat(reconstructed_frames, dim=2)
            return [reconstructed_video.squeeze(0).float().clamp_(-1, 1)]
