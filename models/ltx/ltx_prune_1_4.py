import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.elementwise_affine:
            hidden_states = hidden_states * self.weight
        return hidden_states.to(input_dtype)


class LTXVideoDepthwiseCausalConv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)

        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)


        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            padding=padding,
            padding_mode=padding_mode,
        )

        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]

        pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, time_kernel_size - 1, 1, 1))
        hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)

        hidden_states = self.depthwise(hidden_states)
        hidden_states = self.pointwise(hidden_states)
        return hidden_states


class LTXVideoDepthwiseCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)

        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)


        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            padding=padding,
            padding_mode=padding_mode,
        )

        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]
        pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, time_kernel_size - 1, 1, 1))
        hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)

        hidden_states = self.depthwise(hidden_states)
        hidden_states = self.pointwise(hidden_states)
        return hidden_states

class LTXVideoDepthwiseResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        force_shortcut: bool = False,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = RMSNorm(in_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.conv1 = LTXVideoDepthwiseCausalConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)

        self.norm2 = RMSNorm(out_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = LTXVideoDepthwiseCausalConv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)

        self.norm3 = None
        self.conv_shortcut = None

        if in_channels != out_channels or force_shortcut:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            self.conv_shortcut = LTXVideoDepthwiseCausalConv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1
            )

        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def forward(self, inputs: torch.Tensor, temb: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        hidden_states = inputs
        hidden_states = self.norm1(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.scale_shift_table is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class LTXVideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        is_causal: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]

        if self.is_causal:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, time_kernel_size - 1, 1, 1))
            hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)
        else:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            hidden_states = torch.concatenate([pad_left, hidden_states, pad_right], dim=2)

        hidden_states = self.conv(hidden_states)
        return hidden_states


class LTXVideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        force_shortcut: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = RMSNorm(in_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.conv1 = LTXVideoCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, is_causal=is_causal
        )

        self.norm2 = RMSNorm(out_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = LTXVideoCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, is_causal=is_causal
        )

        self.norm3 = None
        self.conv_shortcut = None


        if in_channels != out_channels or force_shortcut:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            self.conv_shortcut = LTXVideoCausalConv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, is_causal=is_causal
            )

        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def forward(
        self, inputs: torch.Tensor, temb: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.scale_shift_table is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class LTXVideoDownsampler3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        is_causal: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.group_size = (in_channels * stride[0] * stride[1] * stride[2]) // out_channels

        out_channels = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])

        self.conv = LTXVideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.cat([hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2)

        residual = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size))
        residual = residual.mean(dim=2)

        hidden_states = self.conv(hidden_states)
        hidden_states = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        hidden_states = hidden_states + residual

        return hidden_states


class LTXVideoUpsampler3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        is_causal: bool = True,
        residual: bool = False,
        upscale_factor: int = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.residual = residual
        self.upscale_factor = upscale_factor

        out_channels = (in_channels * stride[0] * stride[1] * stride[2]) // upscale_factor

        self.conv = LTXVideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        if self.residual:
            residual = hidden_states.reshape(
                batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
            )
            residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
            repeats = (self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
            residual = residual.repeat(1, repeats, 1, 1, 1)
            residual = residual[:, :, self.stride[0] - 1 :]

        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        hidden_states = hidden_states[:, :, self.stride[0] - 1 :]

        if self.residual:
            hidden_states = hidden_states + residual

        return hidden_states


class LTXVideoDownBlock3D(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        is_causal: bool = True,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTXVideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList(
                [
                    LTXVideoCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        is_causal=is_causal,
                    )
                ]
            )

        self.conv_out = None
        if in_channels != out_channels:
            self.conv_out = LTXVideoResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                is_causal=is_causal,
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        if self.conv_out is not None:
            hidden_states = self.conv_out(hidden_states, temb, generator)

        return hidden_states


class LTXVideo095DownBlock3D(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        is_causal: bool = True,
        downsample_type: str = "conv",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTXVideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList()

            if downsample_type == "conv":
                self.downsamplers.append(
                    LTXVideoCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        is_causal=is_causal,
                    )
                )
            elif downsample_type == "spatial":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels, out_channels=out_channels, stride=(1, 2, 2), is_causal=is_causal
                    )
                )
            elif downsample_type == "temporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels, out_channels=out_channels, stride=(2, 1, 1), is_causal=is_causal
                    )
                )
            elif downsample_type == "spatiotemporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels, out_channels=out_channels, stride=(2, 2, 2), is_causal=is_causal
                    )
                )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class LTXVideoMidBlock3d(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)


        BlockClass = LTXVideoDepthwiseResnetBlock3d if use_depthwise else LTXVideoResnetBlock3d

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                BlockClass(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        return hidden_states


class LTXVideoSpatialResnetBlock3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        force_shortcut: bool = False,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.nonlinearity = get_activation(non_linearity)


        self.norm1 = RMSNorm(in_channels, eps=1e-8, elementwise_affine=elementwise_affine)


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = RMSNorm(out_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(dropout)


        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm3 = None
        self.conv_shortcut = None

        if in_channels != out_channels or force_shortcut:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)

            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def forward(self, inputs: torch.Tensor, temb: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:

        hidden_states = inputs
        hidden_states = self.norm1(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.scale_shift_table is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)


        b, c, t, h, w = hidden_states.shape

        hidden_states_2d = hidden_states.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        hidden_states_2d = self.conv1(hidden_states_2d)


        c_new = hidden_states_2d.shape[1]
        hidden_states = hidden_states_2d.view(b, t, c_new, h, w).permute(0, 2, 1, 3, 4)


        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)


        b, c, t, h, w = hidden_states.shape
        hidden_states_2d = hidden_states.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        hidden_states_2d = self.conv2(hidden_states_2d)
        c_new = hidden_states_2d.shape[1]
        hidden_states = hidden_states_2d.view(b, t, c_new, h, w).permute(0, 2, 1, 3, 4)


        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]


        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)

        if self.conv_shortcut is not None:

            b, c, t, h, w = inputs.shape
            inputs_2d = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs_2d = self.conv_shortcut(inputs_2d)
            c_new = inputs_2d.shape[1]
            inputs = inputs_2d.view(b, t, c_new, h, w).permute(0, 2, 1, 3, 4)

        hidden_states = hidden_states + inputs
        return hidden_states

class LTXVideoUpBlock3d(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self, in_channels: int, out_channels: Optional[int] = None, num_layers: int = 1, dropout: float = 0.0, resnet_eps: float = 1e-6, resnet_act_fn: str = "swish", spatio_temporal_scale: bool = True, is_causal: bool = True, inject_noise: bool = False, timestep_conditioning: bool = False, upsample_residual: bool = False, upscale_factor: int = 1,
        use_depthwise: bool = False,
        force_shortcut: bool = False,
        use_spatial: bool = False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)

        self.conv_in = None
        if in_channels != out_channels or force_shortcut:


            if use_depthwise:
                ConvInClass = LTXVideoDepthwiseResnetBlock3d
            else:
                ConvInClass = LTXVideoResnetBlock3d

            self.conv_in = ConvInClass(
                in_channels=in_channels, out_channels=out_channels, dropout=dropout, eps=resnet_eps, non_linearity=resnet_act_fn, is_causal=is_causal, inject_noise=inject_noise, timestep_conditioning=timestep_conditioning,
                force_shortcut=force_shortcut
            )

        self.upsamplers = None
        if spatio_temporal_scale:


            self.upsamplers = nn.ModuleList([
                LTXVideoUpsampler3d(out_channels * upscale_factor, stride=(2, 2, 2), is_causal=is_causal, residual=upsample_residual, upscale_factor=upscale_factor)
            ])


        resnets = []
        if use_spatial:
            BlockClass = LTXVideoSpatialResnetBlock3d
        elif use_depthwise:
            BlockClass = LTXVideoDepthwiseResnetBlock3d
        else:
            BlockClass = LTXVideoResnetBlock3d

        for _ in range(num_layers):
            resnets.append(BlockClass(
                in_channels=out_channels, out_channels=out_channels, dropout=dropout, eps=resnet_eps, non_linearity=resnet_act_fn, is_causal=is_causal, inject_noise=inject_noise, timestep_conditioning=timestep_conditioning,
                force_shortcut=force_shortcut
            ))
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:

        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator)


        if self.time_embedder is not None:
            temb = self.time_embedder(timestep=temb.flatten(), resolution=None, aspect_ratio=None, batch_size=hidden_states.size(0), hidden_dtype=hidden_states.dtype).view(hidden_states.size(0), -1, 1, 1, 1)


        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)


        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        return hidden_states


class LTXVideoEncoder3d(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        down_block_types: Tuple[str, ...] = (
            "LTXVideoDownBlock3D",
            "LTXVideoDownBlock3D",
            "LTXVideoDownBlock3D",
            "LTXVideoDownBlock3D",
        ),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        downsample_type: Tuple[str, ...] = ("conv", "conv", "conv", "conv"),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels * patch_size**2

        output_channel = block_out_channels[0]

        self.conv_in = LTXVideoCausalConv3d(
            in_channels=self.in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
        )


        is_ltx_095 = down_block_types[-1] == "LTXVideo095DownBlock3D"
        num_block_out_channels = len(block_out_channels) - (1 if is_ltx_095 else 0)
        self.down_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            if not is_ltx_095:
                output_channel = block_out_channels[i + 1] if i + 1 < num_block_out_channels else block_out_channels[i]
            else:
                output_channel = block_out_channels[i + 1]

            if down_block_types[i] == "LTXVideoDownBlock3D":
                down_block = LTXVideoDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    is_causal=is_causal,
                )
            elif down_block_types[i] == "LTXVideo095DownBlock3D":
                down_block = LTXVideo095DownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    is_causal=is_causal,
                    downsample_type=downsample_type[i],
                )
            else:
                raise ValueError(f"Unknown down block type: {down_block_types[i]}")

            self.down_blocks.append(down_block)


        self.mid_block = LTXVideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[-1],
            resnet_eps=resnet_norm_eps,
            is_causal=is_causal,
        )


        self.norm_out = RMSNorm(out_channels, eps=1e-8, elementwise_affine=False)
        self.conv_act = nn.SiLU()
        self.conv_out = LTXVideoCausalConv3d(
            in_channels=output_channel, out_channels=out_channels + 1, kernel_size=3, stride=1, is_causal=is_causal
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = hidden_states.reshape(
            batch_size, num_channels, post_patch_num_frames, p_t, post_patch_height, p, post_patch_width, p
        )

        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(down_block, hidden_states)

            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)

            hidden_states = self.mid_block(hidden_states)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        hidden_states = torch.cat([hidden_states, last_channel], dim=1)

        return hidden_states


class LTXVideoDecoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = False,
        inject_noise: Tuple[bool, ...] = (False, False, False, False, False),
        timestep_conditioning: bool = False,
        upsample_residual: Tuple[bool, ...] = (False, False, False, False),
        upsample_factor: Tuple[int, ...] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels * patch_size**2


        block_out_channels = list(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))


        optimized_channels = []
        for i, ch in enumerate(block_out_channels):
            if i >= 2:
                optimized_channels.append(ch // 4)
            else:
                optimized_channels.append(ch)
        block_out_channels = tuple(optimized_channels)

        output_channel = block_out_channels[0]


        self.conv_in = LTXVideoCausalConv3d(
            in_channels=in_channels, out_channels=output_channel, kernel_size=3, stride=1, is_causal=is_causal
        )


        self.mid_block = LTXVideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            is_causal=is_causal,
            inject_noise=inject_noise[0],
            timestep_conditioning=timestep_conditioning,
            use_depthwise=True
        )


        self.up_blocks = nn.ModuleList([])
        num_block_out_channels = len(block_out_channels)

        for i in range(num_block_out_channels):


            use_depthwise = (i < 2)
            use_spatial = (i == 3)
            force_shortcut = (i >= 2)

            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]

            up_block = LTXVideoUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                is_causal=is_causal,
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
                use_depthwise=use_depthwise,
                use_spatial=use_spatial,
                force_shortcut=force_shortcut
            )
            self.up_blocks.append(up_block)

        self.norm_out = RMSNorm(out_channels, eps=1e-8, elementwise_affine=False)
        self.conv_act = nn.SiLU()


        self.conv_out = LTXVideoCausalConv3d(
            in_channels=output_channel,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal
        )

        self.time_embedder = None
        self.scale_shift_table = None
        self.timestep_scale_multiplier = None
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32))
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(output_channel * 2, 0)
            self.scale_shift_table = nn.Parameter(torch.randn(2, output_channel) / output_channel**0.5)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, feature_enabled: bool = False) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        feature_backup = {}
        if self.timestep_scale_multiplier is not None:
            temb = temb * self.timestep_scale_multiplier

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states, temb)
            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(up_block, hidden_states, temb)
        else:
            hidden_states = self.mid_block(hidden_states, temb)
            if feature_enabled:
                feature_backup["mid_block"] = hidden_states
            for index, up_block in enumerate(self.up_blocks):
                hidden_states = up_block(hidden_states, temb)
                if feature_enabled:
                    feature_backup[f"up_block_{index}"] = hidden_states

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(), resolution=None, aspect_ratio=None,
                batch_size=hidden_states.size(0), hidden_dtype=hidden_states.dtype
            ).view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift, scale = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        p = self.patch_size
        p_t = self.patch_size_t
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if feature_enabled:
            return hidden_states, feature_backup
        else:
            return hidden_states


class AutoencoderKLLTXVideo(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        down_block_types: Tuple[str, ...] = (
            "LTXVideoDownBlock3D",
            "LTXVideoDownBlock3D",
            "LTXVideoDownBlock3D",
            "LTXVideoDownBlock3D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        decoder_layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        decoder_spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        decoder_inject_noise: Tuple[bool, ...] = (False, False, False, False, False),
        downsample_type: Tuple[str, ...] = ("conv", "conv", "conv", "conv"),
        upsample_residual: Tuple[bool, ...] = (False, False, False, False),
        upsample_factor: Tuple[int, ...] = (1, 1, 1, 1),
        timestep_conditioning: bool = False,
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        scaling_factor: float = 1.0,
        encoder_causal: bool = True,
        decoder_causal: bool = False,
        spatial_compression_ratio: int = None,
        temporal_compression_ratio: int = None,
    ) -> None:
        super().__init__()

        self.encoder = LTXVideoEncoder3d(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            downsample_type=downsample_type,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=encoder_causal,
        )
        self.decoder = LTXVideoDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            spatio_temporal_scaling=decoder_spatio_temporal_scaling,
            layers_per_block=decoder_layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=decoder_causal,
            timestep_conditioning=timestep_conditioning,
            inject_noise=decoder_inject_noise,
            upsample_residual=upsample_residual,
            upsample_factor=upsample_factor,
        )

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

        self.spatial_compression_ratio = (
            patch_size * 2 ** sum(spatio_temporal_scaling)
            if spatial_compression_ratio is None
            else spatial_compression_ratio
        )
        self.temporal_compression_ratio = (
            patch_size_t * 2 ** sum(spatio_temporal_scaling)
            if temporal_compression_ratio is None
            else temporal_compression_ratio
        )

        self.use_slicing = False
        self.use_tiling = False
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False
        self.num_sample_frames_batch_size = 16
        self.num_latent_frames_batch_size = 2
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 16
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_num_frames: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_num_frames: Optional[float] = None,
    ) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_framewise_decoding and num_frames > self.tile_sample_min_num_frames:
            return self._temporal_tiled_encode(x)

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        enc = self.encoder(x)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor] = None, return_dict: bool = True,
        feature_enabled: bool = False
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio

        if self.use_framewise_decoding and num_frames > tile_latent_min_num_frames:
            return self._temporal_tiled_decode(z, temb, return_dict=return_dict)

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, temb, return_dict=return_dict)

        if feature_enabled:
            dec, feature = self.decoder(z, temb, feature_enabled=feature_enabled)
            return dec, feature
        dec = self.decoder(z, temb)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor] = None, return_dict: bool = True,
        feature_enabled: bool = False
    ) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            if temb is not None:
                decoded_slices = [
                    self._decode(z_slice, t_slice).sample for z_slice, t_slice in (z.split(1), temb.split(1))
                ]
            else:
                decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            if feature_enabled:
                decoded, feature = self._decode(z, temb, feature_enabled=feature_enabled)
                return decoded, feature
            decoded = self._decode(z, temb).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width


        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                time = self.encoder(
                    x[:, :, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                )

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):


                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor], return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:


        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width


        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                time = self.decoder(z[:, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width], temb)

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):


                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def _temporal_tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

        row = []
        for i in range(0, num_frames, self.tile_sample_stride_num_frames):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (height > self.tile_sample_min_height or width > self.tile_sample_min_width):
                tile = self.tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                result_row.append(tile[:, :, :tile_latent_stride_num_frames, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])

        enc = torch.cat(result_row, dim=2)[:, :, :latent_num_frames]
        return enc

    def _temporal_tiled_decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor], return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

        row = []
        for i in range(0, num_frames, tile_latent_stride_num_frames):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (tile.shape[-1] > tile_latent_min_width or tile.shape[-2] > tile_latent_min_height):
                decoded = self.tiled_decode(tile, temb, return_dict=True).sample
            else:
                decoded = self.decoder(tile, temb)
            if i > 0:
                decoded = decoded[:, :, :-1, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                tile = tile[:, :, : self.tile_sample_stride_num_frames, :, :]
                result_row.append(tile)
            else:
                result_row.append(tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :])

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, temb)
        if not return_dict:
            return (dec.sample,)
        return dec
