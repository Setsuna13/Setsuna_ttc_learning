import torch
from torch import nn
import torch.nn.functional as F
from .network_blocks import BaseConv, CSPLayer, DWConv, get_activation


class MultiScaleFeatureFusion(nn.Module):
    """Fuse local detail, object-level context, and global context features."""

    def __init__(self, channels, act="silu", pool_kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = max(channels // 2, 1)
        self.reduce = BaseConv(channels, hidden_channels, 1, 1, act=act)

        self.detail_branch = nn.Sequential(
            DWConv(hidden_channels, hidden_channels, 3, act=act),
            BaseConv(hidden_channels, hidden_channels, 1, 1, act=act),
        )
        self.context_branches = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                BaseConv(hidden_channels, hidden_channels, 1, 1, act=act),
            )
            for kernel_size in pool_kernel_sizes
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=True),
            get_activation(act, inplace=True),
        )

        fusion_channels = hidden_channels * (len(pool_kernel_sizes) + 2)
        self.fuse = BaseConv(fusion_channels, channels, 1, 1, act=act)
        gate_channels = max(channels // 4, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, gate_channels, 1, bias=True),
            get_activation(act, inplace=True),
            nn.Conv2d(gate_channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3, bias=True),
            nn.Sigmoid(),
        )
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        y = self.reduce(x)
        h, w = y.shape[-2:]

        detail = self.detail_branch(y)
        contexts = [branch(y) for branch in self.context_branches]
        global_context = F.interpolate(
            self.global_branch(y),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        fused = self.fuse(torch.cat([detail] + contexts + [global_context], dim=1))
        fused = fused * self.channel_gate(fused) * self.spatial_gate(fused)
        return x + self.residual_weight * fused


class TTCBase(nn.Module):
    def __init__(
            self,
            dep_mul=1,
            wid_mul=1,
            depthwise=False,
            act="relu",
            kszie = 7,
            use_multiscale_fusion=False,
            multiscale_pool_kernel_sizes=(5, 9, 13),
    ):
        super().__init__()

        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 12)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = BaseConv(3,base_channels,ksize=kszie,act=act,stride=2)

        self.stage2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, kszie, 1, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 3, 2, 1,)
        self.stage3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 2, kszie, 1, act=act),
            Conv(base_channels * 2, base_channels * 2, kszie, 1, act=act)
        )
        self.multiscale_fusion = (
            MultiScaleFeatureFusion(
                base_channels * 2,
                act=act,
                pool_kernel_sizes=multiscale_pool_kernel_sizes,
            )
            if use_multiscale_fusion
            else nn.Identity()
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.stage2(x)
        outputs["stage2"] = x
        h, w = x.shape[-2:]
        x = self.upsample(x, output_size=(h * 2, w * 2))
        x = self.stage3(x)
        x = self.multiscale_fusion(x)
        return x#{k: v for k, v in outputs.items() if k in self.out_features}
