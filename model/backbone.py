import torch
from torch import nn
import torch.nn.functional as F
from .network_blocks import BaseConv, CSPLayer, DWConv, get_activation


class MultiScaleFeatureFusion(nn.Module):
    """Fuse local detail, object-level context, and global context features."""

    def __init__(
            self,
            channels,
            act="silu",
            pool_kernel_sizes=(5, 9, 13),
            use_detail_branch=True,
            use_context_branches=True,
            use_global_branch=True,
            use_channel_gate=True,
            use_spatial_gate=True,
    ):
        super().__init__()
        self.use_detail_branch = use_detail_branch
        self.use_context_branches = use_context_branches
        self.use_global_branch = use_global_branch
        self.use_channel_gate = use_channel_gate
        self.use_spatial_gate = use_spatial_gate

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
        zero_branch = torch.zeros_like(y)

        detail = self.detail_branch(y) if self.use_detail_branch else zero_branch
        contexts = [
            branch(y) if self.use_context_branches else zero_branch
            for branch in self.context_branches
        ]
        if self.use_global_branch:
            global_context = F.interpolate(
                self.global_branch(y),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            global_context = zero_branch

        fused = self.fuse(torch.cat([detail] + contexts + [global_context], dim=1))
        if self.use_channel_gate:
            fused = fused * self.channel_gate(fused)
        if self.use_spatial_gate:
            fused = fused * self.spatial_gate(fused)
        return x + self.residual_weight * fused


class MultiLevelFeatureFusion(nn.Module):
    """Fuse shallow detail, mid-level context, and high-resolution semantic features."""

    def __init__(
            self,
            low_channels,
            mid_channels,
            out_channels,
            act="silu",
            pool_kernel_sizes=(5, 9, 13),
            use_detail_branch=True,
            use_context_branches=True,
            use_global_branch=True,
            use_channel_gate=True,
            use_spatial_gate=True,
    ):
        super().__init__()
        hidden_channels = max(out_channels // 2, 1)
        self.low_proj = BaseConv(low_channels, hidden_channels, 1, 1, act=act)
        self.mid_proj = BaseConv(mid_channels, hidden_channels, 1, 1, act=act)
        self.high_proj = BaseConv(out_channels, hidden_channels, 1, 1, act=act)
        self.fuse = nn.Sequential(
            BaseConv(hidden_channels * 3, out_channels, 3, 1, act=act),
            MultiScaleFeatureFusion(
                out_channels,
                act=act,
                pool_kernel_sizes=pool_kernel_sizes,
                use_detail_branch=use_detail_branch,
                use_context_branches=use_context_branches,
                use_global_branch=use_global_branch,
                use_channel_gate=use_channel_gate,
                use_spatial_gate=use_spatial_gate,
            ),
        )
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, low_feature, mid_feature, high_feature):
        target_size = high_feature.shape[-2:]
        low_feature = F.interpolate(
            self.low_proj(low_feature),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        mid_feature = F.interpolate(
            self.mid_proj(mid_feature),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        high_feature_proj = self.high_proj(high_feature)
        fused = self.fuse(torch.cat([low_feature, mid_feature, high_feature_proj], dim=1))
        return high_feature + self.residual_weight * fused


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
            use_ms_detail_branch=True,
            use_ms_context_branches=True,
            use_ms_global_branch=True,
            use_ms_channel_gate=True,
            use_ms_spatial_gate=True,
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
        self.use_multiscale_fusion = use_multiscale_fusion
        self.multiscale_fusion = (
            MultiLevelFeatureFusion(
                base_channels,
                base_channels * 2,
                base_channels * 2,
                act=act,
                pool_kernel_sizes=multiscale_pool_kernel_sizes,
                use_detail_branch=use_ms_detail_branch,
                use_context_branches=use_ms_context_branches,
                use_global_branch=use_ms_global_branch,
                use_channel_gate=use_ms_channel_gate,
                use_spatial_gate=use_ms_spatial_gate,
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
        if self.use_multiscale_fusion:
            x = self.multiscale_fusion(outputs["stem"], outputs["stage2"], x)
        return x#{k: v for k, v in outputs.items() if k in self.out_features}

class ResNet50Encoder(nn.Module):
    """Switchable ImageNet-pretrained ResNet50 encoder for TTC experiments."""

    _STAGE_CHANNELS = {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,
    }

    def __init__(
            self,
            out_stage="layer2",
            out_channels=24,
            pretrained=True,
            weights_path="",
            trainable=True,
            act="silu",
    ):
        super().__init__()
        if out_stage not in self._STAGE_CHANNELS:
            raise ValueError("out_stage must be one of %s, got %s" % (sorted(self._STAGE_CHANNELS), out_stage))

        backbone = self._build_resnet50(pretrained=pretrained and not weights_path)
        if weights_path:
            self._load_weights(backbone, weights_path)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_stage = out_stage
        self.output_channels = int(out_channels)

        stage_channels = self._STAGE_CHANNELS[out_stage]
        self.proj = (
            nn.Identity()
            if stage_channels == self.output_channels
            else BaseConv(stage_channels, self.output_channels, 1, 1, act=act)
        )

        if not trainable:
            for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in module.parameters():
                    param.requires_grad = False

    @staticmethod
    def _build_resnet50(pretrained=True):
        from torchvision.models import resnet50
        if not pretrained:
            try:
                return resnet50(weights=None)
            except TypeError:
                return resnet50(pretrained=False)
        try:
            from torchvision.models import ResNet50_Weights
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except (ImportError, AttributeError, TypeError):
            return resnet50(pretrained=True)

    @staticmethod
    def _load_weights(model, weights_path):
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        cleaned = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("backbone."):
                key = key[len("backbone."):]
            cleaned[key] = value
        model.load_state_dict(cleaned, strict=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        if self.out_stage == "layer1":
            return self.proj(x)
        x = self.layer2(x)
        if self.out_stage == "layer2":
            return self.proj(x)
        x = self.layer3(x)
        if self.out_stage == "layer3":
            return self.proj(x)
        x = self.layer4(x)
        return self.proj(x)

