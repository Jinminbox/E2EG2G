"""Core E2EG2G model.

The module keeps the paper model in one place so dataset runners only need to
define their data protocol.  Inputs are expected as ``[batch, 1, channels,
time]`` tensors.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentNodeGenerator(nn.Module):
    """Map physical EEG channels to a fixed latent-node representation."""

    def __init__(
        self,
        n_channels,
        latent_nodes=32,
        f1=16,
        temporal_kernel=64,
        pool1=8,
        pool2=8,
        dropout=0.5,
    ):
        super().__init__()
        if latent_nodes % f1 != 0:
            raise ValueError("latent_nodes must be divisible by f1 for grouped spatial projection.")
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, latent_nodes, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(latent_nodes),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(latent_nodes, latent_nodes, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(latent_nodes),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.permute(0, 2, 1, 3).contiguous()


class RelationalGridModule(nn.Module):
    """Construct a 3-channel relation grid from latent nodes."""

    def __init__(self, latent_nodes=32, feature_time=15, dropout=0.5, normalize=True, weighted=True):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, latent_nodes, feature_time))
        self.node_projection = nn.Linear(latent_nodes, latent_nodes)
        self.dropout = nn.Dropout(dropout)
        self.normalize = normalize
        self.weighted = weighted
        self.fusion_logits = nn.Parameter(torch.zeros(3)) if weighted else None

    @staticmethod
    def _normalize_stack(x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, x):
        batch, c_p, n_nodes, length = x.shape
        x = x + self.positional_encoding

        flat = x.reshape(batch * c_p, n_nodes, length)
        sim = torch.bmm(flat, flat.transpose(1, 2)) / (length ** 0.5)
        g_sim = F.softmax(sim.reshape(batch, c_p, n_nodes, n_nodes), dim=-1)

        mean = x.mean(dim=-1)
        mean_flat = mean.reshape(batch * c_p, n_nodes, 1)
        g_outer = torch.bmm(mean_flat, mean_flat.transpose(1, 2)).reshape(batch, c_p, n_nodes, n_nodes)

        projected = self.node_projection(mean)
        projected = F.normalize(projected, dim=-1)
        g_proj = torch.matmul(projected.unsqueeze(-1), projected.unsqueeze(-2))

        grid = torch.stack([g_sim, g_outer, g_proj], dim=2)
        if self.normalize:
            grid = self._normalize_stack(grid)
        if self.weighted:
            weights = F.softmax(self.fusion_logits, dim=0).view(1, 1, 3, 1, 1)
            grid = grid * weights
        grid = grid.reshape(batch, c_p * 3, n_nodes, n_nodes)
        return self.dropout(grid)


class ConvNetGFE(nn.Module):
    """Default lightweight ConvNet GFE for 32 x 32 relation grids."""

    def __init__(self, dropout=0.5, out_dim=512):
        super().__init__()
        hidden = 32
        self.dropout = nn.Dropout(dropout)
        self.conv_1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, hidden, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(hidden),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(hidden, int(hidden * 1.5), kernel_size=3, bias=False),
            nn.BatchNorm2d(int(hidden * 1.5)),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(int(hidden * 1.5), hidden * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden * 4),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )
        self.max = nn.MaxPool2d(2)
        self.layer_second = nn.Sequential(nn.Linear(hidden * 2, out_dim), nn.BatchNorm1d(out_dim))
        self.layer_last = nn.Sequential(nn.Linear(hidden * 4, out_dim), nn.BatchNorm1d(out_dim))

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        out_4 = self.conv_4(out_3)
        out_3_pool = self.max(out_3)
        out1 = self.layer_last(out_4.flatten(start_dim=1))
        out2 = self.layer_second(out_3_pool.flatten(start_dim=1))
        return self.dropout(out1 + out2)


class ResNetGFE(nn.Module):
    """ResNet-based GFE used for controlled backbone replacement."""

    def __init__(self, out_dim=512, variant="resnet18"):
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("ResNetGFE requires torchvision. Install it or use gfe_backbone='convnet'.") from exc
        if variant == "resnet18":
            self.backbone = models.resnet18(weights=None)
            in_features = self.backbone.fc.in_features
        elif variant == "resnet50":
            self.backbone = models.resnet50(weights=None)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(in_features, out_dim)

    def forward(self, x):
        return self.backbone(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.5):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, n_tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(batch, n_tokens, dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.5):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViTGFE(nn.Module):
    """Compact ViT-style GFE for relation-grid replacement experiments."""

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        dim=128,
        depth=2,
        heads=4,
        mlp_ratio=2.0,
        out_dim=512,
        dropout=0.5,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, dim, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch, channels, height, width = x.shape
        patch = self.patch_size
        patches = x.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.contiguous().view(batch, channels, -1, patch, patch)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(batch, -1, channels * patch * patch)
        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.blocks(self.dropout(tokens))
        return self.head(self.norm(tokens[:, 0]))


class GridFeatureExtractor(nn.Module):
    """GFE wrapper for plug-and-play relation-grid backbones.

    The default paper model uses ``ConvNetGFE``.  ``ResNetGFE`` and
    ``TinyViTGFE`` are included to make the backbone replacement interface
    explicit for readers and follow-up experiments.
    """

    def __init__(self, backbone="convnet", dropout=0.5, out_dim=512, **kwargs):
        super().__init__()
        name = backbone.lower()
        if name == "convnet":
            self.backbone = ConvNetGFE(dropout=dropout, out_dim=out_dim)
        elif name in ("resnet18", "resnet50"):
            self.backbone = ResNetGFE(out_dim=out_dim, variant=name)
        elif name in ("vit", "tiny_vit", "tinyvit"):
            self.backbone = TinyViTGFE(out_dim=out_dim, dropout=dropout, **kwargs)
        else:
            raise ValueError(f"Unsupported GFE backbone: {backbone}")
        self.backbone_name = name
        self.out_dim = out_dim

    def forward(self, x):
        return self.backbone(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=1)


class SupConLoss(nn.Module):
    """Supervised contrastive loss without augmented views."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)
        positives = torch.eq(labels, labels.T).float().to(device)
        positives.fill_diagonal_(0)
        has_positive = positives.sum(1) > 0
        if not has_positive.any():
            return torch.zeros((), device=device)

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        logits_mask = ~torch.eye(features.size(0), dtype=torch.bool, device=device)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (
            positives[has_positive] * log_prob[has_positive]
        ).sum(1) / positives[has_positive].sum(1)
        return -mean_log_prob_pos.mean()


class E2EG2G(nn.Module):
    """End-to-end E2EG2G model used by all released runners."""

    def __init__(
        self,
        n_channels,
        n_classes,
        input_time,
        latent_nodes=32,
        lng_f1=16,
        temporal_kernel=64,
        pool1=8,
        pool2=8,
        dropout=0.5,
        use_lng_residual=True,
        projection_dim=128,
        contrastive_target="rgm",
        rgm_normalize=True,
        rgm_weighted=True,
        gfe_backbone="convnet",
        gfe_grid_size=32,
        gfe_kwargs=None,
    ):
        super().__init__()
        feature_time = input_time // pool1 // pool2
        if feature_time <= 0:
            raise ValueError("input_time is too short for the selected pooling factors.")
        self.use_lng_residual = use_lng_residual
        self.contrastive_target = contrastive_target
        self.gfe_grid_size = gfe_grid_size
        self.latent_feature_dim = latent_nodes * feature_time
        self.grid_feature_dim = 512
        classifier_dim = self.grid_feature_dim + self.latent_feature_dim if use_lng_residual else self.grid_feature_dim

        self.lng = LatentNodeGenerator(
            n_channels=n_channels,
            latent_nodes=latent_nodes,
            f1=lng_f1,
            temporal_kernel=temporal_kernel,
            pool1=pool1,
            pool2=pool2,
            dropout=dropout,
        )
        self.rgm = RelationalGridModule(
            latent_nodes=latent_nodes,
            feature_time=feature_time,
            dropout=dropout,
            normalize=rgm_normalize,
            weighted=rgm_weighted,
        )
        self.gfe = GridFeatureExtractor(backbone=gfe_backbone, dropout=dropout, **(gfe_kwargs or {}))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(classifier_dim, n_classes))
        self.lng_projection = ProjectionHead(self.latent_feature_dim, projection_dim)
        self.rgm_projection = ProjectionHead(3 * latent_nodes * latent_nodes, projection_dim)

    def forward(self, x):
        lng = self.lng(x) * math.sqrt(16)
        grid = self.rgm(lng)
        grid_for_gfe = grid
        if self.gfe_grid_size is not None and grid.shape[-1] != self.gfe_grid_size:
            grid_for_gfe = F.interpolate(
                grid,
                size=(self.gfe_grid_size, self.gfe_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        grid_feature = self.gfe(grid_for_gfe)
        lng_flat = lng.flatten(start_dim=1)
        feature = torch.cat([lng_flat, grid_feature], dim=1) if self.use_lng_residual else grid_feature
        logits = self.classifier(feature)

        z = {}
        if self.contrastive_target in ("lng", "both"):
            z["lng"] = self.lng_projection(lng_flat)
        if self.contrastive_target in ("rgm", "both"):
            z["rgm"] = self.rgm_projection(grid.flatten(start_dim=1))
        return {"logits": logits, "z": z, "lng": lng, "grid": grid, "feature": feature}

