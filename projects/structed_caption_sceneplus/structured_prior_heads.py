from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_valid_mask(
    video_features: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if padding_mask is None:
        return torch.ones(
            video_features.shape[:2],
            device=video_features.device,
            dtype=torch.bool,
        )
    if padding_mask.dim() != 2 or padding_mask.shape[:2] != video_features.shape[:2]:
        raise ValueError("padding_mask must have shape [batch, frames] aligned with video_features.")
    if padding_mask.dtype == torch.bool:
        return padding_mask
    return padding_mask > 0


class MultiScaleTemporalAggregator(nn.Module):
    """Aggregate frame features with local temporal branches plus global attention."""

    def __init__(
        self,
        *,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.local_conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.local_conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.local_conv3 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)

        self.global_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.combine = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        video_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if video_features.dim() != 3:
            raise ValueError(
                f"video_features must have shape [batch, frames, dim], got {tuple(video_features.shape)}"
            )
        batch_size, num_frames, _hidden_dim = video_features.shape

        if padding_mask is not None:
            if padding_mask.dim() != 2 or padding_mask.shape[:2] != video_features.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape [batch, frames] aligned with video_features."
                )
            if padding_mask.dtype == torch.bool:
                valid_mask = padding_mask
                float_mask = valid_mask.to(dtype=video_features.dtype).unsqueeze(-1)
            else:
                float_mask = padding_mask.to(dtype=video_features.dtype).unsqueeze(-1)
                valid_mask = float_mask.squeeze(-1) > 0
        else:
            float_mask = torch.ones(
                batch_size,
                num_frames,
                1,
                device=video_features.device,
                dtype=video_features.dtype,
            )
            valid_mask = torch.ones(
                batch_size,
                num_frames,
                device=video_features.device,
                dtype=torch.bool,
            )

        x_t = video_features.transpose(1, 2)
        local1 = F.gelu(self.local_conv1(x_t)).transpose(1, 2)
        local2 = F.gelu(self.local_conv2(x_t)).transpose(1, 2)
        local3 = F.gelu(self.local_conv3(x_t)).transpose(1, 2)

        denom = float_mask.sum(dim=1).clamp(min=1.0)
        local1_pooled = (local1 * float_mask).sum(dim=1) / denom
        local2_pooled = (local2 * float_mask).sum(dim=1) / denom
        local3_pooled = (local3 * float_mask).sum(dim=1) / denom

        query = self.global_query.expand(batch_size, -1, -1)
        key_padding_mask = ~valid_mask
        global_feat, _attn_weights = self.global_attn(
            query=query,
            key=video_features,
            value=video_features,
            key_padding_mask=key_padding_mask,
        )
        global_feat = global_feat.squeeze(1)

        combined = torch.cat(
            [local1_pooled, local2_pooled, local3_pooled, global_feat],
            dim=-1,
        )
        return self.combine(combined)


class TemporalAttentionPool(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        video_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid_mask = _resolve_valid_mask(video_features, padding_mask)
        query = self.query.expand(video_features.size(0), -1, -1)
        pooled, _ = self.attn(
            query=query,
            key=video_features,
            value=video_features,
            key_padding_mask=~valid_mask,
        )
        return self.out_norm(pooled.squeeze(1))


class AttentionNextVLADAggregator(nn.Module):
    """Attention pooling fused with a compact NeXtVLAD-style temporal aggregator."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        num_clusters: int = 16,
        expansion: int = 2,
        groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.expanded_dim = int(d_model) * max(1, int(expansion))
        self.groups = max(1, int(groups))
        if self.expanded_dim % self.groups != 0:
            raise ValueError(
                f"expanded_dim={self.expanded_dim} must be divisible by groups={self.groups}."
            )
        self.group_dim = self.expanded_dim // self.groups
        self.num_clusters = max(4, int(num_clusters))

        self.expansion_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.expanded_dim),
        )
        self.group_attention = nn.Linear(self.expanded_dim, self.groups)
        self.cluster_assignment = nn.Linear(self.expanded_dim, self.groups * self.num_clusters)
        self.cluster_centers = nn.Parameter(
            torch.randn(self.num_clusters, self.group_dim) * 0.02
        )

        self.vlad_proj = nn.Sequential(
            nn.LayerNorm(self.num_clusters * self.group_dim),
            nn.Linear(self.num_clusters * self.group_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.attention_pool = TemporalAttentionPool(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attention_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        video_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if video_features.dim() != 3:
            raise ValueError(
                f"video_features must have shape [batch, frames, dim], got {tuple(video_features.shape)}"
            )

        valid_mask = _resolve_valid_mask(video_features, padding_mask)
        mask_f = valid_mask.to(dtype=video_features.dtype).unsqueeze(-1)

        expanded = self.expansion_proj(video_features)
        expanded = expanded * mask_f
        grouped = expanded.view(
            video_features.size(0),
            video_features.size(1),
            self.groups,
            self.group_dim,
        )

        group_attention = torch.sigmoid(self.group_attention(expanded)).unsqueeze(-1)
        cluster_logits = self.cluster_assignment(expanded).view(
            video_features.size(0),
            video_features.size(1),
            self.groups,
            self.num_clusters,
        )
        cluster_probs = torch.softmax(cluster_logits, dim=-1)

        valid_mask_4d = valid_mask.to(dtype=video_features.dtype).unsqueeze(-1).unsqueeze(-1)
        activation = group_attention * cluster_probs * valid_mask_4d

        activation_sum = activation.sum(dim=1).sum(dim=1)
        weighted_grouped = (
            activation.unsqueeze(-1) * grouped.unsqueeze(-2)
        ).sum(dim=1).sum(dim=1)
        residual = weighted_grouped - activation_sum.unsqueeze(-1) * self.cluster_centers.unsqueeze(0)
        residual = F.normalize(residual, p=2, dim=-1)
        residual = residual.reshape(video_features.size(0), self.num_clusters * self.group_dim)
        vlad_state = self.vlad_proj(residual)

        attention_state = self.attention_proj(self.attention_pool(video_features, padding_mask))
        fused = torch.cat([vlad_state, attention_state], dim=-1)
        return self.fuse(fused)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.norm(hidden_states))


class DeepResidualMLP(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_classes: int,
        hidden_dim: int = 2048,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_proj(features)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.output_norm(hidden_states)
        return self.output_proj(hidden_states)


class StructuredMultiSemanticPriorHead(nn.Module):
    """Drop-in replacement for simple structured prior heads."""

    expects_sequence_inputs = True

    def __init__(
        self,
        *,
        d_model: int,
        num_classes: int,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        num_mlp_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aggregator = MultiScaleTemporalAggregator(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = DeepResidualMLP(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_blocks=num_mlp_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        video_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        aggregated = self.aggregator(video_features, padding_mask)
        return self.classifier(aggregated)


class StructuredAttentionNextVLADPriorHead(nn.Module):
    """Sequence-aware prior head using attention pooling plus NeXtVLAD-style aggregation."""

    expects_sequence_inputs = True

    def __init__(
        self,
        *,
        d_model: int,
        num_classes: int,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        num_mlp_blocks: int = 4,
        num_clusters: int = 16,
        expansion: int = 2,
        groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aggregator = AttentionNextVLADAggregator(
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            expansion=expansion,
            groups=groups,
            dropout=dropout,
        )
        self.classifier = DeepResidualMLP(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_blocks=num_mlp_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        video_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        aggregated = self.aggregator(video_features, padding_mask)
        return self.classifier(aggregated)
