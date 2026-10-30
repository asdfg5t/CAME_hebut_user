#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalMoE(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_experts_per_modality: int = 6, 
                 num_selected_experts: int = 2,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts_per_modality = num_experts_per_modality
        self.num_selected_experts = num_selected_experts


        self.text_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts_per_modality)
        ])

        self.image_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts_per_modality)
        ])

        self.text_expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_experts_per_modality),
        )

        self.image_expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_experts_per_modality),
        )

        self.modality_fusion_router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )

        self.fusion_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts_per_modality)
        ])

        self.fusion_expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim , num_experts_per_modality),
        )

        self.modality_router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 3),

        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param_dtype = next(self.parameters()).dtype
        device = x.device
        if x.dtype != param_dtype:
            x = x.to(dtype=param_dtype)

        batch_size = x.shape[0]

        if x.shape[-1] == self.hidden_dim * 3:
            x_fusion, x_text, x_image = torch.split(x, self.hidden_dim, dim=-1)
        elif x.shape[-1] == self.hidden_dim * 2:
            x_text, x_image = torch.split(x, self.hidden_dim, dim=-1)
            x_fusion = 0.5 * (x_text + x_image)
        else:
            raise ValueError(f"Unexpected input dim {x.shape[-1]}, expected 2*hidden or 3*hidden ({self.hidden_dim})")

        modality_input = torch.cat([x_fusion, x_text, x_image], dim=-1)
        modality_weights = F.softmax(self.modality_router(modality_input), dim=-1)
        text_weight, image_weight, fusion_weight = modality_weights.split(1, dim=-1)

        text_expert_logits = self.text_expert_router(x_text)
        image_expert_logits = self.image_expert_router(x_image)
        fusion_expert_logits = self.fusion_expert_router(x_fusion)

        text_topk_values, text_topk_indices = torch.topk(text_expert_logits, self.num_selected_experts, dim=-1)
        image_topk_values, image_topk_indices = torch.topk(image_expert_logits, self.num_selected_experts, dim=-1)
        fusion_topk_values, fusion_topk_indices = torch.topk(fusion_expert_logits, self.num_selected_experts, dim=-1)

        text_expert_weights = F.softmax(text_topk_values, dim=-1)
        image_expert_weights = F.softmax(image_topk_values, dim=-1)
        fusion_expert_weights = F.softmax(fusion_topk_values, dim=-1)

        def compute_expert_outputs_selected(experts, topk_indices, inputs):
            batch_size_local, num_selected = topk_indices.shape
            num_experts = len(experts)

            expert_indices = topk_indices.flatten()
            sample_indices = torch.arange(batch_size_local, device=device).repeat_interleave(num_selected)

            selected_inputs = inputs[sample_indices]

            all_expert_outputs = []
            for expert_id in range(num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_inputs = selected_inputs[mask]
                    out = experts[expert_id](expert_inputs)
                    all_expert_outputs.append(out)
                else:
                    all_expert_outputs.append(torch.empty(0, self.hidden_dim, device=device, dtype=param_dtype))

            output_tensor = torch.zeros(batch_size_local * num_selected, self.hidden_dim, device=device, dtype=param_dtype)
            for expert_id in range(num_experts):
                expert_output = all_expert_outputs[expert_id]
                if expert_output.numel() > 0:
                    expert_positions = (expert_indices == expert_id).nonzero().squeeze(-1)
                    output_tensor[expert_positions] = expert_output.to(dtype=param_dtype)

            return output_tensor.view(batch_size_local, num_selected, self.hidden_dim)

        text_selected_outputs = compute_expert_outputs_selected(self.text_experts, text_topk_indices, x_text)
        image_selected_outputs = compute_expert_outputs_selected(self.image_experts, image_topk_indices, x_image)
        fusion_selected_outputs = compute_expert_outputs_selected(self.fusion_experts, fusion_topk_indices, x_fusion)

        text_fused = torch.sum(text_expert_weights.unsqueeze(-1) * text_selected_outputs, dim=1)
        image_fused = torch.sum(image_expert_weights.unsqueeze(-1) * image_selected_outputs, dim=1)
        fusion_fused = torch.sum(fusion_expert_weights.unsqueeze(-1) * fusion_selected_outputs, dim=1)

        final_fused = text_weight.squeeze(-1).unsqueeze(-1) * text_fused + \
                     image_weight.squeeze(-1).unsqueeze(-1) * image_fused + \
                     fusion_weight.squeeze(-1).unsqueeze(-1) * fusion_fused

        return final_fused


