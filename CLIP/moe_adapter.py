import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


# Residual CLIP Adapter
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self,x):
        return self.fc(x)

class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        gate_score = F.softmax(self.gate(x), dim=-1)
        #print(gate_score.shape)
        #print(gate_score.unsqueeze(2).shape)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        #print(expert_outputs.shape)
        output = torch.matmul(gate_score.unsqueeze(2), expert_outputs).squeeze(2)
        #print(output.shape)
        return output

class Clip_MoE_Adapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(Clip_MoE_Adapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )

        self.fc2 = nn.Sequential(
            MoELayer(4, bottleneck, c_in),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class CLIP_MoE_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_MoE_adapters = nn.ModuleList([Clip_MoE_Adapter(1024, bottleneck=768) for i in range(len(features))])
        self.det_MoE_adapters = nn.ModuleList([Clip_MoE_Adapter(1024, bottleneck=768) for i in range(len(features))])

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            if i + 1 == 12:
                x = self.image_encoder.transformer.resblocks[i](x)

            else:
                x = self.image_encoder.transformer.resblocks[i](x)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_MoE_adapters[self.features.index(i + 1)](x)
                det_adapt_med, det_adapt_out = self.det_MoE_adapters[self.features.index(i + 1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        x = x.permute(1, 0, 2)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        pooled = 1

        return pooled, seg_patch_tokens, det_patch_tokens




