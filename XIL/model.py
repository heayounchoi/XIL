from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit import (
    Block,
    PatchEmbed,
    VisionTransformer,
    build_model_with_cfg,
    checkpoint_filter_fn,
    resolve_pretrained_cfg,
)


class PromptedVisionTransformer(VisionTransformer):
    """ViT-B/16 with an optional task prompt inserted after the CLS token."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        init_values=None,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            representation_size=representation_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            init_values=init_values,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
        )

    def forward(self, x, prompt_tokens=None):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed.to(x.dtype)

        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens.to(x.dtype)
            prompt_tokens = prompt_tokens + torch.zeros(
                x.shape[0], prompt_tokens.shape[0], x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([x[:, :1, :], prompt_tokens, x[:, 1:, :]], dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 1:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        return x


def create_vit_b16(pretrained: bool = True):
    variant = "vit_base_patch16_224"
    pretrained_cfg = resolve_pretrained_cfg(variant)
    return build_model_with_cfg(
        PromptedVisionTransformer,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=False,
    )


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    batch_size = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), "Shape error"
    nb_classes = int(nb_classes)
    sim_per_class = out.view(batch_size, nb_classes, nb_proxy)
    attention = F.softmax(sim_per_class, dim=-1)
    return (attention * sim_per_class).sum(-1)


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.sigma = nn.Parameter(torch.Tensor(1)) if sigma else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, x):
        out = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.to_reduce:
            out = reduce_proxies(out, self.nb_proxy)
        if self.sigma is not None:
            out = self.sigma * out
        return out


class XEED(nn.Module):
    """
    Each task owns:
      - prompt_pool[task]: prompt generator weights, used as prompt tokens
      - classifier_pool[task]: base classifier on raw features
      - proto_classifier_pool[task]: prototype classifier on [prompted, raw] features
    """

    def __init__(self, args, task_order):
        super().__init__()
        self.args = args
        self.task_order = task_order
        self.numtask = 0
        self.image_encoder = create_vit_b16(pretrained=True)

        n_cls = args["total_cls"]
        dim = args["embd_dim"]
        n_tasks = args["total_sessions"]
        prompt_len = args["prompt_length"]

        self.prompt_pool = nn.ModuleList([nn.Linear(dim, prompt_len, bias=False) for _ in range(n_tasks)])
        self.classifier_pool = nn.ModuleList([CosineLinear(dim, n_cls) for _ in range(n_tasks)])
        self.proto_classifier_pool = nn.ModuleList([CosineLinear(dim * 2, n_cls) for _ in range(n_tasks)])

    def update_task(self):
        self.numtask += 1

    def extract_vector(self, images):
        features = self.image_encoder(images)
        return F.normalize(features, dim=-1)

    def forward(self, images):
        prompt = self.prompt_pool[self.numtask].weight
        features = self.image_encoder(images, prompt)
        return self.classifier_pool[self.numtask](features)

    def raw_features(self, images):
        return self.image_encoder(images)

    def prompted_features(self, images, task_id: int):
        return self.image_encoder(images, self.prompt_pool[task_id].weight)

    def concat_features(self, images, task_id: int):
        prompted = self.prompted_features(images, task_id)
        raw = self.raw_features(images)
        return torch.cat([prompted, raw], dim=1), raw
