#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   dinov3_seg.py
@Time    :   2026/02/04 22:20:37
@Author  :   Ethan
@Email   :   ethanrise.ai@gmail.com
@Version :   1.0
@Desc    :   DINOv3 multi-layer feature extraction + fusion + segmentation
@Note    :   None
'''

# ---------------------- Third-party Library Imports ----------------------


import sys
sys.path.insert(0, "../external/dinov3")


# ---------------------- Third-party Libraries ----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from safetensors.torch import load_file


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from safetensors.torch import load_file
from dinov3.models.vision_transformer import vit_large


class DINOv3SegmentationModel(nn.Module):
    """
    DINOv3 ViT-L/16 based segmentation model
    - Frozen backbone
    - Multi-layer feature extraction
    - Lightweight fusion + segmentation head
    """

    def __init__(
        self,
        weight_path: str,
        img_size: int = 1024,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        selected_layers=(5, 11, 17, 23),
        fusion_out_channels: int = 256,
        device: str | None = None,
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.selected_layers = selected_layers

        # ---------------- backbone ----------------
        self.backbone = vit_large(
            patch_size=patch_size,
            img_size=img_size,
        )

        state_dict = load_file(weight_path)
        missing, unexpected = self.backbone.load_state_dict(
            state_dict, strict=False
        )

        print(f"[DINOv3] Missing keys    : {len(missing)}")
        print(f"[DINOv3] Unexpected keys : {len(unexpected)}")

        self.backbone.eval().to(self.device)

        # ❗ freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ---------------- fusion ----------------
        self.fusion = nn.Conv2d(
            hidden_dim * len(selected_layers),
            fusion_out_channels,
            kernel_size=1,
            bias=False,
        )

        # ---------------- segmentation head ----------------
        self.seg_head = nn.Sequential(
            nn.Conv2d(fusion_out_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

        # ---------------- image transform ----------------
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        self.to(self.device)

    # ====================================================
    # utils
    # ====================================================
    @staticmethod
    def tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C]
        return: [B, C, H, W]
        """
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Token number {N} is not a square"
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)

    def load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        x = self.transform(img).unsqueeze(0)
        return x.to(self.device)

    # ====================================================
    # feature extraction
    # ====================================================
    def extract_multi_layer_tokens(self, x: torch.Tensor) -> dict:
        features = {}
        hooks = []

        def make_hook(layer_id):
            def hook(module, input, output):
                # DINOv3 block output is list / tuple
                out = output[0] if isinstance(output, (list, tuple)) else output
                # [B, N+1, C] → remove CLS
                features[layer_id] = out[:, 1:, :]
            return hook

        for lid in self.selected_layers:
            h = self.backbone.blocks[lid].register_forward_hook(
                make_hook(lid)
            )
            hooks.append(h)

        with torch.no_grad():
            _ = self.backbone(x)

        for h in hooks:
            h.remove()

        return features  # dict[layer_id] = [B,4096,1024]

    # ====================================================
    # forward
    # ====================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]
        return: [B,1,img_size,img_size]
        """
        tokens_dict = self.extract_multi_layer_tokens(x)

        feature_maps = []
        for lid in self.selected_layers:
            fm = self.tokens_to_feature_map(tokens_dict[lid])
            feature_maps.append(fm)

        # fusion
        fused = torch.cat(feature_maps, dim=1)
        fused = self.fusion(fused)

        # segmentation head
        seg = self.seg_head(fused)

        # upsample to pixel-level
        seg = F.interpolate(
            seg,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        return seg

    # ====================================================
    # inference helper
    # ====================================================
    @torch.no_grad()
    def infer_from_path(self, image_path: str) -> torch.Tensor:
        x = self.load_image(image_path)
        return self.forward(x)
    

    # ====================================================
    # visualization utils
    # ====================================================
    @staticmethod
    def _show_map(
        tensor: torch.Tensor,
        title: str,
        cmap="inferno",
        upsample_to: int | None = None,
    ):
        """
        tensor: [H,W] or [1,H,W]
        """
        if tensor.dim() == 3:
            tensor = tensor[0]

        if upsample_to is not None:
            tensor = F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=(upsample_to, upsample_to),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        img = tensor.detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.axis("off")
        plt.show()   

    @torch.no_grad()
    def visualize_patch_layer(
        self,
        image_path: str,
        layer_id: int,
        upsample: bool = True,
    ):
        x = self.load_image(image_path)
        tokens_dict = self.extract_multi_layer_tokens(x)

        assert layer_id in tokens_dict, f"Layer {layer_id} not extracted"

        tokens = tokens_dict[layer_id]           # [1,N,C]
        fmap = self.tokens_to_feature_map(tokens)  # [1,C,64,64]

        # mean over channels → activation
        act = fmap.mean(dim=1)  # [1,64,64]

        self._show_map(
            act,
            title=f"DINOv3 Layer {layer_id} Patch Activation",
            upsample_to=self.img_size if upsample else None,
        )
 
    @torch.no_grad()
    def visualize_segmentation(
        self,
        image_path: str,
    ):
        seg = self.infer_from_path(image_path)  # [1,1,H,W]

        self._show_map(
            seg[0],
            title="DINOv3 Segmentation Output",
            upsample_to=None,
        )


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    WEIGHT_PATH = (
        "/home/ethan/llm_models/modelscope/facebook/"
        "dinov3-vitl16-pretrain-lvd1689m/model.safetensors"
    )
    IMAGE_PATH = (
        "/home/ethan/Pictures/tem_data/Service_MA_TEM_02.jpg"
    )

    model = DINOv3SegmentationModel(
    weight_path=WEIGHT_PATH
    )

    seg_map = model.infer_from_path(IMAGE_PATH)
    print(seg_map.shape)  # [1,1,1024,1024]

    # 看某一层的 patch 特征
    model.visualize_patch_layer(
        IMAGE_PATH,
        layer_id=11,
    )