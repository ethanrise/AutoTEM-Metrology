import cv2
import numpy as np
from typing import List, Tuple

class ImageSlicer:
    def __init__(self, patch_size: int, overlap: int, mode: str = 'hann'):
        self.patch_size = patch_size
        self.overlap = overlap
        self.mode = mode

    def slice(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        h, w = image.shape[:2]
        step = self.patch_size - self.overlap
        patches = []
        positions = []
        for y in range(0, h - self.overlap, step):
            for x in range(0, w - self.overlap, step):
                y1 = min(y + self.patch_size, h)
                x1 = min(x + self.patch_size, w)
                patch = image[y:y1, x:x1]
                patches.append(patch)
                positions.append((y, x))
        return patches, positions

    def _get_weight_map(self, shape):
        h, w = shape
        if self.mode == 'hann':
            wy = np.hanning(h)
            wx = np.hanning(w)
            weight = np.outer(wy, wx)
        elif self.mode == 'gaussian':
            y = np.linspace(-1, 1, h)
            x = np.linspace(-1, 1, w)
            xx, yy = np.meshgrid(x, y)
            sigma = 0.5
            weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        else:
            weight = np.ones((h, w), dtype=np.float32)
        return weight.astype(np.float32)

    def merge(self, patches: List[np.ndarray], positions: List[Tuple[int, int]], image_shape: Tuple[int, int, int]) -> np.ndarray:
        h, w = image_shape[:2]
        c = image_shape[2] if len(image_shape) == 3 else 1
        result = np.zeros((h, w, c), dtype=np.float32)
        weight_sum = np.zeros((h, w, c), dtype=np.float32)
        for patch, (y, x) in zip(patches, positions):
            ph, pw = patch.shape[:2]
            weight_map = self._get_weight_map((ph, pw))
            if c > 1:
                weight_map = weight_map[..., None]
            y1 = min(y + ph, h)
            x1 = min(x + pw, w)
            patch_crop = patch[:y1-y, :x1-x]
            weight_crop = weight_map[:y1-y, :x1-x]
            result[y:y1, x:x1] += patch_crop * weight_crop
            weight_sum[y:y1, x:x1] += weight_crop
        merged = (result / np.maximum(weight_sum, 1e-6)).astype(np.uint8)
        if c == 1:
            merged = merged.squeeze(-1)
        return merged
