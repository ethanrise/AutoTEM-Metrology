import numpy as np
import sys
import cv2

# 推荐使用 hyperspy 支持 dm3/dm4
try:
    import hyperspy.api as hs
except ImportError:
    print("请先安装 hyperspy: pip install hyperspy")
    sys.exit(1)

class DMFileInfo:
    def __init__(self, filepath):
        self.filepath = filepath
        s = hs.load(filepath)
        self.shape = s.data.shape
        self.axes = s.axes_manager.as_dictionary()
        self.metadata = dict(s.metadata)
        self.original_metadata = s.original_metadata.as_dictionary() if hasattr(s, 'original_metadata') else None
        self.scale = {}
        self.units = {}
        for k, v in self.axes.items():
            self.scale[k] = v.get('scale', None)
            self.units[k] = v.get('units', None)
        self.annotations = self.original_metadata.get('AnnotationGroupList') if self.original_metadata else None

    def get_data(self):
        import hyperspy.api as hs
        s = hs.load(self.filepath)
        return s.data

    def get_display_image(self, mode='percentile'):
        return self.normalize(mode=mode, out_range=(0, 255))

    def normalize(self, mode='percentile', out_range=(0, 1)):
        data = self.get_data().astype(np.float32)
        if mode == 'percentile':
            vmin, vmax = np.percentile(data, 1), np.percentile(data, 99)
        elif mode == 'minmax':
            vmin, vmax = np.min(data), np.max(data)
        else:
            raise ValueError('Unknown mode')
        data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        if out_range == (0, 255):
            data = (data * 255).astype(np.uint8)
        elif out_range == (0, 1):
            data = data.astype(np.float32)
        else:
            raise ValueError('Unsupported out_range')
        return data

    def print_info(self):
        print(f"File: {self.filepath}")
        print(f"  Shape: {self.shape}")
        print(f"  Scale: {self.scale}")
        print(f"  Units: {self.units}")
        print(f"  Annotations: {self.annotations}")
        print()
