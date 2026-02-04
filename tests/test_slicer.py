from utils.load_dm import DMFileInfo
from utils.image_slicer import ImageSlicer
import cv2
import os

if __name__ == "__main__":
    # 读取dm3/dm4文件
    dm_path = "data"  # 假设数据在data目录
    files = [os.path.join(dm_path, f) for f in os.listdir(dm_path) if f.endswith(('.dm3', '.dm4'))]
    if not files:
        print("No dm3/dm4 files found.")
        exit(0)
    # 只处理第一张做演示
    info = DMFileInfo(files[0])
    img = info.get_data()
    print(f"Loaded: {files[0]}, shape: {img.shape}")
    # 归一化用于切片和展示
    img_norm = info.normalize(mode='percentile', out_range=(0, 255))
    # 切片
    slicer = ImageSlicer(patch_size=1024, overlap=128, mode='hann')
    patches, positions = slicer.slice(img_norm)
    print(f"Total patches: {len(patches)}")
    # 展示前9个切片
    import matplotlib.pyplot as plt
    plt.figure(figsize=(25,25))
    for i in range(min(25, len(patches))):
        plt.subplot(5,5,i+1)
        plt.imshow(patches[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
