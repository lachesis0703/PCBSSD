import os
import numpy as np
from PIL import Image, ImageFilter

img_dir = '/data/ljx/mmrotate-dev-1.x/data/pcbmo/origintrain/images' # 修改为你的图片文件夹路径
blur_dir = '/data/ljx/mmrotate-dev-1.x/data/pcbmo/origintrain/BlurImages' # 模糊图像保存路径
if not os.path.exists(blur_dir):
    os.makedirs(blur_dir)

for filename in os.listdir(img_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        x1 = 42
        x2 = 86
        y1 = 0
        y2 = 128
        img_blur = Image.fromarray(img)
        for i in range(10):
            img_blur = img_blur.filter(ImageFilter.BLUR)
        img_center = img[x1:x2, y1:y2, :]
        img_blur = np.array(img_blur)
        img_blur[x1:x2, y1:y2, :] = img_center
        img_blur = Image.fromarray(img_blur)
        blur_path = os.path.join(blur_dir, filename)
        img_blur.save(blur_path, 'png')
