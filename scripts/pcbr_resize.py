import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义图片处理函数
def process_image(image_path):
    # 打开原始图片并获取其尺寸
    img = Image.open(image_path)
    width, height = img.size
    
    # 计算需要填充的空白区域
    x_pad = max(SIZE[0] - width, 0)
    y_pad = max(SIZE[1] - height, 0)
    left_pad = x_pad // 2
    right_pad = x_pad - left_pad
    top_pad = y_pad // 2
    bottom_pad = y_pad - top_pad
    
    # 使用pad函数进行填充
    pad_transform = transforms.Pad(padding=(left_pad, top_pad, right_pad, bottom_pad), fill=FILL_COLOR)
    padded_img = pad_transform(img)
    
    # 将图片调整为目标尺寸
    resize_transform = transforms.Resize(SIZE)
    resized_img = resize_transform(padded_img)
    
    # 保存处理后的图片
    processed_path = os.path.join("processedimages", os.path.basename(image_path))
    resized_img.save(processed_path, "JPEG")

# 设置图片处理后的尺寸和填充颜色
SIZE = (210, 210)
FILL_COLOR = (0,0,0)

# 创建存放处理后图片的文件夹
if not os.path.exists("processedimages"):
    os.makedirs("processedimages")

# 遍历images文件夹下的所有jpeg图片文件，并进行处理
for filename in os.listdir("images"):
    if filename.endswith(".jpeg"):
        img_path = os.path.join("images", filename)
        process_image(img_path)
