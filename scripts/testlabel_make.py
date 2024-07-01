import os
import shutil
import random
from PIL import Image
from torchvision.transforms.functional import pad, center_crop
import torchvision.transforms as transforms
# 原始图片大小
w, h = 210, 84

# 填充后的大小
pad_w, pad_h = 210, 210

# 计算填充的大小
pad_left = (pad_w - w) // 2
pad_right = pad_w - w - pad_left
pad_top = (pad_h - h) // 2
pad_bottom = pad_h - h - pad_top
pad_width = (pad_left, pad_top, pad_right, pad_bottom)
# 1001 1003 1201 1500 2000 2400 2701 3900 侧立 翻转 缺件
for now in ['1001','1003', '1201', '1500', '2000', '2400', '2701', '3900', '侧立', '翻转', '缺件']:
    # now = '1001'
    os.makedirs('/data/ljx/PCBR/电阻'+now)
    # 从单个电阻文件夹中采样出128张图片
    image_path = '/data/ljx/PCBR/电阻-'+now
    # sample_image_path = '/data/ljx/PCBR/'+now+'/sample_train_images'
    final_image_path = '/data/ljx/PCBR/'+now+'/images'
    if not os.path.exists(final_image_path):
        os.makedirs(final_image_path)
    '''
    if not os.path.exists(sample_image_path):
        os.makedirs(sample_image_path)

    files = os.listdir(image_path)
    img_files = [os.path.join(image_path, f) for f in files if f.endswith(".jpeg")]
    random_imgs = random.sample(img_files, 128)

    for img in random_imgs:
        shutil.move(img, sample_image_path)
    print(now+'r moved.')
    '''
    # 构建标注文件路径，从文件名中输出标注，并且重命名文件
    annotation_path = '/data/ljx/PCBR/'+now+'/labels'

    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)

    # 遍历图片文件夹
    i = 0
    for filename in os.listdir(image_path):
        if filename.endswith(".jpeg"):  # 只处理.jpeg文件
            i += 1
            # 获取文件名的前三个数字
            annotation = filename.replace(".jpeg", "").split("_")[0:3]
        
            # 构建标注文件名字
            annotation_filename = f'test{i:05d}_'+now+'.txt'
            
            # 将标注信息保存到文件中
            with open(os.path.join(annotation_path, annotation_filename), 'w') as f:
                f.write("\t".join(annotation))
            
            # 重命名图片
            os.rename(os.path.join(image_path, filename),
                    os.path.join(image_path, f'test{i:05d}_'+now+'.jpeg'))
            image = transforms.ToTensor()(Image.open(os.path.join(image_path, f'test{i:05d}_'+now+'.jpeg')))
            padded_image = pad(image, pad_width, padding_mode='constant')
            pil_image = transforms.ToPILImage()(padded_image)

            # 保存为图片
            pil_image.save(os.path.join(final_image_path, f'test{i:05d}_'+now+'.jpeg'))
    print(now+'finished.')