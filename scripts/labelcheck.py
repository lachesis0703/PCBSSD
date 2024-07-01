import os
import shutil 
import random
images_dir="/data/ljx/pcbrd/train/images/" 
labels_dir="/data/ljx/pcbrd/train/annotations/" 
processed_labels_dir="/data/ljx/pcbrd/train/Annotations/" 
if not os.path.exists(processed_labels_dir):
    os.mkdir(processed_labels_dir)
ratio = 0.1  # 抽取比例为20%
png_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]
n_samples = int(len(png_files) * ratio)
sample_files = random.sample(png_files, n_samples)
# 新建一个文件夹，用于存放抽取出来的文件
new_dir_path = "/data/ljx/pcbrd/train/Images/"
if not os.path.exists(new_dir_path):
    os.mkdir(new_dir_path)

# 将抽取出来的文件复制到新的文件夹中
for f in sample_files:
    shutil.move(f, new_dir_path)

for filename in os.listdir(new_dir_path):   
    if filename.endswith(".png"):       
        label_filename= os.path.join(labels_dir, os.path.splitext(filename)[0]+".txt")       
        if os.path.isfile(label_filename):           
            shutil.move(label_filename, processed_labels_dir)
print('finish')
# for now in ['1001','1003', '1201', '1500', '2000', '2400', '2701', '3900', '侧立', '翻转', '缺件']:
#     # 设置源文件夹和目标文件夹路径
#     source_folder_path = '/data/ljx/PCBR/'+now  # 修改为的父文件夹路径
#     destination_folder_path = '/data/ljx/PCBR/labels'    # 修改为你想要移动文件到的目标文件夹路径
#     # 确保目标文件夹存在
#     os.makedirs(destination_folder_path, exist_ok=True)
#     # 遍历源文件夹中的所有文件夹
#     for folder_name in os.listdir(source_folder_path):
#         print(folder_name)
#         folder_path = os.path.join(source_folder_path, folder_name)
#         print(os.path.isdir(folder_path))
#         print(os.path.basename(folder_path))
#         if os.path.isdir(folder_path) and os.path.basename(folder_path) == 'labels':
#             # 遍历子文件夹中的所有文件
#             for file_name in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, file_name)
#                 if os.path.isfile(file_path):
#                     # 移动文件到目标文件夹
#                     shutil.move(file_path, destination_folder_path)