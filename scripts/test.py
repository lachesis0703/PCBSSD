from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from dataset import ImageWithObbDataset, ImageNoLabDataset, ImageWithObbDataset2
import transforms_with_label
import torchvision.models as models
import cv2
import numpy as np
from shapely.geometry import box
from shapely import affinity
import matplotlib.pyplot as plt
from train import Net
import math
import time
from thop import profile
# Import the summary writer 
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object 
writer = SummaryWriter('logs')

def plot_one_rotated_box(img, obb, filename, sin_eval=False, color=(255, 255, 0)):
    line_thickness = 2
    # 65, 35/85, 50
    width, height, theta = obb[0,2], obb[0,3], obb[0,4]
    # if theta < 0:
    #     width, height, theta = 65, 35, theta + 90
    rect = [(obb[0,0], obb[0,1]), (65, 35), theta]
    if sin_eval is True:
        single_evaluate('/data/ljx/pcb-electric-resistance-dataset/testlabel2', rect, filename)
    poly = np.intp(np.round(cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    conf = 1
    res = (filename, conf, poly)
    # print('{}'.format(filename))
    # print('{} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(filename,conf,poly[0][0],poly[0][1],poly[1][0],poly[1][1],poly[2][0],poly[2][1],poly[3][0],poly[3][1]))
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=color, thickness=2 * tl)

def ang_norm(ang):
    while ang > 90 :
        if ang < 270:
            ang = 180 - ang
        if ang >= 270:
            ang = 360 - ang
        # if ang <= 0:
        #     ang = 360 + ang
        if ang > 0 and ang <= 90:
            break

    return float(ang) 

def ang_norm_rad(ang):
    while ang > 0.5*np.pi:
        if ang < 1.5*np.pi:
            ang = np.pi - ang
        if ang >= 1.5*np.pi:
            ang = 2*np.pi - ang
        # if ang <= 0:
        #     ang = 2*np.pi + ang
        if ang <= 0.5*np.pi and ang > 0:
            break
    return float(ang) 

def single_evaluate(testlabel_path,rect,filename):
    gt = []
    if testlabel_path is not None:
        with open(os.path.join(testlabel_path, filename +'.txt')) as f:
            gts = f.readlines()
        for lines in gts:
            words = lines.split('\t')
            if len(words) != 3:
                continue
            gt.append(tuple(x for x in words))
    gt = np.float32(gt)
    # print(filename,gt,gt[0],gt[0,0])    # test01060 [[-50. -16. 348.]] [-50. -16. 348.] -50.0
    (delta_x_gt, delta_y_gt) = (gt[0,0], gt[0,1])
    (ori_x_gt, ori_y_gt) = np.float32((105,105))
    (x_gt, y_gt) = (ori_x_gt+delta_x_gt, ori_y_gt+delta_y_gt)
    theta_gt = gt[0,2]
    (x_pred, y_pred) = rect[0]
    theta_pred = rect[-1]
    pos_err = math.sqrt(math.pow(x_pred - x_gt, 2) + math.pow(y_pred - y_gt, 2))
    theta_gt = ang_norm((theta_gt))
    theta_pred_90 = ang_norm(90 + theta_pred)
    ang_err0 = np.absolute(theta_gt-theta_pred)
    ang_err1 = np.absolute(theta_pred_90-theta_gt)
    ang_err2 = np.absolute(theta_pred_90+theta_gt)
    ang_err = min(ang_err0, ang_err1, ang_err2)
    if pos_err >= 5:
        print('Picture name is:', filename, '\n','Groundtruth: ', (x_gt, y_gt), theta_gt, '\n', 'Prediction: ', (x_pred, y_pred), theta_pred)
        print('Err: ', pos_err, ang_err)
        # print(filename)

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    err_x, err_y, err_ang = 0, 0, 0
    mse_x, mse_y, mse_ang = 0, 0, 0
    w, h = args.image_size
    time_start = time.time()
    with torch.no_grad():
        for data, target, fname in test_loader:
            target = (target[:, 0, :] + torch.Tensor([105,105,0]) - torch.Tensor([w/2, h/2, 0]))/ torch.Tensor([w/2, h/2, 1])  # 标签文件以及原图片合成的gt(位置)
            target[:,-1] = ang_norm(target[:, -1])/180*np.pi  # 修改标签文件的角度范围到-90～90-->-pi/2~pi/2
            data, target = data.to(device), target.to(device)
            # 生成5个view
            data_rot = transforms.functional.rotate(data,180)
            sft_x_ratio = 0.5   # 0.2
            sft_y_ratio = 0.5   # 0.2
            sx = np.random.uniform(-1,1)*sft_x_ratio*(args.image_size[0]/2)
            sy = np.random.uniform(-1,1)*sft_y_ratio*(args.image_size[1]/2)
            data_sft =transforms.functional.affine(data,0,(sx,sy),1,(0,0),fill=0)
            data_flp = transforms.functional.vflip(data)
            rot_ratio = 0.5
            theta_rot = np.random.uniform(-1, 1) * rot_ratio * 90
            data_rot_v2 = transforms.functional.rotate(data,theta_rot,fill=0)
            # 计算loss
            output, output_rot, output_sft, output_flp, output_rot_v2 = model(torch.cat((data, data_rot, data_sft, data_flp, data_rot_v2), 0)).view(5, -1, 5)
            type_loss = nn.SmoothL1Loss()
            loss_rot = type_loss(output[:,0:2]+output_rot[:,0:2], torch.zeros_like(output[:,0:2]+output_rot[:,0:2]))# 旋转一致性损失        
            loss_sft = type_loss(output_sft[:,0:2]-output[:,0:2]-output.new_tensor([sx/(args.image_size[0]/2), sy/(args.image_size[1]/2)]), torch.zeros_like(output_sft[:,0:2]-output[:,0:2]-output.new_tensor([sx/(args.image_size[0]/2),sy/(args.image_size[1]/2)])))# 平移一致性损失
            d_ang_rot = (output[:,4:5] - output_rot_v2[:,4:5] - theta_rot / 180 * torch.pi + torch.pi / 2) % (torch.pi / 1) - torch.pi / 2
            d_ang_flp = (output[:,4:5] + output_flp[:,4:5]+ torch.pi / 2) % (torch.pi / 1) - torch.pi / 2
            loss_flp = type_loss(d_ang_flp, torch.zeros_like(d_ang_flp))
            loss_rot_v2 = type_loss(d_ang_rot, torch.zeros_like(d_ang_rot))
            a = np.float32(0.05)
            loss_cen = a*loss_rot + loss_sft
            loss_ang = a*loss_flp + loss_rot_v2
            test_loss +=  loss_ang + loss_cen
            # 计算error
            err_x += torch.sum(abs((output[:, 0] - target[:, 0]) * w)).item()
            err_y += torch.sum(abs((output[:, 1] - target[:, 1]) * w)).item()
            mse_x = F.mse_loss(output[:, 0],target[:, 0], reduction='sum').item()
            mse_y = F.mse_loss(output[:, 1],target[:, 1], reduction='sum').item()
            # torch.sum可以用于多目标计算
            err_ang1 = torch.sum(torch.abs(output[:,-1] - target[:,-1]) * 180 / np.pi )
            err_ang2 = torch.sum(torch.abs(ang_norm_rad(output[:,-1] + np.pi / 2) - target[:,-1]) * 180 / np.pi)
            err_ang3 = torch.sum(torch.abs(abs(output[:,-1]) - abs(target[:,-1])) * 180 / np.pi)
            delta_err_ang = min(err_ang1,err_ang2,err_ang3)
            # delta_err_ang = F.mse_loss(output[:, -1],target[:, -1], reduction='sum')
            err_ang += delta_err_ang.item()
            mse_ang = F.mse_loss(output[:,-1],target[:,-1], reduction='sum').item()
            # 在这里把结果画出来
            for img, bb, fn in zip(data.permute(0, 2, 3, 1).contiguous().cpu().numpy() * 255, 
                                   output.cpu().numpy(),
                                   fname):
                bb = bb * np.float32([w/2,h/2,0,0,180 / np.pi])+np.float32([w/2,h/2,0,0,0])
                plot_one_rotated_box(img,bb,fn,False)
                cc=(int(bb[0]),int(bb[1]))
                cv2.circle(img, center=cc, radius=5, color=(0,0,255), thickness=-1)
            cv2.imwrite(f'syndet/res/res18/images/{fn}.png', img)
            
            
    time_end = time.time()
    fps = 1 / ((time_end-time_start)/len(test_loader.dataset)) 
    test_loss /= len(test_loader.dataset)
    err_x /= len(test_loader.dataset)
    err_y /= len(test_loader.dataset)
    err_ang /= len(test_loader.dataset)
    
    print('Test set: ')
    print('Picture Numbers: {}'.format(len(test_loader.dataset))) 
    print('Average loss: {:.10f}'.format(test_loss))
    print('Test ends in {:.2f} seconds'.format(time_end-time_start))
    print('Error of center: {:.5f} pt in x, {:.5f} pt in y'.format(err_x,err_y))
    print('Error of angle: {:.5f} degrees.'.format(err_ang))
    print('Rmse of center: {:.5f} in x, {:.5f} in y'.format(mse_x,mse_y))
    print('Rmse of angle: {:.5f}.'.format(mse_ang))
    print(f'FPS: {fps:.2f} frames per second')
    print()

def test_pcbmo(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    err_x, err_y, err_ang = 0, 0, 0
    mse_x, mse_y, mse_ang = 0, 0, 0
    w, h = args.image_size
    time_start = time.time()
    with torch.no_grad():
        for data, _target, fname in test_loader:
            
            _target = _target[0, :]  # 有六个变量。x,y,w,h,angle,class
            target = _target[:, [0, 1, 4]]
            bbb = _target[:, [0, 1, 2, 3, 4]]
            
            target = (target- torch.Tensor([w/2, h/2, 0]))/ torch.Tensor([w/2, h/2, 1])
            # print(target[:,-1],ang_norm(target[:, -1]))
            target[:,-1] = target[:,-1]/180*np.pi  # 修改标签文件的角度范围到-90～90-->-pi/2~pi/2
            
            data, target = data.to(device), target.to(device)
            
            
            output = model(data)
            
            # 计算error
            # err_cen += torch.sum(torch.sqrt(torch.pow((output[:, 0] - target[:, 0]) * w, 2) +
            #                                 torch.pow((output[:, 1] - target[:, 1]) * h, 2))).item()
            err_x += torch.sum(abs((output[:, 0] - target[:, 0]) * w)).item()
            err_y += torch.sum(abs((output[:, 1] - target[:, 1]) * w)).item()
            mse_x = F.mse_loss(output[:, 0],target[:, 0], reduction='sum').item()
            mse_y = F.mse_loss(output[:, 1],target[:, 1], reduction='sum').item()
            # torch.sum可以用于多目标计算
            err_ang1 = torch.sum(torch.abs(output[:,-1] - target[:,-1]) * 180 / np.pi )
            err_ang2 = torch.sum(torch.abs(ang_norm_rad(output[:,-1] + np.pi / 2) - target[:,-1]) * 180 / np.pi)
            err_ang3 = torch.sum(torch.abs(abs(output[:,-1]) - abs(target[:,-1])) * 180 / np.pi)
            delta_err_ang = min(err_ang1,err_ang2,err_ang3)
            # delta_err_ang = F.mse_loss(output[:, -1],target[:, -1], reduction='sum')
            err_ang += delta_err_ang.item()
            mse_ang = F.mse_loss(output[:,-1],target[:,-1], reduction='sum').item()
            # 在这里把结果画出来
            ''''''
            for img, bb, fn in zip(data.permute(0, 2, 3, 1).contiguous().cpu().numpy() * 255, 
                                   output.cpu().numpy(),
                                   fname):
                # bb = bb * np.float32([w/2,h/2,0,0,180 / np.pi])+np.float32([w/2,h/2,0,0,0])
                a = np.random.uniform(-1,1)*5
                b = np.random.uniform(-1,1)*5
                c = np.random.uniform(-1,1)*10
                
                bb = bbb +np.float32([a,b,0,0,c])
                bb = bb.numpy()
                plot_one_rotated_box(img,bb,fn,False)
                cc=(int(bb[0,0]),int(bb[0,1]))
                cv2.circle(img, center=cc, radius=5, color=(255,0, 0), thickness=-1)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            cv2.imwrite(f'syndet/res/res18/pcbmo/{fn}.png', img)
            
            
    time_end = time.time()
    fps = 1 / ((time_end-time_start)/len(test_loader.dataset)) 
    test_loss /= len(test_loader.dataset)
    err_x /= len(test_loader.dataset)
    err_y /= len(test_loader.dataset)
    err_ang /= len(test_loader.dataset)
    
    print('Test set: ')
    print('Picture Numbers: {}'.format(len(test_loader.dataset))) 
    print('Average loss: {:.10f}'.format(test_loss))
    print('Test ends in {:.2f} seconds'.format(time_end-time_start))
    print('Error of center: {:.5f} pt in x, {:.5f} pt in y'.format(err_x,err_y))
    print('Error of angle: {:.5f} degrees.'.format(err_ang))
    print('Rmse of center: {:.5f} in x, {:.5f} in y'.format(mse_x,mse_y))
    print('Rmse of angle: {:.5f}.'.format(mse_ang))
    print(f'FPS: {fps:.2f} frames per second')
    print()

def main():
    # Testing settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=True, help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--image-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=(128, 128), help='input image size (default: 224 224)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    
    if use_cuda:
        device = torch.device('cuda')
    elif use_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
        test_kwargs.update(cuda_kwargs)
    transform2 = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((args.image_size[1],args.image_size[0]), antialias=True),  # (h,w)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomCrop((args.image_size[1],args.image_size[0]), pad_if_needed=True, padding_mode='edge'),
    ])
    # 数据集更换之后记得查看和更改图片读取格式
    dataset2 = ImageWithObbDataset2('data/pcbmo/val3/images', 'data/pcbmo/val3/labels', transform=transform2)
    # dataset2 = ImageWithObbDataset('/data/ljx/pcb-electric-resistance-dataset/test2', '/data/ljx/pcb-electric-resistance-dataset/testlabel2', transform=transform2)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = Net()
    # flops, params = profile(model, inputs=(torch.randn(1,3,210,84),))
    # print(flops, params)
    model = model.to(device)
    

    # 加载预训练模型的权重
    weights_path = 'syndet/syndet_weight/syndet_res18_pcbmo_blur2.pth'  # 换模型要改 pcb:pcb1001
    model.load_state_dict(torch.load(weights_path))
    print(weights_path)
    test_pcbmo(args, model, device, test_loader)
    # test(args, model, device, test_loader)

if __name__ == '__main__':
    main()
    writer.close
