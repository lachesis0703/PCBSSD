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
import time
import torchvision.models as models
import numpy as np
from shapely.geometry import box
from shapely import affinity
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
import math
import cv2
# Import the summary writer 
from torch.utils.tensorboard import SummaryWriter # Create an instance of the object 
writer = SummaryWriter('syndet/res/VTMablation')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.backbone = models.resnet18(pretrained=False)    # 可以试试更改权重
        self.conv = nn.Conv2d(512, 7, 1, 1)

        self.num_step = 3
        self.coef_sin = torch.tensor(tuple(torch.sin(torch.tensor(2 * k * torch.pi / self.num_step)) for k in range(self.num_step)))
        self.coef_cos = torch.tensor(tuple(torch.cos(torch.tensor(2 * k * torch.pi / self.num_step)) for k in range(self.num_step)))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        '''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        '''
        x = self.conv(x).mean(-1).mean(-1)
        output = torch.sigmoid(x) * 2 - 1

        # PSC
        angle_preds = output[:, 4:7]
        self.coef_sin = self.coef_sin.to(angle_preds)
        self.coef_cos = self.coef_cos.to(angle_preds)

        # if len(angle_preds):
        #     print('>', angle_preds.min(), angle_preds.max())

        phase_sin = torch.sum(angle_preds[:, 0:self.num_step] * self.coef_sin, dim=-1, keepdim=True)
        phase_cos = torch.sum(angle_preds[:, 0:self.num_step] * self.coef_cos, dim=-1, keepdim=True)
        # phase_mod = phase_cos**2 + phase_sin**2
        phase = -torch.atan2(phase_sin, phase_cos)  # In range [-pi,pi)
        return torch.cat((output[:, 0:4], phase / 2), dim=-1)

def plot_one_rotated_box(img, obb, filename, color=[0, 255, 0]):
    line_thickness = 2
    width, height, theta = 35, 55, obb[-1]
    # if theta < 0:
    #     width, height, theta = 80, 40, theta + 90
    rect = [(obb[0], obb[1]), (width, height), theta]
    # single_evaluate('/data/ljx/pcb-electric-resistance-dataset/testlabel',rect,filename)
    poly = np.intp(np.round(cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=[0, 255, 0], thickness=2 * tl)

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
    
    theta_gt = ang_norm(theta_gt)
    theta_pred_90 = ang_norm(90 + theta_pred)
    ang_err0 = np.absolute(theta_gt-theta_pred)
    ang_err1 = np.absolute(theta_pred_90-theta_gt)
    ang_err2 = np.absolute(theta_pred_90+theta_gt)
    ang_err = min(ang_err0, ang_err1, ang_err2)
    print('Picture name is:', filename, '\n','Groundtruth: ', (x_gt, y_gt), theta_gt, '\n', 'Prediction: ', (x_pred, y_pred), theta_pred)
    print('Err: ', pos_err, ang_err)

def ang_norm(ang):
    while ang > 90:
        if ang < 270:
            ang = 180 - ang
        if ang >= 270:
            ang = ang - 360

    return ang 

def ang_norm_rad(ang):
    while ang > np.pi/2:
        if ang < 3*np.pi/2:
            ang = np.pi - ang
        if ang >= 3*np.pi/2:
            ang = ang - np.pi*2 

    return ang

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    time_start = time.time()
        
    for batch_idx, (data, fname) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # 在这里产生180度旋转和随机平移的数据
        data_rot = transforms.functional.rotate(data,180)
        sft_ratio = 0.2   # 0.2是合理的
        sx = np.random.uniform(-1,1)*sft_ratio*(args.image_size[0]/2)
        sy = np.random.uniform(-1,1)*sft_ratio*(args.image_size[1]/2)
        # 在这里产生随机平移加边缘填充的数据：affine+edge padding
        data_sft =transforms.functional.affine(data,0,(sx,sy),1,(0,0),fill=0) 
        # 修改了transforms.functional_tensor的_apply_grid_transform中(/usr/local/anaconda3/envs/syndet/lib/python3.9/site-packages/torchvision/transforms/_functional_tensor.py)的padding_mode(如果你不小心更新了环境文件中的torch，记得看看这边改了没有)

        # 在这里产生h2rboxv2的另外两个view
        # 垂直翻转视角
        data_flp = transforms.functional.vflip(data)

        # 随机旋转视角
        rot_ratio = args.rot
        theta_rot = np.random.uniform(-1, 1) * rot_ratio * 90
        data_rot_v2 = transforms.functional.rotate(data,theta_rot,fill=0)
        # output, output_rot, output_sft = model(torch.cat((data, data_rot, data_sft), 0)).view(3, -1, 5)
        output, output_rot, output_sft, output_flp, output_rot_v2 = model(torch.cat((data, data_rot, data_sft, data_flp, data_rot_v2), 0)).view(5, -1, 5)
        
        # 输出是-1到1之间，cv原点是应该是（-1,-1），图像中点是（0,0）
        # 在这里计算输出Loss
        type_loss = nn.SmoothL1Loss() # loss函数修改
        loss_rot = type_loss(output[:,0:2]+output_rot[:,0:2], torch.zeros_like(output[:,0:2]+output_rot[:,0:2]))# 旋转一致性损失        
        # 把loss_sft和sx,sy关联起来，平移后图像输出和原始输出应该相距平移的距离
        loss_sft = type_loss(output_sft[:,0:2]-output[:,0:2]-output.new_tensor([sx/(args.image_size[0]/2), sy/(args.image_size[1]/2)]), torch.zeros_like(output_sft[:,0:2]-output[:,0:2]-output.new_tensor([sx/(args.image_size[0]/2),sy/(args.image_size[1]/2)])))# 平移一致性损失
        # 把loss_rot_v2和theta_rot关联起来，平移后图像输出和原始输出应该相差theta_rot角度（网络输出是-pi/2～pi/2）
        d_ang_rot = (output[:,4:5] - output_rot_v2[:,4:5] - theta_rot / 180 * torch.pi + torch.pi / 2) % (torch.pi / 1) - torch.pi / 2
        d_ang_flp = (output[:,4:5] + output_flp[:,4:5]+ torch.pi / 2) % (torch.pi / 1) - torch.pi / 2
        
        loss_flp = type_loss(d_ang_flp, torch.zeros_like(d_ang_flp))
        loss_rot_v2 = type_loss(d_ang_rot, torch.zeros_like(d_ang_rot))

        a = np.float32(0.05)
        loss_cen = a*loss_rot + loss_sft
        # loss_cen = a*loss_rot
        # loss_ang = a*loss_flp + loss_rot_v2
        loss_ang = a*loss_flp
        
        loss =  loss_cen + loss_ang
        train_loss += data.shape[0] * loss
        loss.backward()
        optimizer.step()
        

    train_loss /= len(train_loader.dataset)
    time_end = time.time()
    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTime: {:.2f}s\tLoss: {:.8f}\tLoss Rot: {:.8f}\tLoss Sft: {:.8f}\tLoss Flp: {:.8f}\tLoss Rot V2: {:.8f}'.format(
                # epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), time_end-time_start,
                # loss.item(), loss_rot.item(), loss_sft.item(), loss_flp.item(), loss_rot_v2.item()))
    # print('Train set: Average loss: {:.4f}'.format(train_loss))
    writer.add_scalar('loss',train_loss,epoch)
    writer.add_scalar('loss_rot',loss_rot,epoch)
    writer.add_scalar('loss_sft',loss_sft,epoch)
    writer.add_scalar('loss_flp',loss_flp,epoch)
    writer.add_scalar('loss_rot_v2',loss_rot_v2,epoch)

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
            sft_x_ratio = 0.2   # 0.2
            sft_y_ratio = 0.2   # 0.2
            sx = np.random.uniform(-1,1)*sft_x_ratio*(args.image_size[0]/2)
            sy = np.random.uniform(-1,1)*sft_y_ratio*(args.image_size[1]/2)
            data_sft =transforms.functional.affine(data,0,(sx,sy),1,(0,0),fill=0)
            data_flp = transforms.functional.vflip(data)
            rot_ratio = 1
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
            target = (target- torch.Tensor([w/2, h/2, 0]))/ torch.Tensor([w/2, h/2, 1])
            # print(target[:,-1],ang_norm(target[:, -1]))
            target[:,-1] = target[:,-1]/180*np.pi  # 修改标签文件的角度范围到-90～90-->-pi/2~pi/2
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
            # err_cen += torch.sum(torch.sqrt(torch.pow((output[:, 0] - target[:, 0]) * w, 2) +
            #                                 torch.pow((output[:, 1] - target[:, 1]) * h, 2))).item()
            err_x += torch.sum(abs((output[:, 0] - target[:, 0]) * w)).item()
            err_y += torch.sum(abs((output[:, 1] - target[:, 1]) * w)).item()
            mse_x = F.mse_loss(output[:, 0],target[:, 0], reduction='sum').item()
            mse_y = F.mse_loss(output[:, 1],target[:, 1], reduction='sum').item()
            # torch.sum可以用于多目标计算
            err_ang1 = torch.sum(torch.abs(output[:,-1] - target[:,-1]) * 180 / np.pi )
            err_ang2 = torch.sum(ang_norm_rad(output[:,-1] - target[:,-1]) * 180 / np.pi)
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
                cv2.circle(img, center=cc, radius=5, color=(0, 0, 255), thickness=-1)
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
    print('Mse of center: {:.5f} in x, {:.5f} in y'.format(mse_x,mse_y))
    print('Mse of angle: {:.5f}.'.format(mse_ang))
    print(f'FPS: {fps:.2f} frames per second')
    print()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=True, help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--image-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=(128, 128), help='input image size (default: 224 224)')
    parser.add_argument('--rot', type=float, default=0.9, metavar='M', help='rot ratio (default: 0.9)')
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

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    transform1 = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((args.image_size[1],args.image_size[0]), antialias=True),  # (h,w)
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(90),
        transforms.RandomCrop((args.image_size[1],args.image_size[0]), pad_if_needed=True, padding_mode='edge'),    # (h,w)
    ])
    transform2 = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((args.image_size[1],args.image_size[0]), antialias=True),  # (h,w) 
        transforms.RandomCrop((args.image_size[1],args.image_size[0]), pad_if_needed=True, padding_mode='edge'),
    ])
    print('load dataset')
    # dataset1 = ImageNoLabDataset('/data/ljx/pcbnet/dataset/train/processedimages', transform=transform1) 
    dataset1 = ImageNoLabDataset('data/pcbmo/origintrain/BlurImages', transform=transform1) # syndet/smtdata/images  
    # 数据集更换之后记得查看和更改图片读取格式(尤其是255- ！！)
    dataset2 = ImageWithObbDataset2('data/pcbmo/originval/BlurImages', 'data/pcbmo/originval/labels', transform=transform2)
    # dataset2 = ImageWithObbDataset('/data/ljx/pcbnet/dataset/val/processedimages', '/data/ljx/pcbnet/dataset/val/labels', transform=transform2)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    print('dataset loaded')

    model = Net()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    scheduler = MultiStepLR(optimizer, milestones=[120, 180, 220], gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # if epoch == 1:
        #     test_pcbmo(args, model, device, test_loader)
        if epoch % 300 == 0:
            test_pcbmo(args, model, device, test_loader)
        scheduler.step()
    # test(args, model, device, test_loader)
    
    if args.save_model:
        weight_save_path='syndet/syndet_weight/syndet_res18_pcbmo_e7.pth'
        torch.save(model.state_dict(), weight_save_path)
        print('Weight has been saved to: ' + weight_save_path)
        
        


if __name__ == '__main__':
    main()
    writer.close()