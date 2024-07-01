from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from dataset import ImageWithObbDataset
import transforms_with_label
import torchvision.models as models
import cv2
import numpy as np
from shapely.geometry import box
from shapely import affinity
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
import math
# Import the summary writer 
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object 
writer = SummaryWriter('logs')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.backbone = models.resnet18(pretrained=True)
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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    use_snap_loss = False
        
    for batch_idx, (data, target, fname) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        # 在这里产生180度旋转和随机平移的数据
        data_rot = transforms.functional.rotate(data,180)
        sft_ratio = 0.2
        # sx = np.random.uniform(-1,1)*sft_ratio*128
        # sy = np.random.uniform(-1,1)*sft_ratio*128
        sx = np.random.uniform(-1,1)*sft_ratio*64
        sy = np.random.uniform(-1,1)*sft_ratio*64
        
        # 在这里产生随机平移加边缘填充的数据：affine+edge padding
        # data_sft = transforms.functional.affine(data,0,(sx,sy),1,(0,0)) # 原本的随机平移操作，只填充纯色（默认黑色）
        edge_color = (torch.mean(data[0][:, :1, :])+torch.mean(data[0][:, -1:, :])+torch.mean(data[0][:, :, :1])+torch.mean(data[0][:, :, -1:]))/4
        data_sft =transforms.functional.affine(data,0,(sx,sy),1,(0,0),fill=edge_color)
        
        # 在这里产生h2rboxv2的另外两个view
        # 垂直翻转视角
        data_flp = transforms.functional.vflip(data)

        # 随机旋转视角
        theta_rot = np.random.uniform(-1, 1) * 90
        if theta_rot >= 0:
            theta_rot += 45
        if theta_rot < 0:
            theta_rot -= 45
        # print('随机旋转角度:',theta_rot/math.pi*180)
        data_rot_v2 = transforms.functional.rotate(data,theta_rot,fill=edge_color)  
        # output, output_rot, output_sft = model(torch.cat((data, data_rot, data_sft), 0)).view(3, -1, 5)
        output, output_rot, output_sft, output_flp, output_rot_v2 = model(torch.cat((data, data_rot, data_sft, data_flp, data_rot_v2), 0)).view(5, -1, 5)
        
        # 输出是-1到1之间，cv原点是应该是（-1，-1），图像中点是（0,0）
        # 在这里计算输出Loss
        type_loss = nn.SmoothL1Loss()
        loss_rot = type_loss(output[:,0:2]+output_rot[:,0:2], torch.zeros_like(output[:,0:2]+output_rot[:,0:2]))# 旋转一致性损失        
        # 把loss_sft和sx,sy关联起来，平移后图像输出和原始输出应该相距平移的距离
        loss_sft = type_loss(output_sft[:,0:2]-output[:,0:2]-output.new_tensor([sx/64, sy/64]), torch.zeros_like(output_sft[:,0:2]-output[:,0:2]-output.new_tensor([sx/64,sy/64])))# 平移一致性损失
        # 把loss_rot_v2和theta_rot关联起来，平移后图像输出和原始输出应该相差theta_rot角度（网络输出是-pi/2～pi/2）
        d_ang_rot = (output[:,4:5] - output_rot_v2[:,4:5] - theta_rot / 180 * torch.pi + torch.pi / 2) % (torch.pi / 1) - torch.pi / 2
        d_ang_flp = (output[:,4:5] + output_flp[:,4:5]+ torch.pi / 2) % (torch.pi / 1) - torch.pi / 2
        
        loss_flp = type_loss(d_ang_flp, torch.zeros_like(d_ang_flp))
        loss_rot_v2 = type_loss(d_ang_rot, torch.zeros_like(d_ang_rot))

        a = np.float32(0.05)
        loss_cen = a*loss_rot + loss_sft
        loss_ang = a*loss_flp + loss_rot_v2
        loss = loss_cen + loss_ang
        train_loss += data.shape[0] * loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tLoss Rot: {:.4f}\tLoss Sft: {:.4f}\tLoss Flp: {:.4f}\tLoss Rot V2: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), loss_rot.item(), loss_sft.item(), loss_flp.item(), loss_rot_v2.item()))
            if args.dry_run:
                break

    train_loss /= len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    writer.add_scalar('loss',train_loss,epoch)
    
def plot_one_rotated_box(img, obb, color=[0, 0, 255], label=None, line_thickness=None):
    width, height, theta = 60, 30, obb[4]
    if theta < 0:
        width, height, theta = 60, 30, theta + 90
    rect = [(obb[0], obb[1]), (width, height), theta]
    poly = np.intp(np.round(cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=color, thickness=2 * tl)
    # c1 = (int(obb[0]), int(obb[1]))
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
    #     textcolor = [0, 0, 0] if max(color) > 192 else [255, 255, 255]
    #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, textcolor, thickness=tf, lineType=cv2.LINE_AA)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_iou = 0
    err_cen, err_siz, err_ang = 0, 0, 0
    with torch.no_grad():
        for data, target, fname in test_loader:
            data = data.to(device)
            output = model(data)
            # 在这里把结果画出来
            for img, bb, fn in zip(data.permute(0, 2, 3, 1).contiguous().cpu().numpy() * 255, 
                                   output.cpu().numpy(),
                                   fname):
                bb = bb * np.float32([64,64,0,0,180 / np.pi])+np.float32([64,64,0,0,0])
                plot_one_rotated_box(img,bb,line_thickness=2)
                cc=(int(bb[0]),int(bb[1]))
                cv2.circle(img, center=cc, radius=5, color=(0,0,255), thickness=-1)
                # angleText = '%.2f' % bb[-1]+'degree'
                # cv2.putText(img,angleText,(int(bb[0]+44),int(bb[1]+54)),0,0.3,[0, 0, 0],1,cv2.LINE_AA)
            cv2.imwrite(f'syndet/res/{fn}.png', img)
    print()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=True, help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

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
    transform1 = transforms_with_label.ComposeWithLabel([
        transforms.ConvertImageDtype(torch.float),
        transforms_with_label.RandomHorizontalFlipWithLabel(0.5),
        transforms_with_label.RandomVerticalFlipWithLabel(0.5),
        transforms_with_label.RandomScaleWithLabel((0.9, 1.1)),
        # transforms_with_label.RandomRotationWithLabel(90),
        transforms_with_label.RandomCropWithLabel((128, 128), pad_if_needed=True, padding_mode='edge'),
        # transforms_with_label.RandomCropWithLabel((1024, 1024), pad_if_needed=True),
    ])
    transform2 = transforms_with_label.ComposeWithLabel([
        transforms.ConvertImageDtype(torch.float),
        transforms_with_label.RandomCropWithLabel((128, 128), pad_if_needed=True, padding_mode='edge'),
        # transforms_with_label.RandomCropWithLabel((1024, 1024), pad_if_needed=True),
    ])
    dataset1 = ImageWithObbDataset('syndet/smtdata/images', 'syndet/smtdata/labels', transform=transform1)
    dataset2 = ImageWithObbDataset('syndet/smtdata/images', 'syndet/smtdata/labels', transform=transform2)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[120, 180, 220], gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        if epoch % 100 == 0:
            test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), 'syndet/syndet_smt.pt')
        


if __name__ == '__main__':
    main()