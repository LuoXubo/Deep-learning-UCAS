"""
@Description :   CNN网络模型
@Author      :   Xubo Luo 
@Time        :   2024/04/04 20:44:54
"""
import torch.nn.functional as F
import torch.nn as nn

class ResidualBlock(nn.Module):
    # Residual Block需要保证输出和输入通道数x一样
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        # 3*3卷积核，为保证图像大小不变将padding设为1
        # 第一个卷积
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        # 第二个卷积
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # 激活
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        # 先求和 后激活
        z = self.conv3(x)
        return F.relu(z + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 192, kernel_size=5, padding=2)

        # 残差神经网络层，其中已经包含了relu
        self.rblock1 = ResidualBlock(32)
        self.rblock2 = ResidualBlock(64)
        self.rblock3 = ResidualBlock(128)
        self.rblock4 = ResidualBlock(192)

        # BN层，归一化，使数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(192)

        # 池化层
        self.mp = nn.MaxPool2d(2)

        # fully connectected全连接层
        self.fc1 = nn.Linear(192 * 7 * 7, 256)  # 线性
        self.fc6 = nn.Linear(256, 10)  # 线性

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
        in_size = x.size(0)

        x = self.conv1(x)  # channels:1-32	w*h:28*28

        x = F.relu(x)
        x = self.bn1(x)
        x = self.rblock1(x)

        x = self.conv2(x)  # channels:32-64	w*h:28*28
        x = F.relu(x)
        x = self.bn2(x)
        x = self.rblock2(x)

        x = self.mp(x)  # 最大池化,channels:64-64	w*h:28*28->14*14
        x = self.drop1(x)

        x = self.conv3(x)  # channels:64-128	w*h:14*14

        x = F.relu(x)
        x = self.bn3(x)
        x = self.rblock3(x)
        # x = self.mp(x)		#channels:128-128	w*h:7*7
        #x = self.drop1(x)

        x = self.conv4(x)  # channels:128-192	w*h:14*14

        x = F.relu(x)
        x = self.bn4(x)
        x = self.rblock4(x)
        x = self.mp(x)  # 最大池化,channels:192-192	w*h:14*14->7*7

        x = x.view(in_size, -1)  # 展开成向量
        x = F.relu(self.fc1(x))  # 使用relu函数来激活
        #x = self.drop2(x)
        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)
        return x

class Vanilla_Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)