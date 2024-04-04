"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/04/04 12:48:24
"""

#构建一个类线性模型类，继承自nn.Module,nn.m中封装了许多方法
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset    #这是一个抽象类，无法实例化
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

#实例化事件对象
writer = SummaryWriter('./log')
#一个针对数值的绘图方法 writer.add_scalar(tag, scalar_value, global_step=None, walltime=None) 
# 其中tag可视为图的名称，scalar_value可视为y值，global_step为x轴的值
# writer.add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
# writer.add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')


#构建一个compose类的实例用于图像预处理，训练集做仿射变换、随即旋转、转tensor（张量）、后面那个是标准化
train_transform = transforms.Compose([
                                transforms.RandomAffine(degrees = 0,translate=(0.1, 0.1)),
                                transforms.RandomRotation((-10,10)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])
test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])

#这2个值也是调参重灾区……
train_batch_size = 512
learning_rate = 0.03
train_epoch = 20
test_batch_size = 100
random_seed = 2         # 随机种子，设置后可以得到稳定的随机数
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) #为gpu提供随机数


train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=train_transform)

test_dataset = datasets.MNIST(root='./dataset/mnist/', train=False, download=True, transform=test_transform)
#将数据存储在cuda固定内存中，提高数据的存储速度
train_loader = DataLoader(dataset=train_dataset,batch_size=train_batch_size,shuffle=True,pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=test_batch_size,shuffle=False,pin_memory=True)


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

        return self.fc6(x)

#实例化模型
model = Net()

#调用GPU，cudnn加速
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True       #启用cudnn底层算法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(device)


model.to(device)

#构建损失函数
criterion = torch.nn.CrossEntropyLoss()      #交叉熵
#
#构建优化器,参数1：模型权重，参数二，learning rate
optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.5)    #带动量0.5

#optimizer = optim.RMSprop(model.parameters(),lr=learning_rate,alpha=0.99,momentum = 0.5)
#设置学习率梯度下降，如果连续三个epoch测试准确率没有上升，则降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# optimizer = torch.optim.Adam(model.parameters(),
#                 lr=0.001,
#                 betas=(0.9, 0.999),
#                 eps=1e-08,
#                 weight_decay=0,
#                 amsgrad=False)

#把训练封装成一个函数
def train(epoch):
    running_loss =0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()

        #forward,backward,update
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        # if batch_idx%300==299:
        train_loss_val.append((running_loss))
            #print('[%d,%5d] loss:%3f'%(epoch+1,batch_idx+1,running_loss/300))
        # writer.add_scalar('Loss',running_loss,int((epoch-1)*50000/train_batch_size)+batch_idx)
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
        running_loss = 0.0
    return running_loss / 300

#把测试封装成函数
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data,dim=1)       #从第一维度开始搜索
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    return correct/total

train_epoch= []
model_accuracy = []
temp_acc = 0.0
train_loss_val = []
now = datetime.datetime.now()
print(now.strftime('%Y/%m/%d %H:%M:%S'))
for epoch in range(20):
    train(epoch)
    acc = test()
    print(epoch,acc)
    train_epoch.append(epoch)
    model_accuracy.append(acc)
    # acc = model_accuracy.append(acc)
    # print(acc)
    
    writer.add_scalar('Accuracy',acc,epoch+1)
    now = datetime.datetime.now()
    print(now.strftime('%Y/%m/%d %H:%M:%S'),"Accuracy:%f "% (acc))

if torch.cuda.is_available():
    graph_inputs = torch.from_numpy(np.random.rand(1,1,28,28)).type(torch.FloatTensor).cuda()
else:
    graph_inputs = torch.from_numpy(np.random.rand(1,1,28,28)).type(torch.FloatTensor)
writer.add_graph(model, (graph_inputs,))

writer.close()
torch.cuda.empty_cache()        #释放显存

plt.plot(train_epoch, model_accuracy)  # 传入列表，plt类用来画图
plt.grid(linestyle=':')
plt.ylabel('accuracy')  # 定义y坐标轴的名字
plt.xlabel('epoch')  # 定义x坐标
plt.show()  # 显示
