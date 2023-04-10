import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Dropout import Dropout
from ALDropout import AlphaDropout
from GSDropout import GaussianDropout
import cv2

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,
                                           shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50,
                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 实现单张图片可视化

batch_size = 200


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Dropout model on a image classification task")
    parser.add_argument("--method", type=str, default=None, help="The method of Dropout.")
    parser.add_argument("--epoch", type=int, default=None, help="The train epochs.")

    args = parser.parse_args()

    return args


# 卷积层使用 torch.nn.Conv2d
# 激活层使用 torch.nn.ReLU
# 池化层使用 torch.nn.MaxPool2d
# 全连接层使用 torch.nn.Linear

class LeNet(nn.Module):
    def __init__(self, method):
        super(LeNet, self).__init__()
        if method == "Dropout":
            self.dropout = Dropout(0.5)
        elif method == "GaussianDropout":
            self.dropout = GaussianDropout(0.5)
        else:
            self.dropout = AlphaDropout(0.5)
        self.fc1 = nn.Sequential(nn.Linear(3072, 512), nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())

        self.fc3 = nn.Sequential(nn.Linear(256, 10), nn.Softmax(dim=1))

    # 最后的结果一定要变为 10，因为数字的选项是 0 ~ 9

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.fc1(x)
        x = self.dropout.forward(x, train="train")
        x = self.fc2(x)
        x = self.dropout.forward(x, train="train")
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR = 0.001

    net = LeNet(args.method).to(device)
    # 损失函数使用交叉熵
    criterion = nn.CrossEntropyLoss()
    # 优化函数使用 Adam 自适应优化算法
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR,
    )

    epoch = args.epoch

    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()  # 将梯度归零
            outputs = net(inputs)  # 将数据传入网络进行前向运算
            loss = criterion(outputs, labels)  # 得到损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 通过梯度做一步参数更新

            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    net.eval()  # 将模型变换为测试模式
    correct = 0
    total = 0
    for data_test in test_loader:
        images, labels = data_test
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        output_test = net(images)
        _, predicted = torch.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("correct1: ", correct)
    print("Test acc: {0}".format(correct.item() /
                                 len(test_dataset)))
