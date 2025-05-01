import torch
from torch import nn
import torch.nn.functional as F


class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)  # 卷积层
        self.ReLU = nn.ReLU()  # 激活函数
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2)  # 最大池化层
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(2)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()  # 展平
        self.f6 = nn.Linear(4608, 2048)  # 全连接层
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        self.f9 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s3(x)
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = F.dropout(x, p=0.5, training=self.training)  # 只取一半有效，防止过拟合
        x = self.f7(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.f8(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.f9(x)
        return x


if __name__ == '__main__':  # 当模块被直接运行时，以下代码块将被运行，当模块是被导入时，代码块不被运行。
    x = torch.rand([1, 3, 224, 224])
    model = MyAlexNet()
    y = model(x)
    print(x)
    print(y)
