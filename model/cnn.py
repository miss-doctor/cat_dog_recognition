import torch.nn as nn

class simplecnn(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 池化后变为16*112*112
            nn.Conv2d(16,32,3,1,1), # 保持图像不变 32*112*112
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 变为32*56*56
        )
        #定义全连接层，做分类
        self.classifier = nn.Sequential(
            nn.Linear(32*56*56,128), # 只接受一维数据
            nn.ReLU(),
            nn.Linear(128,4) # num_class为分类的个数
        )

    def forward(self,x):
        # 前向传播部分
        x = self.features(x) # 先将图像进行特征提取
        x = x.view(x.size(0),-1) # 将图像展平为一维 为batch （32,32*56*56）
        x = self.classifier(x)
        return x
