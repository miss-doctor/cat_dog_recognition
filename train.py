import torch
import torch.nn as nn
import torch.optim as optim # 优化器
from tensorflow.python.keras.backend import learning_phase
from torch.cpu import is_available
from torch.utils.data import DataLoader # 数据加载其
from torchvision import datasets,transforms # 数据集 及 数据变化
from tqdm import tqdm # 训练进度条
import os
from model.cnn import simplecnn

devic = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#对图像做变换
train_transformer = transforms.Compose([
    transforms.Resize([224,224]), # 将数据裁剪为224*224
    transforms.ToTensor(), # 图片转换为tensor张量 0-1的像素值
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 标准化 （-1，1）
])

test_transformer = transforms.Compose([
    transforms.Resize([224,224]), # 将数据裁剪为224*224
    transforms.ToTensor(), # 图片转换为tensor张量 0-1的像素值
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 标准化 （-1，1）
])

# 加载测试集和训练集
trainset = datasets.ImageFolder(root = os.path.join(r"E:\cat_dog_recognition\cats_and_dogs","train"),
                                transform=train_transformer)

testset = datasets.ImageFolder(root = os.path.join(r"E:\cat_dog_recognition\cats_and_dogs","test"),
                                transform=test_transformer)

# 定义训练集加载器
train_loader = DataLoader(trainset,batch_size=32,num_workers=0,shuffle=True) # 32批次，不打开数据加载多线程，打乱加载
test_loader = DataLoader(trainset,batch_size=32,num_workers=0,shuffle=False) # 32批次，不打开数据加载多线程，顺寻加载

def train(model,train_loader,criterion,optimizer,num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs,labels in tqdm(train_loader,desc=f"epoch:{epoch+1}/{num_epochs}",unit="batch"):
            input,labels = inputs.to(devic),labels.to(devic) # 将数据传到设备上
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            running_loss += loss.item() * inputs.size(0) # 用loss乘批次大小，得到该批次的loss
        epoch_loss = running_loss/len(train_loader.dataset)
        print(f"epoch{epoch+1}/{num_epochs},Train_loss:{epoch_loss:.4f}")

        accuracy = evaluate(model,test_loader,criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model,save_path)
            print("model saved with best acc",best_acc)

def evaluate(model,test_loader,criterion):
    model.eval() # 指定模型为验证模型
    test_loss = 0.0
    correct = 0 # 正确的样本数量
    total = 0 # 总共样本数量
    with torch.no_grad(): # 在评估模式下不需计算梯度
        for inputs,labels in test_loader:
            input,labels = inputs.to(devic),labels.to(devic) # 将数据传到设备上
            outputs = model(inputs)
            loss =  criterion(outputs,labels)
            test_loss += loss.item() * inputs.size(0) # 用loss乘批次大小，得到该批次的loss
            _,predicted = torch.max(outputs,1) # 获取模型预测的最大值
            total = total + labels.size(0) # 计算总样本的数量
            correct = correct + (predicted == labels).sum().item() # 正确样本数累加

    avg_loss = test_loss / len(test_loader.dataset)# 计算平均loss
    accuracy = 100.0*correct / total # 计算准确率
    print(f"Test Loss:{avg_loss:.4f},Accuracy:{accuracy:.2f}%")
    return accuracy

def save_model(model,save_path):
    torch.save(model.state_dict(),save_path)

if __name__ == "__main__":
    num_epochs = 10
    learning_rate = 0.001
    num_class = 4
    save_path = r"model_pth\best.pth"
    model = simplecnn(num_class).to(devic) # 对模型进行实例化并送入gpu或cpu中
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(),lr=learning_rate) # 指定优化器为Adam
    train(model,train_loader,criterion, optimizer, num_epochs)
    evaluate(model, test_loader, criterion) # 使用测试集进行验证