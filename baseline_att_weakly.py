import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib
import os
import argparse

from torchsummary import summary
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
import sklearn.metrics as sm

from torchstat import stat


os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(description='PyTorch Har Training')
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

train_x = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/weakly_data_k/train_x.npy')
shape = train_x.shape
train_x = torch.from_numpy(np.reshape(train_x.astype(np.float), [shape[0], 1, shape[1], shape[2]]))
train_x = train_x.type(torch.FloatTensor).cuda()

train_y = (np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/weakly_data_k/train_y.npy'))
train_y = torch.from_numpy(train_y)
train_y = train_y.type(torch.FloatTensor).cuda()


test_x = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/weakly_data_k/test_x.npy')
test_x = torch.from_numpy(np.reshape(test_x.astype(np.float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
test_x = test_x.type(torch.FloatTensor)

test_y = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/weakly_data_k/test_y.npy')
test_y = torch.from_numpy(test_y.astype(np.float32))
test_y = test_y.type(torch.FloatTensor)


print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
trainset = Data.TensorDataset(train_x, train_y)
trainloader = Data.DataLoader(dataset=trainset, batch_size=200, shuffle=True, num_workers=0)

testset = Data.TensorDataset(test_x, test_y)
testloader = Data.DataLoader(dataset=testset, batch_size=713, shuffle=False, num_workers=0)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(57600, 4)
        )



    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        # print(x.shape)
        x = self.layer2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        # print(x.shape)
        x = self.layer3(x)
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x=nn.LayerNorm(x.size())(x.cpu())
        x=x.cuda()
        # x = F.normalize(x.cuda())
        # print(x.shape)
        return x

# def plot_confusion(comfusion,class_data):
#     plt.figure(figsize=(7,7))
#     plt.rcParams['font.family'] = ['Times New Roman']
#     classes = class_data
#     plt.imshow(comfusion, interpolation='nearest', cmap=plt.cm.Reds)  # 按照像素显示出矩阵
#     plt.title('confusion_matrix',fontsize = 10)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes,rotation=315)
#     plt.yticks(tick_marks, classes)
#     plt.tick_params(labelsize=10)
#     thresh = comfusion.max() / 2.
#     # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#     # ij配对，遍历矩阵迭代器
#     iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (comfusion.size, 2))
#     for i, j in iters:
#         plt.text(j, i, format(comfusion[i, j]),verticalalignment="center",horizontalalignment="center")  # 显示对应的数字
#
#     plt.ylabel('Real label',fontsize = 10)
#     plt.xlabel('Prediction',fontsize = 10)
#
#     plt.tight_layout()
#     plt.savefig('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/confusion_matrix/weakly_baseline/weakly_baseline.png')
#     # plt.show()


# Model
print('==> Building model..')

net = cnn()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net = torch.nn.DataParallel(net, device_ids=[0])


criterion = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(net.parameters(), lr=1e-3,weight_decay=1e-6)
# optimizer=torch.optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#
def flat(data):
    data=np.argmax(data,axis=1)
    return data

epoch_list=[]
error_list=[]

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    total=total
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs=inputs.type(torch.FloatTensor)
        inputs,targets=inputs.cuda(),targets
        outputs = net(inputs)
        # targets=torch.max(targets, 1)[1]
        # print(targets)
        loss = criterion(outputs,torch.max(targets, 1)[1].long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # predicted = torch.max(predicted, 1)[1].cuda()
        targets=torch.max(targets, 1)[1].cuda()
        predicted=predicted
        taccuracy = (torch.sum(predicted == targets.long()).type(torch.FloatTensor) / targets.size(0)).cuda()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.FloatTensor)
            inputs, targets = inputs.cuda(), targets
            outputs = net(inputs)
            # targets=torch.max(targets, 1)[1]
            # print(targets)
            loss = criterion(outputs, torch.max(targets, 1)[1].long())
            # scheduler.step()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # predicted = torch.max(predicted, 1)[1].cuda()
            targets = torch.max(targets, 1)[1].cuda()
            taccuracy = (torch.sum(predicted == targets.long()).type(torch.FloatTensor) / targets.size(0)).cuda()
            # print(type(predicted),type(targets),predicted,targets,'type(predicted),type(targets)')
            # correct += predicted.eq(targets).sum().item()
            test_error=1-taccuracy.item()
            print('test:', taccuracy.item(), '||', test_error)
            epoch_list.append(epoch)
            # accuracy_list.append(taccuracy.item())
            error_list.append(test_error)
        # confusion = sm.confusion_matrix(targets.cpu().numpy(), predicted.cpu().numpy())
        # print('The confusion matrix is：', confusion, sep='\n')
        # plot_confusion(confusion,
        #                    ['Going downstairs', 'Jogging', 'Jumping', 'Going upstairs'])
        #     np.save('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/weaklylableddata/epoch_att_new.npy',epoch_list)
        #     np.save('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/weaklylableddata/error_att_new.npy',error_list)



    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):

    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc


for epoch in range(start_epoch, start_epoch+500):
    train(epoch)
    test(epoch)

