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
from torchstat import stat

from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
import sklearn.metrics as sm

os.environ['CUDA_VISIBLE_DEVICES']='1'

parser = argparse.ArgumentParser(description='PyTorch Har Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

train_x = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/pamap2_/train_x.npy')
shape = train_x.shape
train_x = torch.from_numpy(np.reshape(train_x.astype(np.float), [shape[0], 1, shape[1], shape[2]]))
train_x = train_x.type(torch.FloatTensor).cuda()

train_y = (np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/pamap2_/train_y_p.npy'))
train_y = torch.from_numpy(train_y)
train_y = train_y.type(torch.FloatTensor).cuda()


test_x = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/pamap2_/test_x.npy')
test_x = torch.from_numpy(np.reshape(test_x.astype(np.float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
test_x = test_x.type(torch.FloatTensor)

test_y = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/pamap2_/test_y_p.npy')
test_y = torch.from_numpy(test_y.astype(np.float32))
test_y = test_y.type(torch.FloatTensor)


print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
# torch.Size([5568, 1, 171, 40]) torch.Size([5568, 18])
# torch.Size([2048, 1, 171, 40]) torch.Size([2048, 18])
trainset = Data.TensorDataset(train_x, train_y)
trainloader = Data.DataLoader(dataset=trainset, batch_size=300, shuffle=True, num_workers=0)

testset = Data.TensorDataset(test_x, test_y)
testloader = Data.DataLoader(dataset=testset, batch_size=2048, shuffle=True, num_workers=0)

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
    def __init__(self, kernel_size=(7,1)):
        super(SpatialAttention, self).__init__()

        assert kernel_size in ((3,1), (7,1)), 'kernel size must be 3 or 7'
        padding = (3,0) if kernel_size == (7,1) else (1,0)

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def plot_confusion(comfusion,class_data):
    plt.figure(figsize=(12,9))
    plt.rcParams['font.family'] = ['Times New Roman']
    classes = class_data
    plt.imshow(comfusion, interpolation='nearest', cmap=plt.cm.Reds)  # 按照像素显示出矩阵
    plt.title('confusion_matrix',fontsize = 12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=315)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=12)
    thresh = comfusion.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (comfusion.size, 2))
    for i, j in iters:
        plt.text(j, i, format(comfusion[i, j]),verticalalignment="center",horizontalalignment="center")  # 显示对应的数字

    plt.ylabel('Real label',fontsize = 12)
    plt.xlabel('Predicted label',fontsize = 12)

    plt.tight_layout()
    # plt.savefig('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/confusion_matrix/pamap2_resnet_cbam/pamap2_resnet_cbam.png')
    plt.show()


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )
        self.ca1 = ChannelAttention(128)
        self.sa1 = SpatialAttention()

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
        )
        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
        )
        self.ca3 = ChannelAttention(384)
        self.sa3 = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(76800, 18)
        )

    def forward(self, x):
        # print(x.shape)
        h1 = self.Block1(x)
        # print(h1.shape)
        r = self.shortcut1(x)
        # print(r.shape)
        h1 = self.ca1(h1) * h1
        h1=  self.sa1(h1) * h1
        h1 = h1 + r
        # print(h1.shape)
        h2 = self.Block2(h1)
        # print(h2.shape)
        r = self.shortcut2(h1)
        # print(r.shape)
        h2 = self.ca2(h2) * h2
        h2 = self.sa2(h2) * h2
        h2 = h2 + r
        # print(h2.shape)
        h3 = self.Block3(h2)
        # print(h3.shape)
        r = self.shortcut3(h2)
        # print(r.shape)
        h3 = self.ca3(h3) * h3
        h3 = self.sa3(h3) * h3
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        return x


# Model
print('==> Building model..')

net = resnet()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=5e-4,weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
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
        # print(type(predicted),type(targets),predicted,targets,'type(predicted),type(targets)')
        # correct += predicted.eq(targets).sum().item()
        train_error = 1 - taccuracy.item()

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
            scheduler.step()
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
            # print(epoch_list)
            # accuracy_list.append(taccuracy.item())
            error_list.append(test_error)
            confusion = sm.confusion_matrix(targets.cpu().numpy(), predicted.cpu().numpy())
            print('The confusion matrix is：', confusion, sep='\n')
            plot_confusion(confusion,
                           ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Nordic walking',
                            'Ascending stairs', 'Descending stairs', 'Vacuum cleaning', 'Ironing', 'Rope jumping'])
            # print(error_list)
            # np.save('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/pamap2/epoch_resnet_att_1.npy',epoch_list)
            # np.save('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/pamap2/error_resnet_att_1.npy',error_list)

for epoch in range(start_epoch, start_epoch+500):
    train(epoch)
    test(epoch)

model=resnet()
stat(model,(1,171,40))
print(model)
