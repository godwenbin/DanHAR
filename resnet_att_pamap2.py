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
import random
import matplotlib.pyplot as plt
import matplotlib
import os
import argparse
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
def flat(data):
    data=np.argmax(data,axis=1)
    return data

# Data
print('==> Preparing data..')
random.seed(0)
torch.manual_seed(0)
cudnn.deterministic = True
train_x = np.load('/home/tengqi/桌面/Imblanced_Datase_With_LDAM_Loss/preprocessed/X_train.npy')
shape = train_x.shape
train_x = torch.from_numpy(np.reshape(train_x.astype(np.float), [shape[0], 1, shape[1], shape[2]]))
train_x = train_x.type(torch.FloatTensor).cuda()

train_y = (np.load('/home/tengqi/桌面/Imblanced_Datase_With_LDAM_Loss/preprocessed/y_train.npy'))
train_y = torch.from_numpy(train_y)
train_y = train_y.type(torch.FloatTensor).cuda()


test_x = np.load('/home/tengqi/桌面/Imblanced_Datase_With_LDAM_Loss/preprocessed/X_test.npy')
test_x = torch.from_numpy(np.reshape(test_x.astype(np.float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
test_x = test_x.type(torch.FloatTensor)

test_y = np.load('/home/tengqi/桌面/Imblanced_Datase_With_LDAM_Loss/preprocessed/y_test.npy')
test_y = torch.from_numpy(test_y.astype(np.float32))
test_y = test_y.type(torch.FloatTensor)


print(train_x.shape, train_y)
print(test_x.shape, test_y.shape)
trainset = Data.TensorDataset(train_x, train_y)
trainloader = Data.DataLoader(dataset=trainset, batch_size=128, shuffle=False, num_workers=0)

testset = Data.TensorDataset(test_x, test_y)
testloader = Data.DataLoader(dataset=testset, batch_size=2947, shuffle=True, num_workers=0)

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



class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
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
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
        )
        self.ca3 = ChannelAttention(384)
        self.sa3 = SpatialAttention()


        self.fc = nn.Sequential(
            nn.Linear(141696, 12)
        )
        #block3:31104   block4:13824




    def forward(self, x,tatget):
        # print(x.shape)
        h1 = self.Block1(x)
        # print('block_h1:',h1.shape)
        r = self.shortcut1(x)
        # print('shortcut1_r:',r.shape)
        h1 = self.ca1(h1) * h1
        plot_h1=h1
        # print('ca1_h1:',tatget.cpu())
        h1=  self.sa1(h1) * h1
        # print('sa1_h1:',h1)
        h1 = h1 + r
        # print('h1+r:',h1)
        h2 = self.Block2(h1)
        # print(h2.shape)
        r = self.shortcut2(h1)
        # print(r.shape)
        h2 = self.ca2(h2) * h2
        plot_h2 = h2
        # print(h2.shape,'iiiiiiiiii')
        h2 = self.sa2(h2) * h2
        h2 = h2 + r
        # print(h2.shape)
        h3 = self.Block3(h2)
        # print(h3.shape)
        r = self.shortcut3(h2)
        # print(r.shape)
        h3 = self.ca3(h3) * h3
        plot_h3=h3
        # print(h3.shape, 'qqqqqqqqqqq')
        h3 = self.sa3(h3) * h3
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        return x,plot_h1,plot_h2,plot_h3


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
optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#
def flat(data):
    data=np.argmax(data,axis=1)
    return data

epoch_list=[]
error_list=[]
def get_shape(data):
    return data.shape
# Training
def plot_attention(y_data,plot_att,special_epoch,special_features,batch_idx,is_save):
    # print(batch_idx, y[0], y, 'yyyyyyyyyyyyyyyyyyyy')
    # train_y = flat(targets)
    if batch_idx == 2:
        plot_att_shape = get_shape(plot_att)
        plot_att = plot_att[special_epoch, special_features, :, :].transpose(1, 0)

        print(np.array(y_data.cpu()),np.array(y_data.cpu())[special_features],plot_att_shape)
        if np.array(y_data.cpu())[special_features]==0:
            title = 'laying'
        elif np.array(y_data.cpu())[special_features]==1:
            title = 'sitting'
        elif np.array(y_data.cpu())[special_features]==2:
            title = 'standing'
        elif np.array(y_data.cpu())[special_features]==3:
            title = 'walking'
        elif np.array(y_data.cpu())[special_features]==4:
            title = 'running'
        elif np.array(y_data.cpu())[special_features]==5:
            title = 'cycling'
        elif np.array(y_data.cpu())[special_features]==6:
            title = 'Nordic walking'
        elif np.array(y_data.cpu())[special_features]==7:
            title = 'ascending stairs'
        elif np.array(y_data.cpu())[special_features]==8:
            title = 'descending stairs'
        elif np.array(y_data.cpu())[special_features]==9:
            title = 'vacuum cleaning'
        elif np.array(y_data.cpu())[special_features]==10:
            title = 'ironing'
        elif np.array(y_data.cpu())[special_features]==11:
            title = 'rope jumping'

        fig=plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        plt.title(title,fontsize=16)
        plt.imshow(plot_att.cpu().detach().numpy() ,cmap=plt.cm.Reds,origin='lower')
        ax.set_aspect(5)
        # plt.colorbar(fraction=0.006, pad=0.07)
        cb=plt.colorbar()
        # cb.set_label('attention level ')
        plt.xlabel(r"time step", fontsize=16)
        # plt.ylabel(r'class', fontsize=8)
        classes = ['hand_x', 'hand_y', 'hand_z', 'ankle_x', 'ankle_y', 'ankle_z', 'chest_x', 'chest_y', 'chest_z']

        batch_x = np.arange(0, plot_att_shape[2]-1, 8)
        batch_y = np.arange(0, 9, 1)
        tick_marks = np.arange(len(classes))
        plt.tick_params(labelsize=14)
        plt.xticks(batch_x, batch_x, rotation=345)
        plt.yticks(batch_y, classes)


        if is_save==True:
            plt.rcParams['savefig.dpi'] = 400  # 图片像素
            plt.rcParams['figure.dpi'] = 400  # 分辨率
            # plt.savefig("/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/pamap2_channel/%s.png"%title)
            plt.show()
        if is_save!=True:
            plt.show()
    else:
        pass
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
        y = flat(targets.cpu())
        outputs, plot_att, plot_att1,plot_att2 = net(inputs, y)
        # print(type(y),np.array(y.cpu()))
        plot_attention(y, plot_att2, special_epoch=14, special_features=125, batch_idx=batch_idx,is_save=True)
        # if batch_idx==5:
        #
        #     # print(targets,'yyyyyyyyyyyyyyyyyyyy')
        #     outputs,plot_att,plot_att1 = net(inputs,y)
        #     print(batch_idx,y[0],y,'yyyyyyyyyyyyyyyyyyyy')
        #     # train_y = flat(targets)
        #
        #     plot_att=plot_att[0,50,:,:].transpose(1,0)
        #     plt.figure(figsize=(8, 6))
        #     plt.title('walking-upstair')
        #     plt.imshow(plot_att.cpu().detach().numpy(), interpolation='nearest', cmap=plt.cm.Oranges)
        #     plt.colorbar(fraction=0.006, pad=0.07)
        #     plt.xlabel(r"time step", fontsize=10)
        #     plt.ylabel(r'class', fontsize=10)
        #     classes = ['acc0_x', 'acc0_y', 'acc0_z', 'acc1_x', 'acc1_y', 'acc1_z','gyro_x', 'gyro_y', 'gyro_z']
        #
        #     batch_x = np.arange(0, 63, 4)
        #     batch_y = np.arange(0, 9, 1)
        #     tick_marks = np.arange(len(classes))
        #     plt.tick_params(labelsize=8)
        #     plt.xticks(batch_x, batch_x, rotation=345)
        #     plt.yticks(batch_y, classes)
        #     plt.show()
        #     # print(y[124].item(),plot_att[1,1,:,:].shape)
        # else:
        #     outputs, _,_ = net(inputs, y)

            # targets=torch.max(targets, 1)[1]

        loss = criterion(outputs,torch.max(targets, 1)[1].long())
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # # predicted = torch.max(predicted, 1)[1].cuda()
        # targets=torch.max(targets, 1)[1].cuda()
        # predicted=predicted
        # taccuracy = (torch.sum(predicted == targets.long()).type(torch.FloatTensor) / targets.size(0)).cuda()



def test(epoch):
    global best_acc
    net.eval()
    print(net)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.FloatTensor)
            inputs, targets = inputs.cuda(), targets
            outputs = net(inputs,targets)
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
            # print(error_list)
            # np.save('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/uci/epoch_uci_resnet_att.npy',epoch_list)
            # np.save('/home/gaowenbing/desktop/dd/Torch_Har_cbam/store_visual/uci/error_uci_resnet_att.npy',error_list)



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
    # test(epoch)

