import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

train_x = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/weakly_data_k/train_x.npy')
train_y = (np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/weakly_data_k/train_y.npy'))
test_x = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/weakly_data_k/test_x.npy')
test_y = np.load('/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/weakly_data_k/test_y.npy')


print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
#(8380, 2048, 3) (8380, 4)
#(713, 2048, 3) (713, 4)

test_x_1=np.array(test_x)[461,:,0]
test_x_2=np.array(test_x)[461,:,1]
test_x_3=np.array(test_x)[461,:,2]
# plt.title("weakly labled dataset")
plt.plot([i for i in range(len(test_x_1))],test_x_1,'b')
plt.plot([i for i in range(len(test_x_1))]  ,test_x_2,'g')
plt.plot([i for i in range(len(test_x_1))],test_x_3,'r')

plt.show()
