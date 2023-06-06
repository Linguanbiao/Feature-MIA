import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 1000
# Hyperparameters  CIFAR10 的类别
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

visiable_save_path = os.path.join(os.path.curdir, "visualize", "best_fake")


# class Generator(nn.Module):

#     def __init__(self):
#         super(Generator, self).__init__()

#         # input 100*1*1
#         self.layer1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
#                                     nn.ReLU(True))

#         # input 256*4*4
#         self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(256),
#                                     nn.ReLU(True))
#         # input 128*8*8
#         self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(128),
#                                     nn.ReLU(True))
#         # input 64*16*16
#         self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 2, bias=False),
#                                     nn.BatchNorm2d(64),
#                                     nn.ReLU(True))

#         self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 3, 1, 0, bias=False),
#                                     nn.Tanh())

#         # output 3*32*32

#         # self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
#         #                             nn.Tanh())
#         # output 3*64*64

#         self.embedding = nn.Embedding(10, 100)

#     def forward(self, noise, label):

#         label_embedding = self.embedding(label)
#         x = torch.mul(noise, label_embedding)
#         x = x.view(-1, 100, 1, 1)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 3, 1, 0, bias=False),
                                    nn.ReLU(True))

        # input 512*3*3
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))
        # input 256*4*4
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 1, 0, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))
        # input 128*5*5
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 1, 0, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))
        # input 64*6*6
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 16, 2, 1, 0, bias=False),
                                    nn.Tanh())

        # output 3*32*32

        # self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
        #                             nn.Tanh())
        # output 3*64*64

        self.embedding = nn.Embedding(10, 100)

    def forward(self, noise, label):

        label_embedding = self.embedding(label)
        x = torch.mul(noise, label_embedding)
        x = x.view(-1, 100, 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def denormalize(x):
    """Denomalize a normalized image back to uint8.
    """
    # #####
    # TODO: Complete denormalization.
    # #####
    x = x.permute((0, 2, 3, 1)) * 255
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x


def draw_picture(xg):
    xg = denormalize(xg)
    plt.figure(figsize=(10, 5))
    for p in range(20):
        plt.subplot(4, 5, p+1)
        plt.imshow(xg[p])
        plt.text(0, 0, "{}".format(classes[random_y[p].item()]), color='black',
                 backgroundcolor='white', fontsize=8)
        plt.axis('off')

    plt.savefig(os.path.join(os.path.join(visiable_save_path,
                "fakeimg_label= {} .png".format(random_y[0].data))), dpi=300)
    plt.clf()
    plt.close('all')


def shuffleData(features, labels):
    c = list(zip(features, labels))
    random.shuffle(c)
    features, labels = zip(*c)
    shuffle_features = np.array(features[:])
    shffle_labels = np.array(labels[:])
    return shuffle_features, shffle_labels


generator = Generator()
generator.load_state_dict(torch.load(
    '/kolla/lgb_mia/Features_MIA/saved_model/cifar10_features_train_10000/epoch2000_gen.pth'))
generator.to(device)
generator.eval()  # 调整为测试状态

with torch.no_grad():
    for idx in range(0, 10):

        # random_z = torch.randn((1000, latent_dim)).to(device)
        # random_y = torch.randint(low=idx, high=idx + 1, size=(1000, )).to(device)

        # noise 向量
        noise = torch.randn(batch_size, 100, device=device)
        sample_labels = torch.randint(idx, idx+1, (batch_size,), device=device, dtype=torch.long)

        fakeImg = generator(noise, sample_labels)
        fakeImg = fakeImg.cpu().detach().numpy()
        sample_labels = sample_labels.cpu().detach().numpy()
        if(idx == 0):
            fakeDataFeatures = fakeImg
            fakeDataLabels = sample_labels
            continue
        fakeDataFeatures = np.concatenate((fakeDataFeatures, fakeImg), axis=0)
        fakeDataLabels = np.concatenate((fakeDataLabels, sample_labels), axis=0)
    print("生成的假数据的尺寸:{} ,标签的尺寸:{} , 标签:{}".format(fakeDataFeatures.shape, fakeDataLabels.shape, fakeDataLabels))
    np.savez('/kolla/lgb_mia/Features_MIA/fakedata/cifar10/fake_features_cifar10_10000_e2000_targetTrainData.npz',
             fakeDataFeatures, fakeDataLabels)
    # xg = generator(random_z, random_y)
    # print("生成的图片的尺寸是多大的:{}, 类型是什么{}".format(xg.shape, type(xg)))
    # # draw_picture(xg)
