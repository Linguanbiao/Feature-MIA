import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


tf = transforms.Compose([transforms.Resize(32),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])

trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True,
                                        transform=tf)


testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True,
                                       transform=tf)
trainset = Subset(trainset, range(10000))
# dataset = torch.utils.data.ConcatDataset([trainset, testset])


trainloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                          num_workers=2, shuffle=True)

print(len(trainset))
print(trainset[0][0].size())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fake')


def showImage(images, epoch=-1, idx=-1):
    images = images.cpu().numpy()
    images = images/2 + 0.5
    plt.imshow(np.transpose(images, axes=(1, 2, 0)))
    plt.axis('off')
    if idx % 99 == 0 and idx != 0:
        # plt.savefig("./fakedata/cifar10/e" + str(epoch) + "i" + str(idx) + ".png")
        plt.savefig("/kolla/lgb_mia/Features_MIA/visualize/gan_32_10000_cifar_testdata/epoch_" + str(epoch) + ".png")


dataiter = iter(trainloader)
images, labels = dataiter.next()
print("特征的尺寸是: ", images.size())
showImage(make_grid(images[0:32]))

# ------------------------------------------ 生成 3*64*64 的CIFAR10图片 -----------------------------------------
# class Generator(nn.Module):

#     def __init__(self):
#         super(Generator, self).__init__()

#         # input 100*1*1
#         self.layer1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
#                                     nn.ReLU(True))

#         # input 512*4*4
#         self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(256),
#                                     nn.ReLU(True))
#         # input 256*8*8
#         self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(128),
#                                     nn.ReLU(True))
#         # input 128*16*16
#         self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(64),
#                                     nn.ReLU(True))
#         # input 64*32*32
#         self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
#                                     nn.Tanh())
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


#  将 CIFAR10 处理成 3*64*64 的生成器
# class Discriminator(nn.Module):

#     def __init__(self):
#         super(Discriminator, self).__init__()

#         # input 3*64*64  -> 3*32*32
#         self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(64),
#                                     nn.LeakyReLU(0.2, True),
#                                     nn.Dropout2d(0.5))

#         # input 64*32*32
#         self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(128),
#                                     nn.LeakyReLU(0.2, True),
#                                     nn.Dropout2d(0.5))
#         # input 128*16*16
#         self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(256),
#                                     nn.LeakyReLU(0.2, True),
#                                     nn.Dropout2d(0.5))
#         # input 256*8*8
#         self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#                                     nn.BatchNorm2d(512),
#                                     nn.LeakyReLU(0.2, True))
#         # input 512*4*4
#         self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False),
#                                             nn.Sigmoid())

#         self.label_layer = nn.Sequential(nn.Conv2d(512, 11, 4, 1, 0, bias=False),
#                                          nn.LogSoftmax(dim=1))

#     def forward(self, x):

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         validity = self.validity_layer(x)
#         plabel = self.label_layer(x)

#         validity = validity.view(-1)
#         plabel = plabel.view(-1, 11)

#         return validity, plabel
# ---------------------------------------------------- 3*32*32 的生成器和判别器 ------------------------------------------------

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                                    nn.ReLU(True))

        # input 256*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))
        # input 128*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))
        # input 64*16*16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 2, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))
        # input 30*30
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 3, 1, 0, bias=False),
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


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # input 3*32*32
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 16*16*16
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 32*16*16
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 64*8*8
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
       # input 128*8*8
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, True))
       # input 256*4*4
        self.layer6 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2, True))
        # input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                                            nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(512, 11, 4, 1, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, 11)

        return validity, plabel

# custom weights initialization called on netG and netD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


gen = Generator().to(device)
gen.apply(weights_init)

disc = Discriminator().to(device)
disc.apply(weights_init)

paramsG = list(gen.parameters())
print(len(paramsG))

paramsD = list(disc.parameters())
print(len(paramsD))

optimG = optim.Adam(gen.parameters(), 0.0002, betas=(0.5, 0.999))
optimD = optim.Adam(disc.parameters(), 0.0002, betas=(0.5, 0.999))

validity_loss = nn.BCELoss()

real_labels = 0.7 + 0.5 * torch.rand(10, device=device)
fake_labels = 0.3 * torch.rand(10, device=device)
print("real_labels: {} , fake_labels:{}".format(real_labels, fake_labels))
epochs = 2000

for epoch in range(1, epochs+1):

    for idx, (images, labels) in enumerate(trainloader, 0):

        batch_size = images.size(0)
        labels = labels.to(device)
        images = images.to(device)

        real_label = real_labels[idx % 10]
        fake_label = fake_labels[idx % 10]

        fake_class_labels = 10*torch.ones((batch_size,), dtype=torch.long, device=device)
        print("real_label:{} , fake_label{}".format(real_label, fake_label))

        # if idx % 25 == 0:
        #     real_label, fake_label = fake_label, real_label

        # ---------------------
        #         disc
        # ---------------------

        optimD.zero_grad()

        # real
        validity_label = torch.full((batch_size,), real_label, device=device)

        pvalidity, plabels = disc(images)

        errD_real_val = validity_loss(pvalidity, validity_label)
        errD_real_label = F.nll_loss(plabels, labels)

        errD_real = errD_real_val + errD_real_label
        errD_real.backward()

        D_x = pvalidity.mean().item()

        # fake
        noise = torch.randn(batch_size, 100, device=device)
        sample_labels = torch.randint(0, 10, (batch_size,), device=device, dtype=torch.long)

        fakes = gen(noise, sample_labels)
        print("生成fake图片的尺寸是:", fakes.shape)
        validity_label.fill_(fake_label)

        pvalidity, plabels = disc(fakes.detach())

        errD_fake_val = validity_loss(pvalidity, validity_label)
        errD_fake_label = F.nll_loss(plabels, fake_class_labels)

        errD_fake = errD_fake_val + errD_fake_label
        errD_fake.backward()

        D_G_z1 = pvalidity.mean().item()

        # finally update the params!
        errD = errD_real + errD_fake

        optimD.step()

        # ------------------------
        #      gen
        # ------------------------

        optimG.zero_grad()

        noise = torch.randn(batch_size, 100, device=device)
        sample_labels = torch.randint(0, 10, (batch_size,), device=device, dtype=torch.long)

        validity_label.fill_(1)

        fakes = gen(noise, sample_labels)
        print("生成fake图片的尺寸是:", fakes.shape)
        pvalidity, plabels = disc(fakes)

        errG_val = validity_loss(pvalidity, validity_label)
        errG_label = F.nll_loss(plabels, sample_labels)

        errG = errG_val + errG_label
        errG.backward()

        D_G_z2 = pvalidity.mean().item()

        optimG.step()

        print("[{}/{}] [{}/{}] D_x: [{:.4f}] D_G: [{:.4f}/{:.4f}] G_loss: [{:.4f}] D_loss: [{:.4f}] D_label: [{:.4f}] "
              .format(epoch, epochs, idx, len(trainloader), D_x, D_G_z1, D_G_z2, errG, errD,
                      errD_real_label + errD_fake_label + errG_label))

        if idx % 99 == 0 and idx != 0:
            noise = torch.randn(10, 100, device=device)
            labels = torch.arange(0, 10, dtype=torch.long, device=device)

            gen_images = gen(noise, labels).detach()

            showImage(make_grid(gen_images), epoch, idx)

    torch.save(gen.state_dict(), './saved_model/cifar10_32_10000_testdata/epoch{}_gen.pth'.format(epoch))
    torch.save(disc.state_dict(), './saved_model/cifar10_32_10000_testdata/epoch{}_disc.pth'.format(epoch))
