import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(device)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fake')


def load_data(data_name):
    with np.load(data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(
            start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]

# ---------------------------------------------------- 16*7*7 的生成器和判别器 ------------------------------------------------


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


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # input 16*7*7
        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 32*7*7
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 2, 1, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 64*8*8
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 2, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 128*5*5
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
       # input 256*5*5
        self.layer5 = nn.Sequential(nn.Conv2d(256, 512, 2, 2, 1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2, True))
       # input 512*3*3
        # self.layer6 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),
        #                             nn.BatchNorm2d(512),
        #                             nn.LeakyReLU(0.2, True))
        # input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, 2, 2, 0, bias=False),
                                            nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(512, 11, 2, 2, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.layer6(x)
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
batch_size = 100

# 加载数据集
trainx, trainy = load_data('/kolla/z/home/jack/Features_MIA/Preprocessed/CIFAR10_vgg19/16_all_targetTestData.npz')
print("训练图片的尺寸是: ", trainx.shape)

for epoch in range(1, epochs+1):

    idx = 0
    for images, labels in iterate_minibatches(trainx, trainy, batch_size):

        batch_size = batch_size
        images = torch.tensor(images)
        labels = torch.tensor(labels).type(torch.long)
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
              .format(epoch, epochs, idx, 100, D_x, D_G_z1, D_G_z2, errG, errD,
                      errD_real_label + errD_fake_label + errG_label))

        # if idx % 99 == 0 and idx != 0:
        #     noise = torch.randn(10, 100, device=device)
        #     labels = torch.arange(0, 10, dtype=torch.long, device=device)

        #     gen_images = gen(noise, labels).detach()

        torch.save(gen.state_dict(), './saved_model/cifar10_features_test_10000/epoch{}_gen.pth'.format(epoch))
        torch.save(disc.state_dict(), './saved_model/cifar10_features_test_10000/epoch{}_disc.pth'.format(epoch))
