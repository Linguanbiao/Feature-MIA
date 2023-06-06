import warnings
import torch.optim as optim
from sklearn.metrics import accuracy_score
import torch
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from net.cnn import cnnNet
from net.leakcnn import LeaksCNN
warnings.filterwarnings("ignore", category=UserWarning)

# 设置图片的预处理操作

# device 设置为 GPU, 加载预训练模型 , VGG16 分为两部分，一部分是特征提取器，一部分是分类器


def loadData():
    # 图像数据的处理方式
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载数据集 ， 返回的是一个 tuple 数据类型 （image , label)
    ds_train = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
    ds_test = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)

    cluster = 10000
    batch_size = 128
    target_train = Subset(ds_train, range(cluster))

    targetTrainDataLoader = DataLoader(target_train, batch_size=batch_size, shuffle=True)
    targetTestDataLoader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    return targetTrainDataLoader, targetTestDataLoader


# 定义模型，特征提取器, 是指 VGG16 网络最后一个 conv 层进行特征提取
def makePreTrainedModel():
    # VGG16
    # model = models.vgg16(pretrained=True)  # 其实就是定位到第 31 层,最后一层卷积层
    # model[28] = nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # VGG19
    # model = models.vgg19(pretrained=True).features[:]
    # model[34] = nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # ResNet18
    # model = models.resnet18(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # print(model)

    # ResNet34
    # model = models.resnet34(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][2].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # print(model)

    # ResNet50
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-2])
    # 修改提取到的特征尺寸
    model[7][0].downsample[0] = nn.Conv2d(1024, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    model[7][0].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model[7][0].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model[7][1].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model[7][1].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    model[7][2].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model[7][2].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model[7][2].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # print(model)
    print(model)

    model = model.eval()    # 调整模型为测试状态
    # print(model)
    return model

# 定义特征提取的方法


def featuresExtract(model, dataLoader, device, dataType):
    model.eval()  # 将网络调到测试状态aLoader = dataLoader
    # print("特征提取模型结构为: ", model)
    with torch.no_grad():
        for step, (features, labels) in enumerate(dataLoader):
            features = features.to(device)
            labels = labels.to(device)
            # 特征提取
            features_result = model(Variable(features))
            trainX = features_result.cpu().detach().numpy()
            trainY = labels.cpu().detach().numpy()
            if(step == 0):
                trainDataFeatures = trainX
                trainDataLabels = trainY
                continue
            trainDataFeatures = np.concatenate((trainDataFeatures, trainX), axis=0)
            trainDataLabels = np.concatenate((trainDataLabels, trainY), axis=0)
            print("特征提取中: {} / {}".format(step, 79))
        print("提取到的图片特征尺寸大小为：{} , 标签的尺寸大小是：{} , 数据的类型是：{}".format(
            trainDataFeatures.shape, trainDataLabels.shape, type(trainDataFeatures)))
    if(dataType == 0):
        # type = 0  表示是训练数据做特征提取， type = 1 表示是测试数据做特征提取
        np.savez('./Preprocessed/CIFAR10_ResNet50/32_all_targetTrainData.npz', trainDataFeatures, trainDataLabels)
    else:
        np.savez('./Preprocessed/CIFAR10_ResNet50/32_all_targetTestData.npz', trainDataFeatures, trainDataLabels)


def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_pred_cls = y_pred_cls.cpu().detach().numpy()   # 将 tensor 转换成 numpy 形式
    y_true = y_true.cpu().detach().numpy()
    return accuracy_score(y_pred_cls, y_true)


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


def load_data(data_name):
    with np.load(data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y


def train(device):
    data_path = './Preprocessed/CIFAR10_ResNet50'
    trainX, trainY = load_data(data_path + '/32_all_targetTrainData.npz')
    testX, testY = load_data(data_path + '/32_all_targetTestData.npz')
    # print("查看标签是什么----", testY[:100])

    print("训练数据的尺寸是:{} , 标签的尺寸是:{} , 数据类型是:{}".format(trainX.shape, trainY.shape, type(trainX)))
    net = cnnNet()
    # net = LeaksCNN()
    net.to(device)
    cross_loss = nn.CrossEntropyLoss()
    optimezer = optim.Adam(params=net.parameters(), lr=0.0003)
    # optimezer = optim.SGD(params=net.parameters(), lr=0.01, momentum=0.5)
    metric_fuc = accuracy
    epochs = 50
    batch_size = 256
    print("Train...")
    net.train()
    for epoch in range(epochs):
        # 训练循环
        steps = 1
        loss_sum = 0.0
        metric_sum = 0.0
        for features_batch, labels_batch in iterate_minibatches(trainX, trainY, batch_size):
            features_batch, labels_batch = torch.tensor(features_batch), torch.tensor(labels_batch).type(torch.long)
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            # 梯度置 0
            optimezer.zero_grad()
            # 前向传播
            prediction_y = net(features_batch)
            loss = cross_loss(prediction_y, labels_batch)
            # 反向传播
            loss.backward()
            optimezer.step()
            acc = metric_fuc(prediction_y, labels_batch)
            loss_sum += loss.item()
            metric_sum += acc.item()

            if(steps % 10 == 0):
                print("Epoch {} / iteration {} , train loss {} , train accuracy{} ".format(epoch,
                                                                                           steps, loss_sum / steps, metric_sum / steps))
            steps += 1
        # 测试循环
        val_steps = 1
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        net.eval()  # 把模型调成测试状态
        for features_batch, labels_batch in iterate_minibatches(testX, testY, batch_size):
            features_batch, labels_batch = torch.tensor(features_batch), torch.tensor(labels_batch).type(torch.long)
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            with torch.no_grad():
                prediction_y = net(features_batch)
                loss = cross_loss(prediction_y, labels_batch)
                test_acc = metric_fuc(prediction_y, labels_batch)
                val_loss_sum += loss.item()
                val_metric_sum += test_acc.item()
                val_steps += 1
        print("[Epoch {} / Epochs{}] -------- Test loss {} -------- Test accuracy {} ".format(epoch,
              epochs, val_loss_sum / val_steps, val_metric_sum / val_steps))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    isFirstFeaturesExtract = False
    model = makePreTrainedModel()
    model = model.to(device)
    if(isFirstFeaturesExtract):
        targetTrainDataLoader,  targetTestDataLoader = loadData()
        featuresExtract(model, targetTrainDataLoader, device, 0)   # 训练数据的特征提取
        featuresExtract(model, targetTestDataLoader, device, 1)    # 测试数据的特征提取
    train(device)
