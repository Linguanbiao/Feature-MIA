import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
# from net.MLleaksCnn import cnnNet
from net.cnn import cnnNet
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_pred_cls = y_pred_cls.cpu().detach().numpy()   # 将 tensor 转换成 numpy 形式
    y_true = y_true.cpu().detach().numpy()
    return accuracy_score(y_pred_cls, y_true)


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


    # 设置图像处理
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 设置 GPU
device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")
batch_size = 250
# 加载数据  torchvision.datasets 返回的是一个 tuple 类型的数据 （image, target）
ds_train = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transforms)
ds_test = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transforms)


cluster = 10000
target_train = Subset(ds_train, range(cluster))

targetTrainDataLoader = DataLoader(target_train, batch_size=batch_size, shuffle=True)
testDataLoader = DataLoader(ds_test, batch_size=250, shuffle=True)

print("类型是" , targetTrainDataLoader)
model = cnnNet()
model.to(device)
cross_loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
metric_fuc = accuracy
epochs = 50

# 加载真实特征提取的结果
real_trainX, real_trainY = load_data(
    '/kolla/lgb_mia/Features_MIA/Preprocessed/CIFAR10_vgg19/16_all_targetTrainData.npz')
real_testX, real_testY = load_data('/kolla/lgb_mia/Features_MIA/Preprocessed/CIFAR10_vgg19/16_all_targetTestData.npz')

# 加载 fake 数据
trainX, trainY = load_data(
    '/kolla/lgb_mia/Features_MIA/fakedata/cifar10/fake_features_cifar10_10000_e2000_targetTrainData.npz')
print("生成的假图片的尺寸是:{} , 标签是:{}".format(trainX.shape, trainY))
testX, testY = load_data(
    '/kolla/lgb_mia/Features_MIA/fakedata/cifar10/fake_features_cifar10_10000_e2000_targetTestData.npz')
print("fake数据的尺寸: ", testX.shape)
# 开始训练
for epoch in range(1, epochs):
    # 训练循环
    loss_sum = 0.0
    metric_sum = 0.0
    iteration = 1
    model.train()
    # 使用GAN生成的 fake 图片训练
    for features_batch, labels_batch in iterate_minibatches(real_testX, real_testY, batch_size):
        features_batch, labels_batch = torch.tensor(features_batch), torch.tensor(labels_batch).type(torch.long)
        features_batch = features_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()
        prediction_y = model(features_batch)
        loss = cross_loss(prediction_y, labels_batch)
        loss.backward()
        optimizer.step()

        accuracy = metric_fuc(prediction_y, labels_batch)
        loss_sum += loss.item()
        metric_sum += accuracy.item()
        iteration += 1

        if(iteration % 10 == 0):
            print("Epoch {} / iteration {} , train loss {} , train accuracy{} ".format(epoch,
                  iteration, loss_sum / iteration, metric_sum / iteration))

    # 测试循环
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_iteration = 1
    model.eval()  # 将网络转换到测试状态

    # for val_iteration, (features, labels) in enumerate(targetTrainDataLoader, 1):
    #     print("特征的类型：" ,  type(features) , features.shape)
    #     features, labels = torch.tensor(features), torch.tensor(labels).type(torch.long)
    #     features = features.to(device)
    #     labels = labels.to(device)
    #     with torch.no_grad():
    #         prediction_y = model(features)
    #         loss = cross_loss(prediction_y, labels)
    #         test_accuracy = metric_fuc(prediction_y, labels)
    #         val_loss_sum += loss.item()
    #         val_metric_sum += test_accuracy.item()
    #         val_iteration += 1

    # GAN 生成的假图片做测试集
    for features_batch, labels_batch in iterate_minibatches(trainX, trainY, batch_size):
        features_batch, labels_batch = torch.tensor(features_batch), torch.tensor(labels_batch).type(torch.long)
        features_batch = features_batch.to(device)
        labels_batch = labels_batch.to(device)
        with torch.no_grad():
            prediction_y = model(features_batch)
            loss = cross_loss(prediction_y, labels_batch)
            test_acc = metric_fuc(prediction_y, labels_batch)
            val_loss_sum += loss.item()
            val_metric_sum += test_acc.item()
            val_iteration += 1

    print("[Epoch {} / iteration {}] , Test loss {} , Test accuracy {} ".format(epoch,
                                                                                val_iteration, val_loss_sum / val_iteration, val_metric_sum / val_iteration))
