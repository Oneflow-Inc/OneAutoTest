import oneflow.mock_torch as mock
with mock.enable():
    import torch
    import flowvision
    import flowvision.transforms as transforms
    import torch.nn as nn
    import torch.optim as optim
    print(torch.__file__)
    import oneflow as flow
    

    x = torch.zeros(2, 3)
    print(isinstance(x, flow.Tensor))

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = flowvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)

    testset = flowvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=2)

    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.resnet = flowvision.models.resnet18(pretrained=False)
            self.resnet.fc = nn.Linear(512, 10)

        def forward(self, x):
            x = self.resnet(x)
            return x
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


    for epoch in range(10):  # 多次遍历数据集
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播、反向传播、优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()
            if i % 100 == 99:    # 每 100 个小批量数据打印一次平均损失
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    torch.save(net.state_dict(), '/data/home/sunjinfeng/torch_mock/resnet18')

    # 用训练好的模型进行推理
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))    