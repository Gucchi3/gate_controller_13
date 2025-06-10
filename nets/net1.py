import torch
import torch.nn as nn


class net1_ex(nn.Module):
    def __init__(self):
        super(net1_ex, self).__init__()
        # conv1ブロック
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(16)
        self.relu1b = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)  # 160 -> 80

        # conv2ブロック
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(32)
        self.relu2b = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)  # 80 -> 40

        # conv3ブロック
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3b = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(48)
        self.relu3b = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)  # 40 -> 20

        # conv4ブロック
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(48)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4b = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(48)
        self.relu4b = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)  # 20 -> 10

        self.fc = nn.Linear(10*10*48, 9)  # 1x9

    def forward(self, x):
        # conv1ブロック
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.relu1b(x)
        x = self.pool1(x)

        # conv2ブロック
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu2b(x)
        x = self.pool2(x)

        # conv3ブロック
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu3b(x)
        x = self.pool3(x)

        # conv4ブロック
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.relu4b(x)
        x = self.pool4(x)

        x = torch.flatten(x, 1)
        out = self.fc(x)
        #print(out)
        return out
        


