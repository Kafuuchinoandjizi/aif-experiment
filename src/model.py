from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.5)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(120, 64)

        self.fc2 = nn.Linear(64, 10)

        self.fc3 = nn.Linear(10, 2)  # Changed output to 2 classes

        self.model1 = nn.Sequential(
            self.conv1, self.relu, self.pool,
            self.conv2, self.relu, self.pool,
            self.conv3, self.relu,
            self.adaptive_pool,  # 使用 AdaptiveAvgPool2d 调整输出形状
            self.flatten, self.dropout,
            self.fc1, self.relu,
            self.fc2, self.relu,
            self.fc3
        )

    def forward(self, x):
        return self.model1(x)
