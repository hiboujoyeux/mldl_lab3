import torch
import torch.nn as nn
import torch.nn.functional as F

def CustomNet():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            
            self.pool = nn.MaxPool2d(2, 2)
            
            self.fc1 = nn.Linear(512 * 14 * 14, 1024)
            self.fc2 = nn.Linear(1024, 200)  # Tiny ImageNet has 200 classes
            
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return Net()
