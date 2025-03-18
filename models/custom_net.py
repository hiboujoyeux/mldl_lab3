import torch.nn as nn

def CustomNet():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(128 * 56 * 56, 200)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.flatten(x)
            x = self.fc1(x)
            return x
    
    return Net()
