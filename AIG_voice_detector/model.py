from torch import nn

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 conv blocks -> flatten -> 1 dense layer (linear) -> output
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = None  
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Calculate the flattened size dynamically
        if self.linear is None:
            flattened_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.linear = nn.Linear(flattened_size, 1)
        
        x = self.flatten(x)
        x = self.linear(x)
        output = self.sigmoid(x)

        return output
