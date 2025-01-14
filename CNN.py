import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_kernels, out_kernels, kernel_size=3, has_pool=True, pool_kernel_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_kernels, out_kernels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = None
        if has_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, return_indices=True)
        self.indices = None
        self.bn = nn.BatchNorm2d(out_kernels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x, indices = self.pool(x)
            self.indices = indices
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(in_kernels=3, out_kernels=32, has_pool=False)
        self.conv2 = ConvBlock(in_kernels=32, out_kernels=64, has_pool=True, pool_kernel_size=4)
        self.conv3 = ConvBlock(in_kernels=64, out_kernels=128, has_pool=False)
        self.conv4 = ConvBlock(in_kernels=128, out_kernels=256, has_pool=True, pool_kernel_size=4)
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features=256 * 6 * 6, out_features=100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == torch.Size([3, 96, 96]), f"Expected shape (batch, 3, 96, 96) got {x.shape}"
        x = self.conv1(x)
        assert x.shape[1:] == torch.Size([32, 96, 96]), f"Expected shape (batch, 32, 96, 96) got {x.shape}"
        x = self.conv2(x)
        assert x.shape[1:] == torch.Size([64, 24, 24]), f"Expected shape (batch, 64, 24, 24) got {x.shape}"
        x = self.conv3(x)
        assert x.shape[1:] == torch.Size([128, 24, 24]), f"Expected shape (batch, 128, 24, 24) got {x.shape}"
        x = self.conv4(x)
        assert x.shape[1:] == torch.Size([256, 6, 6]), f"Expected shape (batch, 256, 6, 6) got {x.shape}"
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        assert x.shape[1:] == torch.Size([100]), f"Expected shape (batch, 100) got {x.shape}"
        x = self.dropout(x)
        x = self.linear2(x)
        assert x.shape[1:] == torch.Size([10]), f"Expected shape (batch, 10) got {x.shape}"
        return x

def verify_shapes():
    x = torch.randn((1, 3, 96, 96))
    model = CNN()
    model(x)
    print("All shape assertions passed")

if __name__ == "__main__":
    verify_shapes()
