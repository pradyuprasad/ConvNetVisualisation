import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_kernels = 3, out_kernels=16, kernel_size=3, pool_kernel_size=2):
        super().__init__()

        self.conv = nn.Conv2d(in_kernels, out_kernels, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, return_indices=True)
        self.indices = None

    def forward(self, x):
        '''

        '''

        x = self.conv(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        self.indices = indices

        return x



