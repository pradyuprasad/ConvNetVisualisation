import torch
import torch.nn as nn

test_image = torch.rand((3, 32, 32))

m = nn.Conv2d(3, 16, 3)

output = m(test_image)
assert output.shape == torch.Size([16, 30, 30])
print("verified!")
