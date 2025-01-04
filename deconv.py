'''
Plan
1. load trained model (done)
2. create hook to store activations (done)
3. create function to deconvolve the activations (done)
4. load one image (done)
5. deconvolve every filter in the first layer (edit: for now decode the first filter)
5a. find the largest activation in a filter (done)
5b. set all activations except that to zero (done)
5c. deconvolve that filter in part b -> get the part of the image this focuses on (done)
6. plot the deconvolved parts on top of the image
'''

from typing import Callable, Dict
import torch
import torch.nn as nn
from load_data import load_test_data_for_plotting
from CNN import CNN
import einops
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np



model: CNN = torch.load('CNN.pth')
mps_device: torch.device = torch.device('cpu')
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

model.to(device=mps_device)


activations: Dict[str, torch.Tensor] = {}
def store_activations(name) -> Callable:
    def hook(model, input, output):

        activations[name] = output.detach()

    return hook


deconv: nn.ConvTranspose2d = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, bias=False)
# deconv expects something of dimension [b, 32, 96, 96]. gives out [b, 3, 96, 96]
deconv.weight = model.conv1.conv.weight # model.conv1.conv.weight = [32, 96, 96]
deconv = deconv.to(mps_device)


assert torch.allclose(deconv.weight , model.conv1.conv.weight)

test_dataloader:DataLoader = load_test_data_for_plotting()


images: torch.Tensor # image.shape is [32, 3, 96, 96]
labels: torch.Tensor # labels.shape is [32, 10]


images, labels = next(iter(test_dataloader))
images = images.to(mps_device)
print(images.shape)
h1 = model.conv1.register_forward_hook(store_activations('conv1'))
predictions: torch.Tensor = model(images) # [32, 10]

image_number: int = 0
feature_map: torch.Tensor = activations['conv1'][image_number]  # [32, 96, 96]
all_filter_positions = []
all_filter_values = []

# For each filter
for filter_idx in range(32):
    current_filter = feature_map[filter_idx]  # [96, 96]
    flattened_max_index = torch.argmax(current_filter)
    row_idx = flattened_max_index // current_filter.shape[1]
    col_idx = flattened_max_index % current_filter.shape[1]

    # Create zero tensor and set max activation
    filter_to_deconvolve = torch.zeros_like(feature_map)  # [32, 96, 96]
    filter_to_deconvolve[filter_idx, row_idx, col_idx] = current_filter[row_idx, col_idx]

    # Reshape and deconvolve
    filter_to_deconvolve = einops.rearrange(filter_to_deconvolve, 'f r c -> 1 f r c')
    deconvolved_filter = deconv(filter_to_deconvolve)  # [1, 3, 96, 96]

    # Get image format
    img = einops.rearrange(deconvolved_filter, 'b c w h -> h w (b c)')  # [96, 96, 3]

    # Find non-zero positions and values
    non_zero_positions = torch.nonzero(img > 1e-6)
    non_zero_values = img[img > 1e-6]

    all_filter_positions.append((filter_idx, non_zero_positions))
    all_filter_values.append(non_zero_values)

# Now all_filter_positions contains list of (filter_index, positions) for each filter
# and all_filter_values contains the corresponding values


# Create a different color for each filter
import matplotlib.pyplot as plt
from matplotlib import colormaps  # Modern way to get colormaps
import numpy as np
from typing import List, Tuple, TypeVar

T = TypeVar('T')

# Create a different color for each filter
colors = colormaps['viridis'](np.linspace(0, 1, 32))  # 32 distinct colors

# Create separate activation map for each filter
filter_activations = np.zeros((32, 96, 96))  # One layer per filter

# Fix typing for positions and values
for filter_idx, filter_data in enumerate(zip(all_filter_positions, all_filter_values)):
    (_, filter_positions), filter_values = filter_data  # Proper unpacking with type hints
    for pos, val in zip(filter_positions, filter_values):
        row, col, _ = pos  # Ignore channel as we'll use color instead
        filter_activations[filter_idx, row, col] = val.item()

# Plot
fig, axs = plt.subplots(4, 8, figsize=(20, 10))
fig.suptitle('Activations by Filter')

for filter_idx in range(32):
    row = filter_idx // 8
    col = filter_idx % 8
    axs[row, col].imshow(filter_activations[filter_idx], cmap='viridis')
    axs[row, col].set_title(f'Filter {filter_idx}')
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()
