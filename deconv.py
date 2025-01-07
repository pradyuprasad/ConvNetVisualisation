from typing import Dict, Generic, Tuple, TypeVar
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import load_test_data_for_plotting
from CNN import CNN
from torch.utils.data import DataLoader
import einops
B = TypeVar('B')
NUM_FILTERS = 32
NUM_CHANNELS = 3


class TensorShape(Generic[B]):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

class FirstLayer(TensorShape[B]):
    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor)
        assert tensor.shape[1:] == (32, 96, 96)

class ReconstructedImage(TensorShape[B]):
    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor)
        assert tensor.shape[1:] == (3, 96, 96)



def reconstruct_image_from_first_layer(x: FirstLayer[B], deconv_layer: nn.ConvTranspose2d) -> ReconstructedImage[B]:
    '''

    '''
    output = deconv_layer(x.tensor)
    return ReconstructedImage(output)

def get_dataloader_of_specific_class() -> DataLoader:
    return load_test_data_for_plotting(batch_size=32, allowed_classes=[0])

def get_max_activation_coords(tensor: torch.Tensor) -> Tuple[int, int]:
    """Takes a 2D tensor and returns the (height, width) coordinates of its maximum value"""
    assert len(tensor.shape) == 2, f"Expected 2D tensor, got shape {tensor.shape}"
    flat_index = torch.argmax(tensor)
    height = flat_index // tensor.shape[1]  # row
    width = flat_index % tensor.shape[1]   # column
    return int(height.item()), int(width.item())

def reconstruct_wrapper() -> None:
    device: torch.device = torch.device('cpu')
    model: CNN = torch.load('CNN.pth')
    model = model.to(device)
    deconv: nn.ConvTranspose2d = nn.ConvTranspose2d(in_channels=NUM_FILTERS, out_channels=NUM_CHANNELS, kernel_size=3, padding=1, bias=False)
    deconv = deconv.to(device)
    deconv.weight = nn.Parameter(model.conv1.conv.weight)
    test_dataloader: DataLoader = get_dataloader_of_specific_class()
    images, labels = next(iter(test_dataloader))
    images = images.to(device)



    activations: Dict[str, torch.Tensor] = {}
    def store_activations(name: str):
        def hook(module, input, output):
            activations[name] = output

        return hook

    model.conv1.register_forward_hook(store_activations('conv1'))
    model(images)
    image_index = 0

    feature_map_current_image: torch.Tensor = activations['conv1'][image_index]
    for i in range(len(images)):
        if i > 0:
            break
        feature_map_current_filter = feature_map_current_image[i]
        assert feature_map_current_filter.shape == torch.Size([96, 96]), f"Expected [96, 96], got {feature_map_current_image.shape}"
        max_activ_x, max_activ_y = get_max_activation_coords(feature_map_current_filter)
        activations_to_deconv = torch.zeros_like(feature_map_current_image)
        activations_to_deconv[i, max_activ_x, max_activ_y] = feature_map_current_filter[max_activ_x][max_activ_y]
        activations_to_deconv = einops.rearrange(activations_to_deconv, "f x y -> 1 f x y")
        with torch.no_grad():
            activations_to_deconv = activations_to_deconv.to(device)
            projection_of_this_filter = deconv(activations_to_deconv)
            projection_of_this_filter = F.relu(projection_of_this_filter)
            print(projection_of_this_filter)











