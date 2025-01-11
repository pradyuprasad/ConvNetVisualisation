import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN import CNN
from load_data import load_test_data_for_plotting
import numpy as np
from pathlib import Path
from tqdm import tqdm



device = torch.device('cpu')

class DeconvCNN(nn.Module):
    def __init__(self, model:CNN):
        super().__init__()
        self.model: CNN = model.to(device)
        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), bias=False, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), bias=False, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), bias=False, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3, 3), bias=False, padding=1)
        self.deconv4.weight = nn.Parameter(model.conv4.conv.weight)
        self.deconv3.weight = nn.Parameter(model.conv3.conv.weight)
        self.deconv2.weight = nn.Parameter(model.conv2.conv.weight)
        self.deconv1.weight = nn.Parameter(model.conv1.conv.weight)
        self.to(device)  # Move all deconv layers to device

    def forward(self, x: torch.Tensor, filter_number:int) -> torch.Tensor:
        assert x.shape.__len__() == 4
        assert x.shape[1:] == torch.Size([3, 96, 96]), f"Expected [b, 3, 96, 96] got {x.shape}"
        x = x.to(device)
        activation = {}
        def getActivation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        h4 = self.model.conv4.register_forward_hook(getActivation('conv4'))
        h3 = self.model.conv3.register_forward_hook(getActivation('conv3'))
        h2 = self.model.conv2.register_forward_hook(getActivation('conv2'))
        h1 = self.model.conv1.register_forward_hook(getActivation('conv1'))
        self.model(x)
        h4.remove()
        h3.remove()
        h2.remove()
        h1.remove()
        indices_conv_4 = self.model.conv4.indices
        indices_conv_2 = self.model.conv2.indices


        running_activations = einops.rearrange(activation['conv4'][0], 'c h w -> 1 c h w')
        assert running_activations.shape == torch.Size([1, 256, 6, 6]), f"Expected [1, 3, 96, 96], got {running_activations.shape}"

        running_activations = self.get_activations_to_deconv(running_activations, filter_number)

        assert running_activations.shape == torch.Size([1, 256, 6, 6]), f"Expected [1, 3, 96, 96], got {running_activations.shape}"


        indices = einops.rearrange(indices_conv_4[0], 'c h w -> 1 c h w')


        running_activations = F.max_unpool2d(input=running_activations, indices=indices, kernel_size=(4, 4))
        assert running_activations.shape[1:] == torch.Size([256, 24, 24]), f"Expected size[b, 256, 24, 24], got {running_activations.shape} after unpooling" # after unpooling



        running_activations = self.deconv4(running_activations)
        running_activations = F.relu(running_activations)
        assert running_activations.shape[1:] == torch.Size([128, 24, 24]), f"Expected size[b, 128, 24, 24], got {running_activations.shape} after deconvolution4"

        running_activations = self.deconv3(running_activations)
        running_activations = F.relu(running_activations)
        assert running_activations.shape[1:] == torch.Size([64, 24, 24]), f"Expected size[b, 64, 24, 24], got {running_activations.shape} after deconvolution3"


        indices = einops.rearrange(indices_conv_2[0], "c h w -> 1 c h w")
        running_activations = F.max_unpool2d(input=running_activations, indices=indices, kernel_size=(4, 4))

        assert running_activations.shape == torch.Size([1, 64, 96, 96]), f"Expected [1, 964, 6, 96] but got {running_activations.shape}"

        running_activations = self.deconv2(running_activations)
        running_activations = F.relu(running_activations)

        assert running_activations.shape == torch.Size([1, 32, 96, 96]), f"Expected [1, 32, 96, 96] but got {running_activations.shape}"

        running_activations = self.deconv1(running_activations)
        running_activations = F.relu(running_activations)

        assert running_activations.shape == torch.Size([1, 3, 96, 96]), f"Expected [1, 3, 96, 96] but got {running_activations.shape}"


        return running_activations

    def get_activations_to_deconv(self, activations: torch.Tensor, filter_number:int) -> torch.Tensor:
        assert activations.shape == torch.Size([1, 256, 6, 6]), f"Expected [1, 256, 6, 6] but got {activations.shape}"

        assert filter_number <= 255 and filter_number >= 0, f"filter number must be in range [1, 255] but got {filter_number}"

        output = torch.zeros_like(activations)

        activations = activations[0]

        specific_activation = activations[filter_number]

        assert specific_activation.shape.__len__() == 2, f"Expected 2D tensor, got shape {specific_activation.shape}"

        flat_index = torch.argmax(specific_activation)
        height = flat_index // specific_activation.shape[1] #  row
        width = flat_index % specific_activation.shape[1] # column

        output[0, filter_number, height, width] = specific_activation[height, width]

        return output



def main():

    # Create output directory if it doesn't exist
    output_dir = Path("stl10_deconv_visualizations")
    output_dir.mkdir(exist_ok=True)

    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    for class_idx in tqdm(range(10), desc="Processing classes"):
        images, labels = next(iter(load_test_data_for_plotting(allowed_classes=[class_idx], shuffle=True)))

        model = torch.load('CNN.pth', map_location=device)
        deconv = DeconvCNN(model=model)

        for img_idx, specific_image in enumerate(images):
            specific_image = einops.rearrange(specific_image, 'c h w -> 1 c h w')

            # Create a single numpy array to store all filter visualizations
            # Shape will be [256, 96, 96, 3] for all filters
            all_visualizations = np.zeros((256, 96, 96, 3))

            for filter_number in range(256):
                projection_on_image = deconv(specific_image, filter_number)
                projection_on_image = einops.rearrange(projection_on_image, "b c h w -> h w (b c)")
                all_visualizations[filter_number] = projection_on_image.detach().cpu().numpy()

            # Create filename
            filename = output_dir / f"{class_names[class_idx]}_class_{class_idx}_image_{img_idx}.npz"

            # Save compressed
            np.savez_compressed(
                filename,
                visualizations=all_visualizations,
                metadata=np.array([
                    class_idx,
                    img_idx,
                    class_names[class_idx]
                ], dtype=object)
            )

            print(f"Saved visualizations for {class_names[class_idx]} (class {class_idx}), image {img_idx} to {filename}")

if __name__ == "__main__":
    main()
