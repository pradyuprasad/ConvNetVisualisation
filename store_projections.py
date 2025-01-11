'''
1. load all images of a specific class
2. store their projections to a json
'''

import json
from typing import Dict, List, Optional
import einops
import torch
from CNN import CNN
import torch.nn.functional as F
from deconv import get_dataloader_of_specific_class, get_max_activation_coords
import os



class BinaryResult:
    def __init__(self, success: bool, error: Optional[BaseException]=None ):
        if not success:
            assert error is not None, "If there is no success, error is to be mentioned"
        self.success = success
        self.error = error

class DeconvEnv:

    def __init__(self, model:CNN, deconv: torch.nn.ConvTranspose2d, dataloader:torch.utils.data.DataLoader, device: torch.device, batch_size:int):
        assert model is not None
        assert deconv is not None
        assert dataloader is not None
        assert torch.device is not None
        assert deconv.bias is None
        assert deconv.in_channels == model.conv1.conv.out_channels
        assert deconv.out_channels == model.conv1.conv.in_channels
        assert deconv.padding == (1, 1), f"Expected padding = 1, got {deconv.padding}"
        assert deconv.kernel_size == model.conv1.conv.kernel_size
        self.model: CNN = model.to(device)
        assert torch.allclose(model.conv1.conv.weight, deconv.weight)
        self.deconv: torch.nn.ConvTranspose2d = deconv
        self.dataloader: torch.utils.data.DataLoader = dataloader
        self.device: torch.device = device
        self.batch_size = batch_size

    def __repr__(self):
        return f"BinaryResult(success={self.success}, error={self.error})"



def setup(allowed_class:int) -> DeconvEnv:
    model = torch.load('CNN.pth')
    deconv =  torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, bias=False)
    deconv.weight = model.conv1.conv.weight
    allowed_classes = [allowed_class]
    dataloader = get_dataloader_of_specific_class(allowed_classes=allowed_classes)
    device = torch.device('cpu')
    return DeconvEnv(model, deconv, dataloader, device, 32)


def single_image_deconv_save(image: torch.Tensor, deconv_env: DeconvEnv, filename:str) -> BinaryResult:
    if '.' not in filename or filename.split('.', 1)[1].lower() != 'json':
        return BinaryResult(False, ValueError("filename should end in json"))

    if image.shape != torch.Size([1, 3, 96, 96]):
        return BinaryResult(False, ValueError(f"Shape must be [1, 3, 96, 96] got {image.shape}"))

    activations: Dict[str, torch.Tensor] = {}
    def store_activations(name: str):
        def hook(module, input, output):
            activations[name] = output
        return hook

    try:

        image = image.to(device=deconv_env.device)
        model: CNN = deconv_env.model
        model = model.to(deconv_env.device)
        model.conv1.register_forward_hook(store_activations('conv1'))
        model(image)
        feature_map_current_image: torch.Tensor = activations['conv1'][0]
        assert feature_map_current_image.shape == torch.Size([32, 96, 96])
        projections_of_image: Dict[int, List] = {}
        for i in range(32):
            feature_map_current_filter = feature_map_current_image[i]
            assert feature_map_current_filter.shape == torch.Size([96, 96]), f"Expected [96, 96], got {feature_map_current_image.shape}"
            max_activ_x, max_activ_y = get_max_activation_coords(feature_map_current_filter)
            activations_to_deconv = torch.zeros_like(feature_map_current_image)
            activations_to_deconv[i, max_activ_x, max_activ_y] = feature_map_current_filter[max_activ_x][max_activ_y]
            activations_to_deconv = einops.rearrange(activations_to_deconv, "f x y -> 1 f x y")
            activations_to_deconv = activations_to_deconv.to(deconv_env.device)
            projection_of_this_filter = deconv_env.deconv(activations_to_deconv)

            projection_of_this_filter = F.relu(projection_of_this_filter)
            projection_to_save = einops.rearrange( projection_of_this_filter ,"b c h w -> h w (b c)").detach().cpu().numpy().tolist()
            assert len(projection_to_save) == 96
            assert len(projection_to_save[0]) == 96
            assert len(projection_to_save[0][0]) == 3
            projections_of_image[i] = projection_to_save

        with open(filename, "w") as f:
            json.dump(projections_of_image, f)

        return BinaryResult(True)


    except BaseException as e:
       if isinstance(e, KeyboardInterrupt):
           raise
       else:
           return BinaryResult(success=False, error=e)


def main() -> None:
    os.makedirs("projections", exist_ok=True)
    for allowed_class in range(10):
        deconv_env = setup(allowed_class=allowed_class)
        dataloader = deconv_env.dataloader
        for batch_no, (images, labels) in enumerate(dataloader):
            print("starting to process ", batch_no)
            for idx, image in enumerate(images):
                img_count = batch_no*deconv_env.batch_size + idx
                file_name = f"projections/class_{allowed_class}_image{img_count}" + ".json"
                image = einops.rearrange(image, "c h w -> 1 c h w")
                result: BinaryResult = single_image_deconv_save(image=image, deconv_env=deconv_env, filename=file_name)
                if result.success:
                    print(f"in class {allowed_class} image {idx} of batch {batch_no} processed")
                else:
                    print(f"failed to process image {idx} of batch {batch_no} in class {allowed_class} with error {result.error}")


if __name__ == "__main__":
    main()
