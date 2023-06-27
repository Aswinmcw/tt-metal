import torch
import torchvision
import torchvision.transforms as transforms
import tt_lib
from lenet.reference.lenet import LeNet5
from PIL import Image

def load_torch_lenet(weka_path, num_classes):
    model2 = LeNet5(num_classes).to("cpu")
    checkpoint = torch.load(weka_path, map_location=torch.device("cpu"))
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2.eval()
    return model2, checkpoint["model_state_dict"]


def prepare_image(image: Image) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
        ]
    )(image).unsqueeze(0)
