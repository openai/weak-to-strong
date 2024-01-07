import torch
import torchvision

RESIZE, CROP = 256, 224
TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(RESIZE),
        torchvision.transforms.CenterCrop(CROP),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_imagenet(datapath, split, batch_size, shuffle, transform=TRANSFORM):
    ds = torchvision.datasets.ImageNet(root=datapath, split=split, transform=transform)
    loader = torch.utils.data.DataLoader(ds, shuffle=shuffle, batch_size=batch_size)
    return ds, loader
