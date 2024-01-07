import torch
import torchvision


class HeadAndEmbedding(torch.nn.Module):
    def __init__(self, head):
        super(HeadAndEmbedding, self).__init__()
        self.head = head

    def forward(self, x):
        return x, self.head(x)


def _alexnet_replace_fc(model):
    model.classifier = HeadAndEmbedding(model.classifier)
    return model


def resnet50_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    return model


def vitb8_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    return model


def alexnet():
    model = torchvision.models.alexnet(pretrained=True)
    return _alexnet_replace_fc(model)
