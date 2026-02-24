from torch import nn

from clip_reproduction.models.clip import CLIPModel, build_clip_model
from clip_reproduction.models.vision import CNNModel, ResNet50Classifier


def create_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()

    if key == "clip":
        return build_clip_model(**kwargs)

    num_classes = kwargs.get("num_classes", 100)
    pretrained = kwargs.get("pretrained", True)

    if key == "cnn":
        return CNNModel(num_classes=num_classes)

    if key == "resnet50":
        return ResNet50Classifier(num_classes=num_classes)

    available = ["clip", "cnn", "resnet50"]
    raise ValueError(f"Unknown model '{name}'. Available models: {available}")


def is_clip_model(model: nn.Module) -> bool:
    return isinstance(model, CLIPModel)
