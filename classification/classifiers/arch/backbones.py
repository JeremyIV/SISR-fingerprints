from classification.utils.registry import ARCH_REGISTRY
from torchvision import models, transforms
from torch import nn
from classification.classifiers.arch.utils import Identity
from classification.classifiers.arch.arch_base import arch_base
import timm

preprocessor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@ARCH_REGISTRY.register()
class ResNet50(arch_base):
    def __init__(self, num_classes):
        rnet = models.resnet50(pretrained=True, progress=False)
        in_features = rnet.fc.in_features
        rnet.fc = Identity()
        fc = nn.Linear(in_features, num_classes)
        super().__init__(feature_extractor=rnet, fc=fc, patch_size=224)

    @staticmethod
    def preprocess(image):
        return preprocessor(image)


@ARCH_REGISTRY.register()
class EfficientNet(arch_base):
    def __init__(self, num_classes):
        enet = models.efficientnet_b2(pretrained=True, progress=False)
        in_features = enet.classifier[1].in_features
        enet.classifier[1] = Identity()
        fc = nn.Linear(in_features, num_classes)
        super().__init__(feature_extractor=enet, fc=fc, patch_size=288)

    @staticmethod
    def preprocess(image):
        return preprocessor(image)


# vit
@ARCH_REGISTRY.register()
class ViT(arch_base):
    def __init__(self, num_classes):
        vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        in_features = vit.head.in_features
        vit.head = Identity()
        fc = nn.Linear(in_features, num_classes)
        super().__init__(feature_extractor=vit, fc=fc, patch_size=224)

    @staticmethod
    def preprocess(image):
        return preprocessor(image)


# convNext
@ARCH_REGISTRY.register()
class ConvNext(arch_base):
    def __init__(self, num_classes):
        convnext = timm.create_model("convnext_base", pretrained=True)
        in_features = convnext.head.fc.in_features
        convnext.head.fc = Identity()
        fc = nn.Linear(in_features, num_classes)
        super().__init__(feature_extractor=convnext, fc=fc, patch_size=224)

    @staticmethod
    def preprocess(image):
        return preprocessor(image)
