from classification.utils.registry import ARCH_REGISTRY
from torchvision import models, transforms
from torch import nn
from classification.classifiers.arch.utils import Identity
from classification.classifiers.arch import arch_base

preprocessor = transforms.compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@ARCH_REGISTRY.register()
class ResNet50(arch_base):
    def __init__(self, num_classes):
        rnet = models.resnet50(pretrained=True)
        in_features = self.rnet.fc.in_features
        rnet.fc = Identity()
        fc = Linear(in_features, num_classes)
        super().__init__(feature_extractor=rnet, fc=fc, patch_size=224)

    @static
    def preprocess(image):
        return preprocessor(image)


@ARCH_REGISTRY.register()
class Xception(arch_base):
    def __init__(self, num_classes):
        net = xception.Xception()
        in_features = self.net.fc.in_features
        net.fc = Identity()
        fc = Linear(in_features, num_classes)
        super().__init__(feature_extractor=net, fc=fc, patch_size=299)

    @static
    def preprocess(image):
        return preprocessor(image)


# efficientNet

# vit
