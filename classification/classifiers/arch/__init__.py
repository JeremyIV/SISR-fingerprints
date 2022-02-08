from classification.utils.registry import ARCH_REGISTRY
import classification.classifiers.arch.backbones
import classification.classifiers.arch.xception

# common interface supplied by all cnnes:
# they are all a nn.Module
# static preprocess(image)
#   takes in a PIL image and spits out a torch tensor
# transfer_state_dict(state_dict)
#   Like load state dict, except
#   this method should handle case where the last
#   layer is of a different size by replacing with
#   a newly initialized last layer
# freeze_all_but_last_layer
# unfreeze_all


def get_cnn(cnn_opt):
    cnn_opt = cnn_opt.copy()
    cnn_type = cnn_opt.pop("type")
    return ARCH_REGISTRY.get(cnn_type)(**cnn_opt)
