# arch_base.py
from torch import nn


class arch_base(nn.Module):
    def __init__(self, feature_extractor, fc, patch_size):
        super(arch_base, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = fc
        self.patch_size = patch_size

    def forward(self, x):
        features = self.feature_extractor(x)
        probs = self.fc(features)
        return probs, features

    def transfer_state_dict(self, state_dict):
        if self.fc.weight.shape != state_dict["fc.weight"].shape:
            state_dict = state_dict.copy()
            state_dict["fc.weight"] = self.fc.weight.detach()
            state_dict["fc.bias"] = self.fc.bias.detach()
        self.load_state_dict(state_dict)

    def freeze_all_but_last_layer(self):
        for i, param in self.named_parameters():
            param.requires_grad = False
        for i, param in self.fc.named_parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for i, param in self.named_parameters():
            param.requires_grad = True
