# TODO:
import yaml
from utils import *
from pathlib import Path

parser_opt_dir = Path("classification/options/parsers")


def make_parser_opt_quick_test(label_param, reserved_param, reserved_param_val):
    classifier_name = f"ConvNext_CNN_SISR_{label_param}_parser_withholding_{reserved_param_val}_quick_test"
    opt = {
        "classifier": {
            "cnn": {"type": "ConvNext"},
            "name": classifier_name,
            "train": {
                "batch_size": 16,
                "learning_rate_end": 4e-06,
                "learning_rate_start": 0.0005,
                "num_full_train_epochs": 1,
                "num_pretrain_epochs": 1,
            },
            "type": "CNN",
        },
        "evaluation_datasets": [
            {
                "include_custom_trained": True,
                "include_pretrained": True,
                "label_param": label_param,
                "name": f"SISR_{label_param}_parser_test_quick_test",
                "random_crop": False,
                "type": "SISR",
                "data_retention": 0.01,
            }
        ],
        "training_dataset": {
            "include_custom_trained": True,
            "include_pretrained": False,
            "label_param": label_param,
            "name": f"SISR_{label_param}_parser_withholding_{reserved_param_val}_train_quick_test",
            "random_crop": True,
            "reserved_param": reserved_param,
            "reserved_param_value": reserved_param_val,
            "type": "SISR",
            "data_retention": 0.01,
        },
        "validation_dataset": {
            "include_custom_trained": True,
            "include_pretrained": False,
            "label_param": label_param,
            "name": f"SISR_{label_param}_parser_withholding_{reserved_param_val}_val_quick_test",
            "random_crop": False,
            "reserved_param": reserved_param,
            "reserved_param_value": reserved_param_val,
            "type": "SISR",
            "data_retention": 0.01,
        },
    }
    opt_path = parser_opt_dir / f"{classifier_name}.yaml"
    with open(opt_path, "w") as f:
        yaml.dump(opt, f)


def make_parser_opt(label_param, reserved_param, reserved_param_val):
    classifier_name = (
        f"ConvNext_CNN_SISR_{label_param}_parser_withholding_{reserved_param_val}"
    )
    opt = {
        "classifier": {
            "cnn": {"type": "ConvNext"},
            "name": classifier_name,
            "train": {
                "batch_size": 16,
                "learning_rate_end": 4e-06,
                "learning_rate_start": 0.0005,
                "num_full_train_epochs": 15,
                "num_pretrain_epochs": 3,
            },
            "type": "CNN",
        },
        "evaluation_datasets": [
            {
                "include_custom_trained": True,
                "include_pretrained": True,
                "label_param": label_param,
                "name": f"SISR_{label_param}_parser_test",
                "random_crop": False,
                "type": "SISR",
            }
        ],
        "training_dataset": {
            "include_custom_trained": True,
            "include_pretrained": False,
            "label_param": label_param,
            "name": f"SISR_{label_param}_parser_withholding_{reserved_param_val}_train",
            "random_crop": True,
            "reserved_param": reserved_param,
            "reserved_param_value": reserved_param_val,
            "type": "SISR",
        },
        "validation_dataset": {
            "include_custom_trained": True,
            "include_pretrained": False,
            "label_param": label_param,
            "name": f"SISR_{label_param}_parser_withholding_{reserved_param_val}_val",
            "random_crop": False,
            "reserved_param": reserved_param,
            "reserved_param_value": reserved_param_val,
            "type": "SISR",
        },
    }
    opt_path = parser_opt_dir / f"{classifier_name}.yaml"
    with open(opt_path, "w") as f:
        yaml.dump(opt, f)


def make_parser_opts(label_param, reserved_param, reserved_param_val):
    make_parser_opt(label_param, reserved_param, reserved_param_val)
    make_parser_opt_quick_test(label_param, reserved_param, reserved_param_val)


# scale predictor withholding:
make_parser_opts("scale", "loss", "L1")
make_parser_opts("scale", "loss", "VGG_GAN")
make_parser_opts("scale", "architecture", "RCAN")
make_parser_opts("scale", "architecture", "SwinIR")
make_parser_opts("scale", "dataset", "flickr2k")
make_parser_opts("scale", "seed", 3)
# loss predictor withholding:
make_parser_opts("loss", "architecture", "RCAN")
make_parser_opts("loss", "architecture", "SwinIR")
make_parser_opts("loss", "dataset", "flickr2k")
make_parser_opts("loss", "seed", 3)
# arch predictor withholding:
make_parser_opts("architecture", "loss", "L1")
make_parser_opts("architecture", "loss", "VGG_GAN")
make_parser_opts("architecture", "dataset", "flickr2k")
make_parser_opts("architecture", "seed", 3)

# dataset predictor withholding:
make_parser_opt("dataset", "loss", "L1")
make_parser_opt("dataset", "loss", "VGG")
make_parser_opt("dataset", "architecture", "RCAN")
make_parser_opt("dataset", "architecture", "SwinIR")
make_parser_opt("dataset", "dataset", "flickr2k")
make_parser_opt("dataset", "seed", 3)
