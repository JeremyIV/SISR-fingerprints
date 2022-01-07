# sisr.py
from classification.utils.registry import DATASET_REGISTRY
from torch.utils.data import Dataset
from pathlib import Path
from easydict import EasyDict as edict
import pdb
import re
import database.api as db
from PIL import Image
from classification.datasets.image_patch_metadata import ImagePatchMetadata
import random

seed_regex = re.compile(r"s(\d+)")
scale_regex = re.compile(r"x(\d+)")


def is_in_dataset(
    model_params,
    is_test,
    reserved_param,
    reserved_param_value,
    include_pretrained,
    include_custom_trained,
):
    if not include_pretrained and model_params.pretrained:
        return False
    if not include_custom_trained and not model_params.pretrained:
        return False
    if not is_test and reserved_param is not None:
        return model_params[reserved_param] == reserved_param_value
    return True


def get_params(sisr_model):
    pretrained = False
    pretrained_suffix = "-pretrained"
    if sisr_model.endswith(pretrained_suffix):
        pretrained = True
        sisr_model = sisr_model[: -len(pretrained_suffix)]
    split = sisr_model.split("-")
    if len(split) != 5:
        return None  # not a valid sisr_model directory!
    architecture, dataset, scale, loss, seed = split
    seed_match = seed_regex.match(seed)
    seed = None if seed_match is None else int(seed_match.group(1))
    scale_match = scale_regex.match(scale)
    scale = None if scale_match is None else int(scale_match.group(1))
    return edict(
        {
            "pretrained": pretrained,
            "architecture": architecture,
            "scale": scale,
            "dataset": dataset,
            "loss": loss,
            "seed": seed,
        }
    )


SISR_DATASET_PATH = Path("classification/datasets/data/SISR")

SPLIT_PATHS = {
    "train": SISR_DATASET_PATH / "train-split.txt",
    "val": SISR_DATASET_PATH / "val-split.txt",
    "test": SISR_DATASET_PATH / "test-split.txt",
}


@DATASET_REGISTRY.register()
class SISR(Dataset):
    def __init__(
        self,
        name=None,
        label_param=None,
        phase=None,
        patch_size=299,
        random_crop=True,
        reserved_param=None,
        reserved_param_value=None,
        include_pretrained=False,
        include_custom_trained=True,
        data_retention=1,
    ):
        self.samples = []
        self.name = name
        self.label_param = label_param
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.phase = phase
        self.reserved_param = reserved_param
        self.reserved_param_value = reserved_param_value
        self.include_pretrained = include_pretrained
        self.include_custom_trained = include_custom_trained
        self.data_retention = data_retention
        self.generators = {}

        split_filepath = SPLIT_PATHS[phase]
        split = {line.rstrip() for line in open(split_filepath).readlines()}

        for sisr_model_dir in SISR_DATASET_PATH.iterdir():
            if not sisr_model_dir.is_dir():
                continue
            sisr_model = sisr_model_dir.stem
            params = get_params(sisr_model)
            if params is None:  # indicates not a valid SISR model directory
                continue
            self.generators[sisr_model] = params
            if not is_in_dataset(
                params,
                phase == "test",
                reserved_param,
                reserved_param_value,
                include_pretrained,
                include_custom_trained,
            ):
                continue
            label = sisr_model if label_param is None else params[label_param]
            for image_path in sisr_model_dir.iterdir():
                if image_path.name not in split:
                    continue
                self.samples.append((image_path, label))

        self.ordered_labels = list(sorted(set(label for path, label in self.samples)))
        random.shuffle(self.samples)
        if self.data_retention < 1:
            cutoff_index = int(len(self.samples) * self.data_retention)
            self.samples = self.samples[:cutoff_index]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path)
        width, height = image.size
        if self.random_crop:
            crop_upper = np.random.randint(height - self.patch_size)
            crop_lower = crop_upper + self.patch_size
            crop_left = np.random.randint(width - self.patch_size)
            crop_right = crop_left + self.patch_size
        else:  # center crop
            crop_upper = (height - self.patch_size) // 2
            crop_lower = crop_upper + self.patch_size
            crop_left = (width - self.patch_size) // 2
            crop_right = crop_left + self.patch_size
        crop = db.CropCoords(
            left=crop_left, upper=crop_upper, right=crop_right, lower=crop_lower
        )
        image = image.crop(crop)
        metadata = ImagePatchMetadata(
            image_path=image_path, crop=crop, ground_truth_path=None
        )
        return image, label, metadata

    def add_to_database(self):
        opt = {
            "name": self.name,
            "type": "SISR",
            "label_param": self.label_param,
            "phase": self.phase,
            "random_crop": self.random_crop,
            "reserved_param": self.reserved_param,
            "reserved_param_value": self.reserved_param_value,
            "include_pretrained": self.include_pretrained,
            "include_custom_trained": self.include_custom_trained,
            "data_retention": self.data_retention,
        }
        dataset_id = db.idempotent_insert_unique_row(
            "SISR_dataset",
            {
                "type": "SISR",
                "name": self.name,
                "phase": self.phase,
                "ordered_labels": self.ordered_labels,
                "opt": opt,
                "label_param": self.label_param,
                "reserved_param": self.reserved_param,
                "reserved_param_value": self.reserved_param_value,
                "include_pretrained": self.include_pretrained,
                "include_custom_trained": self.include_custom_trained,
            },
        )

        for sisr_model, params in self.generators.items():
            generator_id = db.idempotent_insert_unique_row(
                "SISR_generator",
                {
                    "type": "SISR",
                    "name": sisr_model,
                    "parameters": params,
                    "architecture": params.architecture,
                    "dataset": params.dataset,
                    "scale": params.scale,
                    "loss": params.loss,
                    "seed": params.seed,
                },
            )
            db.idempotent_insert_unique_row(
                "generators_in_dataset",
                {"dataset_id": dataset_id, "generator_id": generator_id},
            )

        return dataset_id
