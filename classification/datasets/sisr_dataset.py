# sisr.py
from utils.registry import DATASET_REGISTRY
from torch.utils.data import Dataset
from pathlib import Path
from easydict import EasyDict as edict


def is_in_dataset(
    model_params,
    is_train,
    reserved_param,
    reserved_param_value,
    include_pretrained,
    include_custom_trained,
):
    if not include_pretrained and model_params.pretrained:
        return False
    if not include_custom_trained and not model_params.pretrained:
        return False
    if is_train and reserved_param is not None:
        return model_params[reserved_param] == reserved_param_value
    return True


def get_params(sisr_model):
    pretrained = False
    if sisr_model.startswith("pretrained-"):
        pretrained = True
        first_hyphen_index = sisr_model.index("-")
        sisr_model = sisr_model[first_hyphen_index + 1 :]
    architecture, scale, dataset, loss, seed = sisr_model.split("-")
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
TRAIN_SPLIT_PATH = Path("classification/datasets/data/SISR/train_split.txt")
TEST_SPLIT_PATH = Path("classification/datasets/data/SISR/test_split.txt")


@DATASET_REGISTRY.register()
class SISR(Dataset):
    def __init__(
        self,
        name=None,
        label_param=None,
        is_train=True,
        patch_size=299,
        random_crop=True,
        reserved_param=None,
        reserved_param_value=None,
        include_pretrained=False,
        include_custom_trained=True,
    ):
        if is_reserved:
            assert not is_train, "Reserved SISR models cannot be used for training."
        self.samples = []
        self.name = name
        self.label_param = label_param
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.is_train = is_train
        self.reserved_param = reserved_param
        self.reserved_param_value = reserved_param_value
        self.include_custom_trained = include_custom_trained

        split_filepath = TRAIN_SPLIT_PATH if is_train else TEST_SPLIT_PATH
        split = {line.rstrip() for line in open(split_filepath).readlines()}

        for sisr_model_dir in SISR_DATASET_PATH.iterdir():
            if not sisr_model_dir.is_dir():
                continue
            sisr_model = sisr_model_dir.stem
            params = get_params(sisr_model)
            if not is_in_dataset(
                params,
                is_train,
                reserved_param,
                reserved_param_value,
                include_pretrained,
                include_custom_trained,
            ):
                continue
            label = sisr_model if label_param is None else params[label_param]
            for image_path in sisr_model_dir.iterdir():
                if image_path not in split:
                    continue
                self.samples.append((image_path, label))

        self.ordered_labels = list(sorted(set(label for path, label in self.samples)))

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
            "is_train": self.is_train,
            "random_crop": self.random_crop,
            "reserved_param": self.reserved_param,
            "reserved_param_value": self.reserved_param_value,
            "include_pretrained": self.include_pretrained,
            "include_custom_trained": self.include_custom_trained,
        }
        db.idempotent_insert_unique_row(
            "SISR_dataset",
            {
                "type": "SISR",
                "name": self.name,
                "is_train": self.is_train,
                "ordered_labels": self.ordered_labels,
                "opt": opt,
                "label_param": self.label_param,
                "reserved_param": self.reserved_param,
                "reserved_param_value": self.reserved_param_value,
                "include_pretrained": self.include_pretrained,
                "include_custom_trained": self.include_custom_trained,
            },
        )
