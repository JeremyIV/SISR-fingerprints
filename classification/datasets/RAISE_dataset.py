from classification.utils.registry import DATASET_REGISTRY
from torch.utils.data import Dataset
import database as db
from pathlib import Path
from PIL import Image

CROP_SIZE = 299
RAISE_PATH = Path("classification/datasets/data/RAISE")

DATASET_REGISTRY.register()


class RAISE(Dataset):
    def __init__(self, name=None, is_train=True):
        self.name = name
        self.samples = []
        for generator_path in RAISE_PATH.iterdir():
            label = generator_path.stem
            sorted_image_paths = sorted(generator_path.iterdir())
            split_index = 412
            split_image_paths = (
                sorted_image_paths[:split_index]
                if is_train
                else sorted_image_paths[split_index:]
            )
            for image_path in split_image_paths:
                self.samples.append(image_path, label)

        self.ordered_labels = list(sorted(set(label for path, label in self.samples)))

    def __len__(self):
        return len(self.samples)

    def add_to_database(self):
        db.add_or_get_dataset(
            type="RAISE",
            name=self.name,
            is_train=self.is_train,
            ordered_labels=self.ordered_labels,
            opt={"name": self.name, "type": "RAISE", "is_train": self.is_train},
        )

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = PIL.image.open(image_path)
        width, height = image.size
        crop_upper = (height - CROP_SIZE) // 2
        crop_lower = crop_top + CROP_SIZE
        crop_left = (width - CROP_SIZE) // 2
        crop_right = crop_left + CROP_SIZE
        crop = db.CropCoords(
            left=crop_left, upper=crop_upper, right=crop_right, lower=crop_lower
        )
        image = image.crop(crop)
        metadata = ImagePatchMetadata(
            image_path=image_path, crop=crop, ground_truth_path=None
        )
        return image, label, metadata
