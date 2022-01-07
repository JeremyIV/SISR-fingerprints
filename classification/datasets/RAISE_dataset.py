from classification.utils.registry import DATASET_REGISTRY
from torch.utils.data import Dataset
import database.api as db
from pathlib import Path
from PIL import Image
from classification.datasets.image_patch_metadata import ImagePatchMetadata

CROP_SIZE = 299
RAISE_PATH = Path("classification/datasets/data/RAISE")


@DATASET_REGISTRY.register()
class RAISE(Dataset):
    def __init__(self, name=None, phase=None):
        # TODO: replace is_train with phase
        self.name = name
        assert phase in {
            "train",
            "test",
        }, f"phase must be either 'train' or 'test', not {phase}"
        self.is_train = phase == "train"
        self.samples = []
        self.label_param = None
        self.generators = set()
        for generator_path in RAISE_PATH.iterdir():
            label = generator_path.stem
            self.generators.add(label)
            sorted_image_paths = sorted(generator_path.iterdir())
            split_index = 412
            split_image_paths = (
                sorted_image_paths[:split_index]
                if is_train
                else sorted_image_paths[split_index:]
            )
            for image_path in split_image_paths:
                self.samples.append((image_path, label))

        self.ordered_labels = list(sorted(set(label for path, label in self.samples)))

    def __len__(self):
        return len(self.samples)

    def add_to_database(self):
        phase = "train" if self.is_train else "test"
        opt = {"name": self.name, "type": "RAISE", "phase": phase}
        dataset_id = db.idempotent_insert_unique_row(
            "dataset",
            {
                "type": "RAISE",
                "name": self.name,
                "phase": phase,
                "ordered_labels": self.ordered_labels,
                "opt": opt,
            },
        )
        for generator in self.generators:
            generator_id = db.idempotent_insert_unique_row(
                "generator", {"name": generator, "type": "RAISE", "parameters": {}}
            )
            db.idempotent_insert_unique_row(
                "generators_in_dataset",
                {
                    "dataset_id": dataset_id,
                    "generator_id": generator_id,
                },
            )
        return dataset_id

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path)
        width, height = image.size
        crop_upper = (height - CROP_SIZE) // 2
        crop_lower = crop_upper + CROP_SIZE
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
