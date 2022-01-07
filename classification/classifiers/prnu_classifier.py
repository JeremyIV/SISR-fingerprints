# prnu_classifier.py
import numpy as np
import os
import PIL.Image
import bm3d
from tqdm import tqdm
from classification.utils.registry import CLASSIFIER_REGISTRY
from classification.utils.misc import mkdir_unique
import database.api as db
import gzip
from pathlib import Path
from classification.utils.image_patch import get_patch_hash

PRNU = "PRNU"
SAVED_NOISE_REDIDUALS_PATH = Path("classification/classifiers/saved_noise_residuals")
PRNU_CLASSIFIER_EXPERIMENTS_PATH = Path("classification/classifiers/experiments/PRNU")
ORDERED_FINGERPRINTS_FILENAME = "ordered_fingerprints.npy.gz"


def get_noise_residual(image, memoize=True):
    image_hash = get_patch_hash(image)
    noise_residual_filename = f"{image_hash}.npy.gz"
    noise_residual_filepath = SAVED_NOISE_REDIDUALS_PATH / noise_residual_filename
    residual = None
    if noise_residual_filepath.exists():
        with gzip.GzipFile(noise_residual_filepath, "r") as f:
            residual = np.load(f)
    else:
        denoised_image = bm3d.bm3d(image, 5)
        residual = (image - denoised_image).astype(np.int8)
        if memoize:
            with gzip.GzipFile(noise_residual_filepath, "w") as f:
                np.save(f, residual)
    return residual


@CLASSIFIER_REGISTRY.register()
class PRNU:
    def __init__(self, name, ordered_labels, ordered_fingerprints, memoize=True):
        # takes parallel arrays of classes and fingerprints
        self.name = name
        self.ordered_labels = ordered_labels
        # shape (c, h, w, 3)
        self.ordered_fingerprints = ordered_fingerprints
        self.memoize = memoize

    def __call__(self, image):
        c = len(self.ordered_fingerprints)
        noise_residual = get_noise_residual(image, self.memoize)  # shape (h, w, 3)
        # shape (c, h, w, 3)
        diff = self.ordered_fingerprints - noise_residual[np.newaxis, :, :, :]
        fingerprint_distance = np.sum(diff ** 2, axis=(1, 2, 3))  # shape (c,)
        fingerprint_prediction = fingerprint_distance.argmin()  # scalar
        one_hot_encoding = np.zeros(shape=(c))
        one_hot_encoding[fingerprint_prediction] = 1
        # Features are too large and burdensome, so just return None.
        return one_hot_encoding, None

    # third argument is val_dataset, which is unused.
    @staticmethod
    def train_and_save_classifier(classifier_opt, dataset, _=None):
        training_dataset_id = db.get_unique_row("dataset", {"name": dataset.name}).id
        ordered_labels = dataset.ordered_labels
        memoize = classifier_opt.get("memoize", True)
        classifier_name = classifier_opt["name"]
        label_param = dataset.label_param  # TODO: do someting with this label param?
        noise_residuals_sum = [0] * len(ordered_labels)
        noise_residuals_count = [0] * len(ordered_labels)
        print("Training PRNU classifier!")
        for image, label, metadata in tqdm(dataset):
            residual = get_noise_residual(image, memoize=memoize)
            label_index = ordered_labels.index(label)
            noise_residuals_sum[label_index] += residual
            noise_residuals_count[label_index] += 1

        ordered_fingerprints = (
            np.array(noise_residuals_sum)
            / np.array(noise_residuals_count)[:, np.newaxis, np.newaxis, np.newaxis]
        )

        classifier_dir = PRNU_CLASSIFIER_EXPERIMENTS_PATH / classifier_name
        mkdir_unique(PRNU_CLASSIFIER_EXPERIMENTS_PATH, classifier_name + "{}")
        ordered_fingerprints_path = classifier_dir / ORDERED_FINGERPRINTS_FILENAME
        with gzip.GzipFile(ordered_fingerprints_path, "w") as f:
            np.save(f, ordered_fingerprints)
        db.idempotent_insert_unique_row(
            "classifier",
            {
                "training_dataset_id": training_dataset_id,
                "name": classifier_name,
                "path": ordered_fingerprints_path,
                "type": "PRNU",
                "opt": classifier_opt,
            },
        )
        return classifier_name

    @staticmethod
    def load_classifier(classifier_row):
        ordered_labels = db.get_unique_row(
            "dataset", {"id": classifier_row.training_dataset_id}
        ).ordered_labels
        with gzip.GzipFile(classifier_row.path, "r") as f:
            ordered_fingerprints = np.load(f, allow_pickle=True)
        return PRNU(classifier_row.name, ordered_labels, ordered_fingerprints)
