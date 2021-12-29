# prnu_classifier.py
import numpy as np
import os
import PIL.Image
import bm3d
import tqdm
from utils.registry import CLASSIFIER_REGISTRY
from utils.misc import mkdir_unique
import database as db
import gzip
from pathlib import Path

PRNU = 'PRNU'
SAVED_NOISE_REDIDUALS_PATH = Path("classification/classifiers/saved_noise_residuals")
PRNU_CLASSIFIER_EXPERIMENTS_PATH = Path("classification/classifiers/experiments/PRNU")
ORDERED_FINGERPRINTS_FILENAME = 'ordered_fingerprints.npy'

def get_noise_residual(image, memoize=True):
    image_hash = hex(hash(image.data))
    noise_residual_filename = f'{image_hash}.npy'
    noise_residual_filepath = SAVED_NOISE_REDIDUALS_PATH / noise_residual_filename
    residual = None
    if noise_residual_filepath.exists():
        with gzip.GzipFile('file.npy.gz', 'r') as f:
            residual = np.load(f)
    else:
        denoised_image = bm3d.bm3d(image, 5)
        residual = (image - denoised_image).astype(np.int8)
        if memoize:
            with gzip.GzipFile('file.npy.gz', 'w') as f:
                np.save(f, residual)
    return residual

@CLASSIFIER_REGISTRY.register()
class PRNU:
    def __init__(self, ordered_labels, ordered_fingerprints, memoize=True):
        # takes parallel arrays of classes and fingerprints
        self.ordered_labels = ordered_labels
        # shape (c, h, w, 3)
        self.ordered_fingerprints = ordered_fingerprints
        self.memoize = memoize

    def __call__(self, images):
        n = len(images)
        c = len(ordered_fingerprints)
        noise_residuals = np.array([get_noise_residual(image, self.memoize) for image in images]) # shape (n, h, w, 3)
        # shape (n, c, h, w, 3)
        diff = self.ordered_fingerprints.unsqueeze(0) - self.noise_residuals.unsqueeze(1)
        fingerprint_distances = np.sum(diff**2, axis=(2,3,4)) # shape (c, n)
        fingerprint_predictions = fingerprint_distances.argmin(axis=1) # shape (n,)
        one_hot_encoding = np.zeros(shape=(n,c))
        one_hot_encoding[np.arange(n, dtype=int), fingerprint_predictions] = 1
        # Features are too large and burdensome, so just return None.
        return one_hot_encoding, None

    @staticmethod
    def train_and_save_classifier(classifier_opt, dataset):
        ordered_labels = dataset.ordered_labels
        memoize = classifier_opt['memoize']
        classifier_name = classifier_opt['name']
        param_to_predict = dataset.param_to_predict
        noise_residuals_sum = [0]*len(ordered_labels)
        noise_residuals_count = [0]*len(ordered_labels)
        for image, label, image_path in dataset:
            residual = get_noise_residual(image)
            label_index = ordered_labels.index(label)
            noise_residuals_sum[label_index] += residual
            noise_residuals_count[label_index] += 1
        
        ordered_fingerprints = np.array(noise_residuals_sum) / np.array(noise_residuals_count)

        classifier_dir = PRNU_CLASSIFIER_EXPERIMENTS_PATH / classifier_name
        mkdir_unique(PRNU_CLASSIFIER_EXPERIMENTS_PATH, classifier_name + "{}")
        ordered_fingerprints_path = classifier_dir / ORDERED_FINGERPRINTS_FILENAME
        with gzip.GzipFile(ordered_fingerprints_path, 'w') as f:
                np.save(f, ordered_fingerprints_path)
        # record this classifier (and its ordered classes) in the database.
        # TODO: record the dataset to the database.
        # TODO: record the classifier to the database.
        db.add_classifier_row(
                name=classifier_name,
                dataset_name=dataset.name,
                type=PRNU,
                opt=classifier_opt)
        return classifier_name

    @staticmethod
    def load_classifier(classifier_row):
        ordered_labels = classifier_row.ordered_labels
        ordered_fingerprints_path = os.path.join(path, 'ordered_fingerprints.npy')
        ordered_labels = np.load(ordered_labels_path)
        ordered_fingerprints = np.load(ordered_fingerprints_path)
        return PRNU_classifier(ordered_labels, ordered_fingerprints)
