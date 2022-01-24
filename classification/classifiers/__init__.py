# __init__.py
from classification.utils.registry import CLASSIFIER_REGISTRY
import classification.classifiers.prnu_classifier
import classification.classifiers.cnn_classifier
from classification.utils import image_patch
import database.api as db
import numpy as np
from tqdm import tqdm

# each classifier is a class added to the CLASSIFIER_REGISTRY
# with the following functions:
# train_and_save_classifier(classifier_opt, dataset) ->
#   trains classifier, saves it, and saves its metadata to database
# load_classifier(path)
#   loads the classifier and returns it.
#   Model's __call__ function should take in a PIL image.
#   and returns class labels of shape (c,)
#   where c is the number of possible class labels.
#   optionally, also returns data representing some feature embedding.
#       This feature embedding will be stored as a binary blob in the database.


def train_and_save_classifier(classifier_opt, train_dataset, val_dataset=None):
    """Trains a classifier, saves the model to experiments/, and saves all the
    data needed to recreate the classifier to the database.
    """
    train_dataset.add_to_database()
    if val_dataset is not None:
        val_dataset.add_to_database()
    classifier_opt = classifier_opt.copy()
    classifier_type = classifier_opt["type"]
    result = CLASSIFIER_REGISTRY.get(classifier_type).train_and_save_classifier(
        classifier_opt, train_dataset, val_dataset
    )
    db.con.commit()
    return result


def load_classifier(name):
    classifier_row = db.get_unique_row("classifier", {"name": name})
    return CLASSIFIER_REGISTRY.get(classifier_row.type).load_classifier(classifier_row)


def evaluate(classifier, dataset, evaluation_opt):
    # TODO: this feels cluttery here. Is there a better place for it?
    if hasattr(classifier, "patch_size"):
        dataset.patch_size = classifier.patch_size
    dataset_id = dataset.add_to_database()
    classifier_id = db.get_unique_row("classifier", {"name": classifier.name}).id
    print(f"Evaluating on dataset {dataset.name}")
    for image, label, sample_metadata in tqdm(dataset):
        image_patch_id = image_patch.update_image_patch(image, sample_metadata)
        probabilities, feature = classifier(image)
        predicted_label = classifier.ordered_labels[np.argmax(probabilities)]

        db.idempotent_insert_unique_row(
            "prediction",
            {
                "classifier_id": classifier_id,
                "image_patch_id": image_patch_id,
                "dataset_id": dataset_id,
                "actual": label,
                "predicted": predicted_label,
                "class_probabilities": probabilities,
                "feature": feature,
            },
        )
        db.con.commit()
