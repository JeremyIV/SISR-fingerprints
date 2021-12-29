# __init__.py
from classificaiton.utils.registry import CLASSIFIER_REGISTRY
from classification.utils import image_patch
import database

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

def train_and_save_classifier(classifier_opt, dataset):
    dataset.add_to_database()
    classifier_opt = classifier_opt.deepcopy()
    classifier_type = classifier_opt.pop('type')
    return CLASSIFIER_REGISTRY.get(classifier_type).train_and_save_classifier(classifier_opt, dataset)

def load_classifier(name):
    classifier_row = db.get_classifier_row(name)
    return CLASSIFIER_REGISTRY.get(classifier_row.type).load_classifier(classifier_row.path)

def evaluate(classifier, dataset, evaluation_opt):
    # NOTE: evaluation_opt is currently unused, but may be useful in the future.
    dataset.add_to_database()
    for image, label, sample_metadata in dataset:
        image_patch_id = image_patch.update_image_patch(image, sample_metadata)
        probabilities, feature = classifier(image)
        predicted_label = classifier.ordered_labels[np.argmax(probabilities)]

        db.add_prediction(
            classifier_name=classifier.name,
            image_patch_id=image_patch_id,
            dataset_name=dataset.name,
            actual_label=label,
            predicted_label=predicted_label,
            class_probabilities=probabilities,
            feature=feature)
