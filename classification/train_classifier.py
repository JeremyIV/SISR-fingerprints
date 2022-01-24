# train_classifier.py
# takes one argument, -opt
# and trains the classifier based on that options file
import argparse
import yaml
from yaml import Loader
from classification import classifiers
from classification import datasets
from classification.utils.registry import CLASSIFIER_REGISTRY

from pathlib import Path

parser = argparse.ArgumentParser(
    description="Train an image classifier based on the configuration "
    + "options in the given file."
)
parser.add_argument(
    "-opt",
    required=True,
    type=Path,
    help="Path to the yaml config file containing the options for how to "
    + "train and evaluate the classifier. e.g. "
    + "classificaiton/options/Xception_CNN_SISR_all_models.yml",
)

args = parser.parse_args()
opt = yaml.load(open(args.opt), Loader=Loader)


def train_and_save_classifier(
    classifier_opt, training_dataset_opt, validation_dataset_opt=None
):
    training_dataset = datasets.get_dataset(training_dataset_opt, phase="train")

    val_dataset = (
        None
        if validation_dataset_opt is None
        else datasets.get_dataset(validation_dataset_opt, phase="val")
    )

    classifier_name = classifiers.train_and_save_classifier(
        classifier_opt, training_dataset, val_dataset
    )
    return classifier_name


def evaluate_classifier(classifier_name, evaluation_dataset_opts, evaluation_opt=None):
    classifier = classifiers.load_classifier(classifier_name)
    for evaluation_dataset_opt in evaluation_dataset_opts:
        dataset = datasets.get_dataset(evaluation_dataset_opt, phase="test")
        classifiers.evaluate(classifier, dataset, evaluation_opt)


classifier_name = train_and_save_classifier(
    classifier_opt=opt["classifier"],
    training_dataset_opt=opt["training_dataset"],
    validation_dataset_opt=opt.get("validation_dataset"),
)

evaluate_classifier(
    classifier_name,
    evaluation_dataset_opts=opt["evaluation_datasets"],
    evaluation_opt=opt.get("evaluation"),
)
