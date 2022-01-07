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
    description="Train an image classifier based on the configuration options in the given file."
)
parser.add_argument("-opt", required=True, type=Path)

args = parser.parse_args()
opt = yaml.load(open(args.opt), Loader=Loader)

classifier_opt = opt["classifier"]
training_dataset_opt = opt["training_dataset"].copy()
training_dataset_opt["phase"] = "train"
validation_dataset_opt = opt.get("validation_dataset")
evaluation_dataset_opts = opt["evaluation_datasets"]
evaluation_opt = opt.get("evaluation")

training_dataset = datasets.get_dataset(training_dataset_opt)
val_dataset = None
if validation_dataset_opt is not None:
    validation_dataset_opt = validation_dataset_opt.copy()
    validation_dataset_opt["phase"] = "val"
    val_dataset = datasets.get_dataset(validation_dataset_opt)
classifier_name = classifiers.train_and_save_classifier(
    classifier_opt, training_dataset, val_dataset
)
classifier = classifiers.load_classifier(classifier_name)

for evaluation_dataset_opt in evaluation_dataset_opts:
    evaluation_dataset_opt = evaluation_dataset_opt.copy()
    evaluation_dataset_opt["phase"] = "test"
    dataset = datasets.get_dataset(evaluation_dataset_opt)
    classifiers.evaluate(classifier, dataset, evaluation_opt)
