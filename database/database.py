# TODO
from collections import namedtuple
# utilities for interacting with the database.

# make a namedtuple for each type of row in the database

def update_image_patch(
        generator=None,
        image_path=None,
        crop=None,
        acutance=None,
        psnr=None,
        lpips=None):
    pass

def add_prediction(
        classifier_name=None,
        image_patch_id=None,
        dataset_name=None,
        actual_label=None,
        predicted_label=None,
        class_probabilities=None,
        feature=None):
    pass

def add_dataset_row(
        type=None, 
        name=None,
        is_test=None,
        label_param=None,
        ordered_labels=None, 
        test_param=None,
        opt=None):
    pass

def get_classifier_row(name):
    pass
    # TODO: return an easydict with all the fields from the classifier's row

def add_classifier_row(
        name=None,
        type=None,
        dataset_slice=None,
        test_param_value=None,
        param_to_predict=None,
        ordered_labels=None,
        opt=None):
    pass