# TODO
from collections import namedtuple
# utilities for interacting with the database.

# make a namedtuple for each type of row in the database

def get_image_patch(
        image_path,
        crop_top,
        crop_bottom,
        crop_left,
        crop_right):
    pass

def add_image_patch(
        image_path,
        crop_top,
        crop_bottom,
        crop_left,
        crop_right,
        acutance,
        psnr,
        lpips):
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