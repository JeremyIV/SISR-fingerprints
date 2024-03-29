# __init__.py
from classification.utils.registry import DATASET_REGISTRY
import classification.datasets.RAISE_dataset, classification.datasets.sisr_dataset

# each dataset should fulfill the following interface:
# constructor takes in its args from the options
# the dataset has attributes
#   ordered_labels
#   name
#   add_to_database(self)
# the dataset yeilds triples of (image, label, metadata)
# image: a PIL image
# label: a string label. should be a string from ordered_labels
# metadata:
# crop: a named tuple with the same order as PIL crop() args
# ground_truth_path - path to the corresponding ground truth (HR) image.
# image_path


def get_dataset(opt, phase=None):
    """Returns a dataset created from the given options dictionary.

    args:
    opt: the dictionary of options
    phase (optional): one of {'train', 'val', 'test'}.
        Specifies which role this dataset will play.
    """
    opt = opt.copy()
    if phase is not None:
        if "phase" in opt:
            assert opt["phase"] == phase, (
                f"dataset of phase type {opt['phase']} "
                + f"cannot be used for phase {phase}"
            )
        opt["phase"] = phase
    dataset_type = opt.pop("type")
    return DATASET_REGISTRY.get(dataset_type)(**opt)
