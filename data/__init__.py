# __init__.py
from utils.registry import DATASET_REGISTRY

# each dataset should fulfill the following interface:
# constructor takes in its args from the options
# the dataset has an ordered_classes attribute
# the dataset yeilds triples of (image, label, filepath)

def get_dataset(opt):
	dataset_type = opt.pop('type')
	return DATASET_REGISTRY.get(dataset_type)(**opt)