# __init__.py
from utils.registry import DATASET_REGISTRY

# each dataset should fulfill the following interface:
# constructor takes in its args from the options
# the dataset has attributes
#	ordered_labels
#	name
#	add_to_database(self)
# the dataset yeilds triples of (image, label, metadata)
	# image: a PIL image
	# label: a string label. should be a string from ordered_labels
	# metadata:
		# crop: a named tuple with the same order as PIL crop() args
		# ground_truth_path - path to the corresponding ground truth (HR) image.
		# image_path

def get_dataset(opt):
	dataset_type = opt.pop('type')
	return DATASET_REGISTRY.get(dataset_type)(**opt)