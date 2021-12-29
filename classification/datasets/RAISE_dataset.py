from classification.utils.registry import DATASET_REGISTRY
from torch.utils.data import Dataset
import database as db

def is_in_dataset(sisr_model, is_test, test_param):
	pass
	# TODO

DATASET_REGISTRY.register()
class RAISE_dataset(Dataset):
	def __init__(self, name=None, is_training=True):
		self.name = name
		self.samples = [] # TODO
		# for each camera model in the RAISE dataset
			# label = camera_model
			# get the images in that model folder in sorted order.
			# take the first half as training
			# if is_test: take the second half, else take the first half
			# for each image in the relevant split:
				# make the image path
				# image_paths.append(image_path, label)

		self.ordered_labels = list(sorted(set(label for path, label in self.samples)))

	def __len__(self):
		return len(self.samples)

	def add_to_database(self):
		pass # TODO

	def __getitem__(self, index):
		image_path, label = self.samples[index]
		# TODO: load the image
		# center crop the image
		return image, label, image_path
