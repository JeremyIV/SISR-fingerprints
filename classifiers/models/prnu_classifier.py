# prnu_classifier.py
import numpy as np
import os
import PIL.Image
import bm3d
import tqdm
from utils.registry import CLASSIFIER_REGISTRY
import database as db

def get_noise_residual(image):
	# TODO: memoize this to the hard disk somehow.
	# hash the image, stringify the hash, and
	# use this as a filename for the noise residual.
	# if this file exists, just load it.
	# otherwise, compute the noise residual and save it.
	# then return the noise residual

@CLASSIFIER_REGISTRY.register()
class PRNU_classifier:
	def __init__(self, ordered_classes, ordered_fingerprints):
		# takes parallel arrays of classes and fingerprints
		self.ordered_classes = ordered_classes
		# shape (c, h, w, 3)
		self.ordered_fingerprints = ordered_fingerprints

	def __call__(self, images):
		# TODO: get the noise residual for each of the images
		n = len(images)
		c = len(ordered_fingerprints)
		noise_residuals = # TODO shape (n, h, w, 3)
		closest_labels = None # shape (n,)
		# shape (n, c)
		fingerprint_distances = np.sum((self.ordered_fingerprints.unsqueeze(0) - self.noise_residuals.unsqueeze(1))**2, axis=(2,3,4))# TODO shape (c, n)
		# shape (n,)
		fingerprint_predictions = fingerprint_distances.argmin(axis=1)
		one_hot_encoding = np.zeros(shape=(n,c))
		one_hot_encoding[np.arange(n, dtype=int), fingerprint_predictions] = 1
		# Features are too large and burdensome, so just return None.
		return one_hot_encoding, None

	@staticmethod
	def train_and_save_classifier(classifier_opt, dataset):
		pass
		# TODO: get the ordered_classes from the dataset
		# TODO: create a total_noise_residual for each possible label
		# TODO: create a num_samples for each possible label
		# for each image, label, image_path in the dataset:
			# compute the noise residual
			# add it to the appropriate total based on the label
			# increment the num_samples appropriately
		# divide the totals by the num_samples to get ordered_fingerprints
		# turn the ordered_fingerprints into a numpy array
		# save the ordered_fingerprints somewhere, according to the options
		# record this classifier (and its ordered classes) in the databse.

	@staticmethod
	def load_classifier(classifier_row):
		ordered_classes = classifier_row.ordered_classes
		ordered_fingerprints_path = os.path.join(path, 'ordered_fingerprints.npy')
		ordered_classes = np.load(ordered_classes_path)
		ordered_fingerprints = np.load(ordered_fingerprints_path)
		return PRNU_classifier(ordered_classes, ordered_fingerprints)
# each classifier is a class added to the CLASSIFIER_REGISTRY
# with the following functions:
# train_and_save_classifier(classifier_opt, dataset) ->
#	trains classifier, saves it, and saves its metadata to database
# load_classifier(path)
#	loads the classifier and returns it.
#	Model takes in a tensor of images of shape (n, 3, w, h)
#	and returns class labels of shape (n, c) 
#	where c is the number of possible class labels.
#	optionally, also returns features of shape (n, f)