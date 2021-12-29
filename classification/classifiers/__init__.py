# __init__.py
from classificaiton.utils import CLASSIFIER_REGISTRY
import database

# each classifier is a class added to the CLASSIFIER_REGISTRY
# with the following functions:
# train_and_save_classifier(classifier_opt, dataset) ->
#   trains classifier, saves it, and saves its metadata to database
# load_classifier(path)
#   loads the classifier and returns it.
#   Model takes in a tensor of images of shape (n, 3, w, h)
#   and returns class labels of shape (n, c) 
#   where c is the number of possible class labels.
#   optionally, also returns features of shape (n, f)

def train_and_save_classifier(classifier_opt, dataset):
	classifier_opt = classifier_opt.deepcopy()
	classifier_type = classifier_opt.pop('type')
	return CLASSIFIER_REGISTRY.get(classifier_type).train_and_save_classifier(classifier_opt, dataset)

def load_classifier(name):
	classifier_row = db.get_classifier_row(name)
	return CLASSIFIER_REGISTRY.get(classifier_row.type).load_classifier(classifier_row.path)

def evaluate(classifier, dataset, evaluation_opt):
	# TODO: get the classifier ID
	# TODO: get the dataset ID
	classifier_id = None # TODO
	# TODO: evaluate the classifier on the provided dataset.
	# save the analysis data in the sqlite3 database.
	# create the evaluation dataset
	# for each image, label in the evaluation dataset:
	for image, label, image_path in dataset:
		# TODO:
		# load the image
		# get the coords needed to center crop the image
		# look up this image patch in the database
			# if it exists, get its ID
			# if it doesn't exist, create it and get its ID
		# pass the image through the classifier
		probabilities, feature = classifier(image)
		# get the classifier's prediction
		predicted_label = classifier.ordered_labels[np.argmax(probabilities)]

		# save the following data to the analysis databse
			# classifier ID
			# image patch ID
			# actual label
			# predicted label
			# probabilities
			# features (optionally)