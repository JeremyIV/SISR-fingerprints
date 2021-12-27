# __init__.py
import databse as db # for interacting with the sqlite3 analysis database
import importlib

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


def load_classifier(name):
	classifier_row = db.get_classifier_row(name)
	return CLASSIFIER_REGISTRY.get(classifier_row.type).load_classifier(classifier_row.path)

def evaluate(classifier, evaluation_dataset_opt):
	# TODO: evaluate the classifier on the provided dataset.
	# save the analysis data in the sqlite3 database.
	# create the evaluation dataset
	# for each image, label in the evaluation dataset:
		# pass the image through the classifier
		# get the classifier's prediction
		# save the following data to the analysis databse
			# classifier ID
			# dataset ID
			# image path
			# generating model (same as actual label for attribution)
			# actual label
			# predicted label
			# probabilities
			# features (optionally)
	