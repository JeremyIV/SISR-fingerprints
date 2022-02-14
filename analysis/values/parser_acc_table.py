# TODO
# parser accuracy table
import database.api as db
from analysis.values.values_registry import VALUES_REGISTRY

parameters = ["scale", "loss", "architecture", "dataset"]

N_ITERATIONS = 1000
N_VOTES = 10
def get_10_vote_accuracy(results):
	num_correct = 0
	total = 0
	actual_classes = set(results.actual)
	for actual in actual_classes:
		results_from_actual_class = results[results.actual == actual]
		for _ in N_ITERATIONS:
			predictions = np.random.choice(results_from_actual_class, size=N_VOTES)
			prediction_counts = {}
			for pred in predictions:
				if pred not in prediction_counts:
					prediction_counts[pred] = 0
				prediction_counts[pred] += 1
			modal_prediction = max(prediction_counts.keys(), key=prediction_counts.get)
			total += 1
			if modal_prediction == actual:
				num_correct += 1
	return num_correct / total

@VALUES_REGISTRY.register()
def model_parsing_table():
	values = {}
	for param in parameters:
		cnn_type = 'ConvNext'
		param_parsers_query = (
			'select sd.reserved_param, sd.reserved_param_value, c.name'
			'from SISR_dataset sd'
			'inner join dataset d on d.id = sd.dataset_id'
			'inner join classifier c on c.training_dataset_id = d.id'
			r'where c.name like "ConvNext%"'
			f'and sd.label_param = {param}')
		result = db.read_sql_query(param_parsers_query)
		for reserved_param, reserved_param_val, classifier_name in result.itertuples():
			# get the predicted and actual classes on the reserved models
			assert reserved_param in parameters
			prediction_query = (
				'select predicted, actual'
				'from sisr_analysis'
				'where classifier_name = :classifier_name'
				f'and {reserved_param} = :reserved_param_val'
				r'and generator_name not like "%-pretrained"')
			params = {
				'classifier_name': classifier_name,
				'reserved_param_val': reserved_param_val
			}
			parser_results = db.read_sql_query(prediction_query, params)
			parser_accuracy = (parser_results.predicted == parser_results.actual).mean()
			parser_10_vote_accuracy = get_10_vote_accuracy(parser_results)
			parser_acc_value_name = f"TODO"
			parser_10_vote_acc_value_name = f"TODO"
			formatted_parser_accuracy = f"{parser_accuracy*100:.01f}"
			formatted_parser_10_vote_accuracy = f"{f"{parser_accuracy*100:.01f}"*100:.01f}"
			values[parser_acc_value_name] = formatted_parser_accuracy
			values[parser_10_vote_acc_value_name] = formatted_parser_10_vote_accuracy