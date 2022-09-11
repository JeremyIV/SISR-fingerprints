import database.api as db
from analysis.values.values_registry import VALUES_REGISTRY
from analysis.values.val_utils import fmt

query = (
    "select classifier_name, classifier_opt, avg(predicted = actual) as acc"
    " from analysis "
    ' where train_dataset_name = "SISR_all_models_train"'
    " and classifier_type = 'CNN'"
    " group by classifier_name, classifier_opt"
    " order by avg(predicted = actual) desc"
)

latex_friendly = {"ResNet50": "Resnet"}


@VALUES_REGISTRY.register()
def architecture_comparison():
    result = db.read_and_decode_sql_query(query)
    values = {}
    for row in result.itertuples(index=False):
        cnn_type = row.classifier_opt["cnn"]["type"]
        latex_friendly_cnn_type = latex_friendly.get(cnn_type, cnn_type)
        val_name = f"{latex_friendly_cnn_type}Acc"
        values[val_name] = fmt(row.acc)
    return values
