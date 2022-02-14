import database.api as db
from analysis.values.values_registry import VALUES_REGISTRY

query = (
    "select classifier_name, classifier_opt, avg(predicted = actual) as acc"
    " from analysis "
    ' where train_dataset_name = "SISR_all_models_train"'
    " and classifier_type = CNN"
    " group by classifier_name, classifier_opt"
    " order by avg(predicted = actual) desc"
)


@VALUES_REGISTRY.register()
def arch_all_class_acc():
    result = db.read_sql_query(query)
    values = {}
    for row in result.itertuples(index=False):
        cnn_type = row.classifier_opt["cnn_opt"]["type"]
        val_name = f"{cnn_type}_all_class_acc"
        values[val_name] = row.acc
    return values
