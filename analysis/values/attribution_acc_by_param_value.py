# TODO
# accuracy of the custom-model classifier grouped by different parameters.
import database.api as db
from analysis.values.values_registry import VALUES_REGISTRY


parameters = ["scale", "loss", "architecture", "dataset"]

VALUES_REGISTRY.register()


def attribution_acc_by_param_value():
    values = {}
    for param in parameters:
        query = (
            f"select {param}, avg(predicted = actual)"
            "from SISR_analysis"
            'where classifier_name = "ConvNext_CNN_SISR_custom_models"'
            f"group by {param}"
        )
        result = db.read_sql_query(query)
        for row in result.itertuples():
            param_val, acc = row
            value_name = f"attribution_acc_for_{param_val}"
            formatted_acc = f"{acc*100:.1f}"
            values[value_name] = formatted_acc
    return values
