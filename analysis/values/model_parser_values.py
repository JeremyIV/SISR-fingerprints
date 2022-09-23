import database.api as db
import numpy as np
from analysis.values.values_registry import VALUES_REGISTRY
from analysis.values.val_utils import (
    fmt,
    unfmt,
    latexify,
    get_acc_val,
    aggregate_across_seeds,
)

parameters = ["scale", "loss", "architecture", "dataset", "seed"]
params_to_predict = ["scale", "loss", "architecture", "dataset"]


def model_parsing_table_for(classifier_type):
    values = {}
    for param in params_to_predict:
        param_parsers_query = (
            "select sd.reserved_param, sd.reserved_param_value, c.name"
            " from SISR_dataset sd"
            " inner join dataset d on d.id = sd.dataset_id"
            " inner join classifier c on c.training_dataset_id = d.id"
            f' where c.name like "{classifier_type}%"'
            f' and sd.label_param = "{param}"'
            " and sd.reserved_param is not null"
        )
        result = db.read_sql_query(param_parsers_query)
        for reserved_param, reserved_param_val, classifier_name in result.itertuples(
            index=False
        ):
            # get the predicted and actual classes on the reserved models
            assert (
                reserved_param in parameters
            ), f"unrecognized reserved param {reserved_param}"
            prediction_query = (
                "select predicted, actual"
                " from sisr_analysis"
                " where classifier_name = :classifier_name"
                f" and {reserved_param} = :reserved_param_val"
                r' and generator_name not like "%-pretrained"'
            )
            if param == "dataset":
                prediction_query += " and seed = 1"
            params = {
                "classifier_name": classifier_name,
                "reserved_param_val": reserved_param_val,
            }
            parser_results = db.read_sql_query(prediction_query, params)
            assert len(parser_results) > 0, f"{prediction_query}, {params}"
            parser_accuracy = (parser_results.predicted == parser_results.actual).mean()
            reserved_param_latex_friendly = latexify(reserved_param_val)
            parser_acc_value_name = f"{param}Withholding{reserved_param_latex_friendly}"
            formatted_parser_accuracy = fmt(parser_accuracy)
            values[parser_acc_value_name] = formatted_parser_accuracy

    scale_parsing_vals = [
        unfmt(val)
        for key, val in values.items()
        if key.startswith("scale") and not ("LOne" in key)
    ]
    values["ScaleParsingLow"] = fmt(min(scale_parsing_vals))
    values["ScaleParsingHigh"] = fmt(max(scale_parsing_vals))

    return values


def get_pretrained_parser_vals(classifier_name):
    values = {}
    pretrained_generators_with_known_arch = ", ".join(
        [
            '"EDSR-div2k-x2-L1-NA-pretrained"',
            '"RCAN-div2k-x2-L1-NA-pretrained"',
            '"RDN-div2k-x2-L1-NA-pretrained"',
            '"SwinIR-div2k-x2-L1-NA-pretrained"',
            '"NLSN-div2k-x2-L1-NA-pretrained"',
            '"EDSR-div2k-x4-L1-NA-pretrained"',
            '"RCAN-div2k-x4-L1-NA-pretrained"',
            '"RDN-div2k-x4-L1-NA-pretrained"',
            '"SwinIR-div2k-x4-L1-NA-pretrained"',
            '"NLSN-div2k-x4-L1-NA-pretrained"',
            '"SwinIR-div2k-x4-GAN-NA-pretrained"',
        ]
    )
    values["PretrainedArchParsingAccuracy"] = get_acc_val(
        "select avg(predicted = actual) as acc"
        " from analysis "
        f" where classifier_name = '{classifier_name}'"
        f" and generator_name in ({pretrained_generators_with_known_arch})"
    )
    return values


@VALUES_REGISTRY.register()
def model_parsing_table():
    return aggregate_across_seeds(
        [
            model_parsing_table_for("ConvNext_CNN"),
            model_parsing_table_for("seed_2_ConvNext_CNN"),
            model_parsing_table_for("seed_3_ConvNext_CNN"),
        ]
    )


@VALUES_REGISTRY.register()
def pretrained_parser_vals():
    return aggregate_across_seeds(
        [
            get_pretrained_parser_vals("ConvNext_CNN_SISR_architecture_parser"),
            get_pretrained_parser_vals("seed_2_ConvNext_CNN_SISR_architecture_parser"),
            get_pretrained_parser_vals("seed_3_ConvNext_CNN_SISR_architecture_parser"),
        ]
    )
