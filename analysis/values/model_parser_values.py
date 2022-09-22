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

N_ITERATIONS = 1000
N_VOTES = 10


def get_10_vote_accuracy(results):
    num_correct = 0
    total = 0
    actual_classes = set(results.actual)
    for actual in actual_classes:
        results_from_actual_class = results[results.actual == actual]
        for _ in range(N_ITERATIONS):
            predictions = np.random.choice(
                results_from_actual_class.predicted, size=N_VOTES
            )
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


def add_prefix_to_values(prefix, values):
    return {prefix + key: value for key, value in values.items()}


def filtered_model_parsing_table(classifier_type, clause):
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
            if classifier_name.endswith("SISR_dataset_parser_withholding_3"):
                continue  # this parser was created in error
            if reserved_param_val in "VGG":
                continue  # this parser was created in error.
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
                f" and {clause}"
            )
            if param == "dataset":
                prediction_query += " and seed = 1"
            params = {
                "classifier_name": classifier_name,
                "reserved_param_val": reserved_param_val,
            }
            parser_results = db.read_sql_query(prediction_query, params)
            if len(parser_results) == 0:
                continue
            if param == "dataset":
                print(
                    f"withholding-{reserved_param_val} where {clause}: {len(parser_results)}"
                )
            parser_accuracy = (parser_results.predicted == parser_results.actual).mean()
            reserved_param_latex_friendly = latexify(reserved_param_val)
            parser_acc_value_name = f"{param}Withholding{reserved_param_latex_friendly}"
            formatted_parser_accuracy = fmt(parser_accuracy)
            values[parser_acc_value_name] = formatted_parser_accuracy
    return values


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
            if classifier_name.endswith("SISR_dataset_parser_withholding_3"):
                continue  # this parser was created in error
            if reserved_param_val in "VGG":
                continue  # this parser was created in error.
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
            if param == "dataset":
                print(f"withholding-{reserved_param_val}: {len(parser_results)}")
            assert len(parser_results) > 0, f"{prediction_query}, {params}"
            parser_accuracy = (parser_results.predicted == parser_results.actual).mean()
            # parser_10_vote_accuracy = get_10_vote_accuracy(parser_results)
            reserved_param_latex_friendly = latexify(reserved_param_val)
            parser_acc_value_name = f"{param}Withholding{reserved_param_latex_friendly}"
            # parser_10_vote_acc_value_name = (
            #    f"{param}Withholding{reserved_param_latex_friendly}TenVote"
            # )
            formatted_parser_accuracy = fmt(parser_accuracy)
            # formatted_parser_10_vote_accuracy = fmt(parser_10_vote_accuracy)
            values[parser_acc_value_name] = formatted_parser_accuracy
            # values[parser_10_vote_acc_value_name] = formatted_parser_10_vote_accuracy

    # ScaleParsingLow
    scale_parsing_vals = [
        unfmt(val)
        for key, val in values.items()
        if key.startswith("scale") and not ("LOne" in key)
    ]
    values["ScaleParsingLow"] = fmt(min(scale_parsing_vals))
    values["ScaleParsingHigh"] = fmt(max(scale_parsing_vals))

    def record_parser_acc_val_v2(val_name, classifiers, condition=None):
        classifier_names_list = ",".join(f'"{c}"' for c in classifiers)
        classifier_ordered_labels_query = (
            "select c.name as classifier_name, td.ordered_labels as ordered_labels"
            " from classifier c"
            " inner join dataset td on td.id = c.training_dataset_id"
            f" where c.name in ({classifier_names_list})"
        )
        result = db.read_and_decode_sql_query(classifier_ordered_labels_query)

        classifier_ordered_labels = {}
        for classifier, ordered_labels in result.itertuples(index=False):
            classifier_ordered_labels[classifier] = ordered_labels

        priors_query = (
            "select c.name as classifier_name, p.actual as actual, count(*) as cnt"
            " from prediction p"
            " inner join classifier c on c.id = p.classifier_id"
            " inner join image_patch i on i.id = p.image_patch_id"
            " inner join sisr_generator sg on sg.generator_id = i.generator_id"
            " where not p.from_withheld"
            f" and c.name in ({classifier_names_list})"
            " group by c.name, p.actual"
        )

        result = db.read_sql_query(priors_query)

        priors = {}
        for classifier in set(result.classifier_name):
            ordered_labels = classifier_ordered_labels[classifier]
            classifier_rows = result[result.classifier_name == classifier]
            label_counts = np.zeros(len(ordered_labels))
            for _, actual, cnt in classifier_rows.itertuples(index=False):
                if actual is None:
                    continue
                actual_index = ordered_labels.index(actual)
                label_counts[actual_index] = cnt
            if np.any(label_counts == 0):
                raise Exception(
                    "Zero instances of one of the labels in the training data!"
                    f"{dict(zip(ordered_labels, label_counts))}"
                )
            priors[classifier] = 1 / label_counts
            priors[classifier] /= priors[classifier].sum()

        query = (
            "select c.name as classifier_name, p.class_probabilities as class_probabilities, p.actual as actual"
            " from prediction p"
            " inner join classifier c on c.id = p.classifier_id"
            " inner join image_patch i on i.id = p.image_patch_id"
            " inner join sisr_generator sg on sg.generator_id = i.generator_id"
            " where p.from_withheld"
            f" and c.name in ({classifier_names_list})"
        )
        if condition is not None:
            query += " and " + condition
        result = db.read_and_decode_sql_query(query)

        num_correct = 0
        num_total = 0
        for classifier_name, class_probabilities, actual in result.itertuples(
            index=False
        ):
            prior = priors[classifier_name]
            posterior = class_probabilities * prior
            ordered_labels = classifier_ordered_labels[classifier_name]
            prediction = ordered_labels[np.argmax(posterior)]
            if prediction == actual:
                num_correct += 1
            num_total += 1
        values[val_name] = fmt(num_correct / num_total)

    def record_parser_acc_val(val_name, classifiers, condition=None):
        classifier_names_list = ",".join(f'"{c}"' for c in classifiers)
        query = (
            "select avg(p.predicted = p.actual) as acc"
            " from prediction p"
            " inner join classifier c on c.id = p.classifier_id"
            " inner join image_patch i on i.id = p.image_patch_id"
            " inner join sisr_generator sg on sg.generator_id = i.generator_id"
            " where p.from_withheld"
            f" and c.name in ({classifier_names_list})"
        )

        if condition is not None:
            query += " and " + condition
        values[val_name] = get_acc_val(query)

    # loss_parsers = [
    #     f"{classifier_type}_SISR_loss_parser_withholding_3",
    #     f"{classifier_type}_SISR_loss_parser_withholding_RCAN",
    #     f"{classifier_type}_SISR_loss_parser_withholding_SwinIR",
    #     f"{classifier_type}_SISR_loss_parser_withholding_flickr2k",
    # ]

    # record_parser_acc_val("AverageScaleParsingAccuracy", loss_parsers)

    # record_parser_acc_val(
    #     "GANLossDistinctionAccuracy",
    #     loss_parsers,
    #     "sg.loss != 'L1'",
    # )

    # arch_parsers = [
    #     f"{classifier_type}_SISR_architecture_parser_withholding_3",
    #     f"{classifier_type}_SISR_architecture_parser_withholding_VGG_GAN",
    #     f"{classifier_type}_SISR_architecture_parser_withholding_flickr2k",
    # ]
    # record_parser_acc_val(
    #     "AverageArchParsingAccuracy",
    #     arch_parsers,
    # )

    # record_parser_acc_val(
    #     "AverageFourXArchParsingAccuracy",
    #     arch_parsers,
    #     "sg.scale = '4'",
    # )

    # record_parser_acc_val(
    #     "AverageTwoXArchParsingAccuracy",
    #     arch_parsers,
    #     "sg.scale = '2'",
    # )

    # dataset_parsers = [
    #     f"{classifier_type}_SISR_dataset_parser_withholding_RCAN",
    #     f"{classifier_type}_SISR_dataset_parser_withholding_SwinIR",
    #     f"{classifier_type}_SISR_dataset_parser_withholding_VGG_GAN",
    #     f"{classifier_type}_SISR_dataset_parser_withholding_flickr2k",
    # ]
    # record_parser_acc_val_v2(
    #     "AverageDatasetParsingAccuracy", dataset_parsers, "sg.seed = 1"
    # )

    # record_parser_acc_val_v2(
    #     "AverageFourXDatasetParsingAccuracy",
    #     dataset_parsers,
    #     "sg.scale = '4' and sg.seed = 1",
    # )

    # record_parser_acc_val_v2(
    #     "AverageTwoXDatasetParsingAccuracy",
    #     dataset_parsers,
    #     "sg.scale = '2' and sg.seed = 1",
    # )

    # record_parser_acc_val_v2(
    #     "AverageQuarterDatasetParsingAccuracy",
    #     dataset_parsers,
    #     "sg.dataset like 'quarter%' and sg.seed = 1",
    # )

    # record_parser_acc_val_v2(
    #     "AverageFullDatasetParsingAccuracy",
    #     dataset_parsers,
    #     "sg.dataset not like 'quarter%' and sg.seed = 1",
    # )

    ## NEW in-text parsing values!
    # swinIR_split_loss_parser = [
    #     f"{classifier_type}_SISR_loss_parser_withholding_SwinIR"
    # ]
    # record_parser_acc_val(
    #     "SwinIRSplitLoneLossParsingAccuracy", swinIR_split_loss_parser, "sg.loss = 'L1'"
    # )
    # record_parser_acc_val(
    #     "SwinIRSplitVGGLossParsingAccuracy",
    #     swinIR_split_loss_parser,
    #     "sg.loss = 'VGG_GAN'",
    # )
    # record_parser_acc_val(
    #     "SwinIRSplitResNetLossParsingAccuracy",
    #     swinIR_split_loss_parser,
    #     "sg.loss = 'ResNet_GAN'",
    # )
    # record_parser_acc_val(
    #     "SwinIRSplitTwoXLossParsingAccuracy", swinIR_split_loss_parser, "sg.scale = '2'"
    # )
    # record_parser_acc_val(
    #     "SwinIRSplitFourXLossParsingAccuracy",
    #     swinIR_split_loss_parser,
    #     "sg.scale = '4'",
    # )

    # three_split_architecture_parser = [
    #     f"{classifier_type}_SISR_architecture_parser_withholding_3"
    # ]
    # record_parser_acc_val(
    #     "ThreeSplitTwoXArchitectureParsingAccuracy",
    #     three_split_architecture_parser,
    #     "sg.scale = '2'",
    # )
    # record_parser_acc_val(
    #     "ThreeSplitFourXArchitectureParsingAccuracy",
    #     three_split_architecture_parser,
    #     "sg.scale = '4'",
    # )
    # record_parser_acc_val(
    #     "ThreeSplitLOneArchitectureParsingAccuracy",
    #     three_split_architecture_parser,
    #     "sg.loss = 'L1'",
    # )
    # record_parser_acc_val(
    #     "ThreeSplitVGGArchitectureParsingAccuracy",
    #     three_split_architecture_parser,
    #     "sg.loss = 'VGG_GAN'",
    # )
    # record_parser_acc_val(
    #     "ThreeSplitResNetArchitectureParsingAccuracy",
    #     three_split_architecture_parser,
    #     "sg.loss = 'ResNet_GAN'",
    # )

    # VGG_split_architecture_parser = [
    #     f"{classifier_type}_SISR_architecture_parser_withholding_VGG_GAN"
    # ]
    # record_parser_acc_val(
    #     "VGGSplitTwoXArchitectureParsingAccuracy",
    #     VGG_split_architecture_parser,
    #     "sg.scale = '2'",
    # )
    # record_parser_acc_val(
    #     "VGGSplitFourXArchitectureParsingAccuracy",
    #     VGG_split_architecture_parser,
    #     "sg.scale = '4'",
    # )

    # rcan_split_dataset_parser = [
    #     f"{classifier_type}_SISR_dataset_parser_withholding_RCAN"
    # ]
    # # break down by sclae
    # record_parser_acc_val(
    #     "RCANSplitTwoXDatasetParsingAccuracy",
    #     rcan_split_dataset_parser,
    #     "sg.scale = '2' and sg.seed = 1",
    # )
    # record_parser_acc_val(
    #     "RCANSplitFourXDatasetParsingAccuracy",
    #     rcan_split_dataset_parser,
    #     "sg.scale = '4' and sg.seed = 1",
    # )
    # # break down by loss
    # record_parser_acc_val(
    #     "RCANSplitLOneDatasetParsingAccuracy",
    #     rcan_split_dataset_parser,
    #     "sg.loss = 'L1' and sg.seed = 1",
    # )
    # record_parser_acc_val(
    #     "RCANSplitVGGDatasetParsingAccuracy",
    #     rcan_split_dataset_parser,
    #     "sg.loss = 'VGG_GAN' and sg.seed = 1",
    # )
    # record_parser_acc_val(
    #     "RCANSplitResNetDatasetParsingAccuracy",
    #     rcan_split_dataset_parser,
    #     "sg.loss = 'ResNet_GAN' and sg.seed = 1",
    # )

    # # break down by quarter/full dataset
    # record_parser_acc_val(
    #     "RCANSplitQuarterDatasetParsingAccuracy",
    #     rcan_split_dataset_parser,
    #     "sg.dataset like 'quarter%' and sg.seed = 1",
    # )
    # record_parser_acc_val(
    #     "RCANSplitFullDatasetParsingAccuracy",
    #     rcan_split_dataset_parser,
    #     "sg.dataset not like 'quarter%' and sg.seed = 1",
    # )

    return values


@VALUES_REGISTRY.register()
def model_parsing_table():
    return model_parsing_table_for("ConvNext_CNN")


@VALUES_REGISTRY.register()
def model_parsing_table_across_seeds():
    return aggregate_across_seeds(
        [
            model_parsing_table_for("ConvNext_CNN"),
            model_parsing_table_for("seed_2_ConvNext_CNN"),
            model_parsing_table_for("seed_3_ConvNext_CNN"),
        ]
    )


@VALUES_REGISTRY.register()
def disaggregated_model_parsing_tables():
    values = {}
    values.update(
        add_prefix_to_values(
            "twoX", filtered_model_parsing_table("ConvNext_CNN", "scale = 2")
        )
    )
    values.update(
        add_prefix_to_values(
            "fourX", filtered_model_parsing_table("ConvNext_CNN", "scale = 4")
        )
    )
    values.update(
        add_prefix_to_values(
            "lOne", filtered_model_parsing_table("ConvNext_CNN", "loss = 'L1'")
        )
    )
    values.update(
        add_prefix_to_values(
            "adv", filtered_model_parsing_table("ConvNext_CNN", "loss != 'L1'")
        )
    )
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
def pretrained_parser_vals():
    return get_pretrained_parser_vals("ConvNext_CNN_SISR_architecture_parser")


@VALUES_REGISTRY.register()
def pretrained_parser_vals_across_seeds():
    return aggregate_across_seeds(
        [
            get_pretrained_parser_vals("ConvNext_CNN_SISR_architecture_parser"),
            get_pretrained_parser_vals("seed_2_ConvNext_CNN_SISR_architecture_parser"),
            get_pretrained_parser_vals("seed_3_ConvNext_CNN_SISR_architecture_parser"),
        ]
    )


@VALUES_REGISTRY.register()
def PRNU_model_parsing_table():
    values = model_parsing_table_for("PRNU")
    return {f"PRNU{key}": val for key, val in values.items()}


# average arch parsing accuracy
# arch parsing acc for 4x vs. 2x
# arch parsing for L1 vs GAN
# average dataset parsing accuracy
# dataset parsing acc for 4x vs. 2x
# dataset parsing for L1 vs GAN

# arch parsing accuracy on pretrained models
