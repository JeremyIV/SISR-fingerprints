import database.api as db
from analysis.values.values_registry import VALUES_REGISTRY
from analysis.values.val_utils import (
    fmt,
    unfmt,
    latexify,
    get_acc_val,
    aggregate_across_seeds,
)
import utils
import numpy as np

parameters = ["scale", "loss", "architecture", "dataset"]


def acc_by_param_val_for(classifier_name):
    """Computes the accuracy of the custom model parser
    disaggregated by each parameter value."""
    values = {}
    for param in parameters:
        query = (
            f"select {param}, avg(predicted = actual)"
            " from SISR_analysis"
            f' where classifier_name = "{classifier_name}"'
            ' and generator_name not like "%-pretrained"'
            f" group by {param}"
        )
        result = db.read_sql_query(query)
        for row in result.itertuples(index=False):
            param_val, acc = row
            param_val = latexify(param_val)
            values[f"AccFor{param_val}"] = fmt(acc)
    values["ScaleAccRange"] = fmt(
        unfmt(values["AccForFourX"]) - unfmt(values["AccForTwoX"])
    )
    mean_adv_acc = (unfmt(values["AccForResNet"]) + unfmt(values["AccForVGG"])) / 2
    values["LossAccRange"] = fmt(mean_adv_acc - unfmt(values["AccForLOne"]))
    return values


def pretrained_acc_for(classifier_name):
    values = {}

    def add_acc_val(name, query):
        values[name] = get_acc_val(query)

    add_acc_val(
        "PretrainedClassifierAcc",
        "select avg(predicted = actual) as acc"
        " from analysis"
        f' where classifier_name ="{classifier_name}"',
    )

    add_acc_val(
        "PreTrainedTwoXModelClassifierAcc",
        "select avg(predicted = actual) as acc"
        " from sisr_analysis"
        f' where classifier_name ="{classifier_name}"'
        ' and scale = "2"',
    )

    add_acc_val(
        "PreTrainedFourXLOneModelClassifierAcc",
        "select avg(predicted = actual) as acc"
        " from sisr_analysis"
        f' where classifier_name ="{classifier_name}"'
        ' and scale = "4"'
        ' and loss = "L1"',
    )

    add_acc_val(
        "PreTrainedGANModelClassifierAcc",
        "select avg(predicted = actual) as acc"
        " from sisr_analysis"
        f' where classifier_name ="{classifier_name}"'
        ' and scale = "4"'
        ' and loss != "L1"',
    )

    return values


@VALUES_REGISTRY.register()
def acc_by_param_val():
    return aggregate_across_seeds(
        [
            acc_by_param_val_for("ConvNext_CNN_SISR_custom_models"),
            acc_by_param_val_for("seed_2_ConvNext_CNN_SISR_custom_models"),
            acc_by_param_val_for("seed_3_ConvNext_CNN_SISR_custom_models"),
        ]
    )


@VALUES_REGISTRY.register()
def prnu_acc_by_param_val():
    values = acc_by_param_val_for("PRNU_SISR_custom_models")
    return {f"PRNU{key}": val for key, val in values.items()}


@VALUES_REGISTRY.register()
def pretrained_acc():
    return aggregate_across_seeds(
        [
            pretrained_acc_for("ConvNext_CNN_SISR_pretrained_models"),
            pretrained_acc_for("seed_2_ConvNext_CNN_SISR_pretrained_models"),
            pretrained_acc_for("seed_3_ConvNext_CNN_SISR_pretrained_models"),
        ]
    )


@VALUES_REGISTRY.register()
def prnu_pretrained_acc():
    values = pretrained_acc_for("PRNU_SISR_pretrained_models")
    return {f"PRNU{key}": val for key, val in values.items()}


def get_seed_triplets():
    models = utils.get_sisr_model_names(dataset=[utils.div2k])
    seed_triplets = {}
    for model in models:
        seedless_prefix = model[:-3]
        if seedless_prefix not in seed_triplets:
            seed_triplets[seedless_prefix] = []
        seed_triplets[seedless_prefix].append(model)
    return list(seed_triplets.values())


@VALUES_REGISTRY.register()
def seed_distinction_acc_ConvNext():
    return aggregate_across_seeds(
        [
            seed_distinction_acc("ConvNext_CNN"),
            seed_distinction_acc("seed_2_ConvNext_CNN"),
            seed_distinction_acc("seed_3_ConvNext_CNN"),
        ]
    )


@VALUES_REGISTRY.register()
def seed_distinction_acc_PRNU():
    values = seed_distinction_acc("PRNU")
    return {f"PRNU{key}": val for key, val in values.items()}


def seed_distinction_acc(classifier_type):
    seed_triplets = get_seed_triplets()
    ordered_labels = db.read_and_decode_sql_query(
        "select d.ordered_labels as ordered_labels"
        " from classifier c"
        " inner join dataset d on c.training_dataset_id = d.id"
        f' where c.name = "{classifier_type}_SISR_custom_models"'
    )
    ordered_labels = np.array(list(ordered_labels.ordered_labels))[0]
    losses = ["L1", "VGG_GAN", "ResNet_GAN"]
    num_correct = np.array([[0, 0], [0, 0], [0, 0]])
    num_total = np.array([[0, 0], [0, 0], [0, 0]])

    for triplet in seed_triplets:
        if triplet[0].startswith("NLSN"):
            continue
        loss_index = None
        for idx, loss in enumerate(losses):
            if loss in triplet[0]:
                loss_index = idx
                break
        is_x4 = "-x4-" in triplet[0]
        prior = np.array([model in triplet for model in ordered_labels])
        seed_triplet_sql_tuple = ", ".join(f'"{model}"' for model in triplet)

        result = db.read_and_decode_sql_query(
            " select class_probabilities, actual from analysis"
            f" where actual in ({seed_triplet_sql_tuple})"
            f' and classifier_name = "{classifier_type}_SISR_custom_models"'
        )
        probs = np.array(list(result.class_probabilities))
        posterior = probs * prior
        prediction_indices = np.argmax(posterior, axis=1)
        predictions = ordered_labels[prediction_indices]
        num_total[loss_index, int(is_x4)] += len(predictions)
        num_correct[loss_index, int(is_x4)] += (predictions == result.actual).sum()
        # print(f"{triplet[0]}: {(predictions == result.actual).mean()}")

    values = {}
    values["DistinctionBetweenSeedAcc"] = fmt(num_correct.sum() / num_total.sum())
    # print(num_total)

    def get_acc_for(loss, is_x4):
        loss_index = losses.index(loss)
        return fmt(
            num_correct[loss_index, int(is_x4)] / num_total[loss_index, int(is_x4)]
        )

    values["DistinctionBetweenSeedTwoXLOneAcc"] = get_acc_for("L1", is_x4=False)
    values["DistinctionBetweenSeedFourXLOneAcc"] = get_acc_for("L1", is_x4=True)
    values["DistinctionBetweenSeedTwoXVGGAcc"] = get_acc_for("VGG_GAN", is_x4=False)
    values["DistinctionBetweenSeedFourXVGGAcc"] = get_acc_for("VGG_GAN", is_x4=True)
    values["DistinctionBetweenSeedTwoXResNetAcc"] = get_acc_for(
        "ResNet_GAN", is_x4=False
    )
    values["DistinctionBetweenSeedFourXResNetAcc"] = get_acc_for(
        "ResNet_GAN", is_x4=True
    )
    return values
