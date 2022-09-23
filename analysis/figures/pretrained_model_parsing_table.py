import json
import database.api as db
import numpy as np
import argparse
from matplotlib import colors, cm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Create the pretrained model parsing table as a PDF figure."
)

parser.add_argument(
    "--data_file",
    default=None,
    help="If provided, reads the data to format into the table from the "
    "specified JSON file, instead of from the database."
    " Database reading is slow, so this is useful for experimenting "
    "with the figure formatting.",
)

args = parser.parse_args()
# get all the "all-pretrained" classifiers, excluding model
classifiers = db.read_sql_query(
    "select c.name as name, sd.label_param as label_param"
    " from classifier c"
    " inner join dataset d on c.training_dataset_id = d.id"
    " inner join SISR_dataset sd on d.id = sd.dataset_id"
    " where sd.reserved_param is null"
    " and c.name like 'ConvNext%'"
    " and sd.label_param is not null"
    " and sd.include_pretrained is false"
)

#####################################################################
## Read the data from the database.
#####################################################################

if args.data_file is None:
    all_data = []
    # data is a dictionary from classifier names to that classifier's predictions
    # the predictions are a dictionary from SISR model to prediction counts
    # prediction counts are a dict from predicted label to the the counts of that prediction for this true SISR model for this classifier.
    for classifier_prefix in ["ConvNext", "seed_2_ConvNext", "seed_3_ConvNext"]:
        data = {}
        classifiers = db.read_sql_query(
            "select c.name as name, sd.label_param as label_param"
            " from classifier c"
            " inner join dataset d on c.training_dataset_id = d.id"
            " inner join SISR_dataset sd on d.id = sd.dataset_id"
            " where sd.reserved_param is null"
            f" and c.name like '{classifier_prefix}%'"
            " and sd.label_param is not null"
            " and sd.include_pretrained is false"
        )

        # for each parser trained on all the custom classifiers:
        for classifier, label_param in classifiers.itertuples(index=False):
            classifier_data = {}
            # for each pretrained SISR method analyzed by the classifier:
            pretrained_sisr_models = db.read_sql_query(
                "select distinct generator_name from analysis "
                " where classifier_name = :classifier_name"
                r" and generator_name like '%-pretrained'",
                params={"classifier_name": classifier},
            )
            for sisr_model in pretrained_sisr_models.generator_name:
                # get the percentage of predicted_class which predict this true value for this pretrained SISR method
                predictions_counts = db.read_sql_query(
                    "select predicted, count(*) from analysis"
                    " where generator_name = :sisr_model "
                    "and classifier_name = :classifier_name "
                    "group by predicted",
                    params={"classifier_name": classifier, "sisr_model": sisr_model},
                )
                # add this percentage to table data.
                predictions_counts = dict(predictions_counts.itertuples(index=False))
                classifier_data[sisr_model] = predictions_counts
            data[label_param] = classifier_data
        all_data.append(data)

    with open("analysis/figures/pretrained_model_parsing_table_data.json", "w") as f:
        json.dump(all_data, f, indent=2)

#####################################################################
## Plot the figure and save it as a PDF
#####################################################################
if args.data_file is None:
    args.data_file = "analysis/figures/pretrained_model_parsing_table_data.json"
data = json.load(open(args.data_file))
rows = [
    ("EDSR-2x", "EDSR-div2k-x2-L1-NA-pretrained"),
    ("LIIF-2x", "LIIF-div2k-x2-L1-NA-pretrained"),
    ("RCAN-2x", "RCAN-div2k-x2-L1-NA-pretrained"),
    ("RDN-2x", "RDN-div2k-x2-L1-NA-pretrained"),
    ("R. ESRGAN-2x", "Real_ESRGAN-div2k-x2-GAN-NA-pretrained"),
    ("SRFBN-2x", "SRFBN-NA-x2-L1-NA-pretrained"),
    ("SwinIR-2x", "SwinIR-div2k-x2-L1-NA-pretrained"),
    ("NLSN-2x", "NLSN-div2k-x2-L1-NA-pretrained"),
    ("DRN-4x", "DRN-div2k-x4-L1-NA-pretrained"),
    ("EDSR-4x", "EDSR-div2k-x4-L1-NA-pretrained"),
    ("LIIF-4x", "LIIF-div2k-x4-L1-NA-pretrained"),
    ("proSR-4x", "proSR-div2k-x4-L1-NA-pretrained"),
    ("RCAN-4x", "RCAN-div2k-x4-L1-NA-pretrained"),
    ("RDN-4x", "RDN-div2k-x4-L1-NA-pretrained"),
    ("SAN-4x", "SAN-div2k-x4-L1-NA-pretrained"),
    ("SRFBN-4x", "SRFBN-NA-x4-L1-NA-pretrained"),
    ("SwinIR-4x", "SwinIR-div2k-x4-L1-NA-pretrained"),
    ("NLSN-4x", "NLSN-div2k-x4-L1-NA-pretrained"),
    ("ESRGAN-4x", "ESRGAN-NA-x4-ESRGAN-NA-pretrained"),
    ("E.Net-4x", "EnhanceNet-NA-x4-EnhanceNet-NA-pretrained"),
    ("NCSR-4x", "NCSR-div2k-x4-NCSR_GAN-NA-pretrained"),
    ("proSR-4x", "proSR-div2k-x4-ProSRGAN-NA-pretrained"),
    ("R. ESRGAN-4x", "Real_ESRGAN-div2k-x4-GAN-NA-pretrained"),
    ("SPSR-4x", "SPSR-div2k-x4-SPSR_GAN-NA-pretrained"),
    ("SwinIR-adv-4x", "SwinIR-div2k-x4-GAN-NA-pretrained"),
]

columns = [
    ("scale", "x2", "2x"),
    ("scale", "x4", "4x"),
    ("loss", "L1", "L₁"),
    ("loss", "VGG_GAN", "VGG+A"),
    ("loss", "ResNet_GAN", "R.Net+A"),
    ("architecture", "EDSR", "EDSR"),
    ("architecture", "RCAN", "RCAN"),
    ("architecture", "RDN", "RDN"),
    ("architecture", "SwinIR", "SwinIR"),
    ("architecture", "NLSN", "NLSN"),
]
scale_cols = [("2", "2x"), ("4", "4x")]
loss_cols = [("L1", "L₁"), ("VGG_GAN", "VGG+A."), ("ResNet_GAN", "R.Net+A.")]
arch_cols = [
    ("EDSR", "EDSR"),
    ("RCAN", "RCAN"),
    ("RDN", "RDN"),
    ("SwinIR", "SwinIR"),
    ("NLSN", "NLSN"),
]

cmap_name = "Blues"
cmap = cm.get_cmap(cmap_name, lut=2)


def plot_param_predictions(ax, parameter, rows, cols, rowlabels=False, title=None):
    matrix = np.zeros((len(rows), len(cols)), dtype=int)
    for row_index, row in enumerate(rows):
        row_label, model = row
        for col_index, col in enumerate(cols):
            param_value, col_label = col
            value = data[0][parameter][model].get(param_value, 0)
            matrix[row_index, col_index] = value
    ax.imshow(np.zeros_like(matrix), aspect=0.45, cmap=cmap_name)

    for row_index, row in enumerate(rows):
        row_label, model = row
        for col_index, col in enumerate(cols):
            param_value, col_label = col
            values = [
                seed_data[parameter][model].get(param_value, 0) for seed_data in data
            ]
            values_mean = int(np.mean(values))
            values_std = int(np.std(values))
            text_color = cmap(1)  # cmap(value < 50)
            value_str = (
                values_mean if values_std == 0 else f"{values_mean}±{values_std}"
            )
            valid_arches = {
                "EDSR-2x",
                "RDN-2x",
                "RCAN-2x",
                "SwinIR-2x",
                "NLSN-2x",
                "EDSR-4x",
                "RDN-4x",
                "RCAN-4x",
                "SwinIR-4x",
                "NLSN-4x",
                "SwinIR-adv-4x",
            }
            if parameter == "architecture" and row_label not in valid_arches:
                value_str = "-"
            ax.text(
                col_index,
                row_index + 0.2,
                str(value_str),
                color=text_color,
                horizontalalignment="center",
            )

    if title is not None:
        ax.set_title(title, y=-0.04)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([col_label for _, col_label in cols])
    ax.xaxis.tick_top()
    if rowlabels:
        ax.set_ylabel("Actual model")
        ax.yaxis.set_label_coords(-1.1, 0.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row_label for row_label, _ in rows])
    else:
        ax.set_yticks([])


def draw_rectangle(ax, row, col, width=1, height=1):
    rect = plt.Rectangle(
        (col - 0.5, row - 0.5),
        width,
        height,
        linewidth=2,
        facecolor="none",
        edgecolor=(0.1, 0.1, 0.7),
    )
    ax.add_patch(rect)


fig, axs = plt.subplots(
    1,
    3,
    figsize=(len(columns) * 0.8, len(rows) * 0.8),
    gridspec_kw={"width_ratios": [2, 3, 5]},
)

plot_param_predictions(axs[0], "scale", rows, scale_cols, True, "Scale")
draw_rectangle(axs[0], 0, 0, 1, 8)
draw_rectangle(axs[0], 8, 1, 1, 17)

plot_param_predictions(axs[1], "loss", rows, loss_cols, False, "Loss")
draw_rectangle(axs[1], 0, 0, 1, 4)
draw_rectangle(axs[1], 5, 0, 1, 13)

plot_param_predictions(axs[2], "architecture", rows, arch_cols, False, "Architecture")

draw_rectangle(axs[2], 0, 0, 1, 1)
draw_rectangle(axs[2], 2, 1, 1, 1)
draw_rectangle(axs[2], 3, 2, 1, 1)
draw_rectangle(axs[2], 6, 3, 1, 1)
draw_rectangle(axs[2], 7, 4, 1, 1)

draw_rectangle(axs[2], 9, 0, 1, 1)
draw_rectangle(axs[2], 12, 1, 1, 1)
draw_rectangle(axs[2], 13, 2, 1, 1)
draw_rectangle(axs[2], 16, 3, 1, 1)
draw_rectangle(axs[2], 17, 4, 1, 1)
draw_rectangle(axs[2], 24, 3, 1, 1)

plt.subplots_adjust(wspace=0.1)

fig.suptitle("Predicted hyperparameter value", y=0.685, x=0.5)
plt.gca().xaxis.set_label_position("top")
plt.savefig(
    f"paper/figures/pretrained-model-parsing-table.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=200,
)
