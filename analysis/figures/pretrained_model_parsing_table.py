# pretrained_model_parsing_table.py
# outputs pretrained_model_parsing_table.pdf

import sqlite3
import json
import database.api as db
import argparse

parser = argparse.ArgumentParser(
    description="Create the pretrained model parsing table as a PDF figure."
)

parser.add_argument(
    "--data_file",
    default=None,
    help="If provided, reads the data to format into the table from the "
    "specified JSON file, instead of from the database."
    " Database reading is slow, so this is useful for experimenting "
    "with the figure fomratting.",
)
# get all the "all-pretrained" classifiers, excluding model
classifiers = db.read_sql_query(
    "select c.name as name, sc.label_param as label_param"
    " from classifier c"
    " inner join sisr_classifier sc on sc.classifier_id = c.id"
    " inner join dataset d on c.training_dataset_id = d.id"
    " inner join SISR_dataset sd on d.id = sd.dataset_id"
    " where sd.reserved_param is null"
    " and sd.include_pretrained is false"
)

#####################################################################
## Read the data from the database.
#####################################################################

if args.data_file is None:
    # data is a dictionary from classifier names to that classifier's predictions
    # the predictions are a dictionary from SISR model to prediction counts
    # prediction counts are a dict from predicted label to the the counts of that prediction for this true SISR model for this classifier.
    data = {}
    # for each parser trained on all the custom classifiers:
    for classifier, label_param in classifiers.iterrows():
        classifier_data = {}
        # for each pretrained SISR method analyzed by the classifier:
        pretrained_sisr_models = db.read_sql_query(
            "select distinct generator_name from analysis "
            " where classifier_name = :classifier_name"
            r" and generator_name like '%-pretrained'",
            params={"classifier_name": classifier},
        )
        # TODO: this for loop is probably wrong
        for sisr_model in pretrained_sisr_models:
            # get the percentage of predicted_class which predict this true value for this pretrained SISR method
            predictions_counts = db.read_sql_query(
                "select prediction, count(*) from analysis"
                " where sisr_model = :sisr_model "
                "and classifier_name = :classifier_name "
                "group by prediction",
                params={"classifier_name": classifier, "sisr_model": sisr_model},
            )
            # add this percentage to table data.
            # TODO: this conversion to a dict is probably wrong
            predictions_counts = dict(predictions_counts)
            classifier_data[sisr_model] = predictions_counts
        data[classifier] = classifier_data

    with open("pretrained_model_parsing_table_data.json", "w") as f:
        json.dump(data, f, indent=2)

#####################################################################
## Plot the figure and save it as a PDF
#####################################################################
if args.data_file is None:
    args.data_file = "pretrained_model_parsing_table_data.json"
data = json.load(open(args.data_file))
rows = [
    # TODO: add NLSN
    ("EDSR-2x", "EDSR-div2k-x2-L1-NA-pretrained"),
    ("LIIF-2x", "LIIF-div2k-x2-L1-NA-pretrained"),
    ("RCAN-2x", "RCAN-div2k-x2-L1-NA-pretrained"),
    ("RDN-2x", "RDN-div2k-x2-L1-NA-pretrained"),
    ("R. E.GAN-2x", "Real_ESRGAN-div2k-x2-GAN-NA-pretrained"),
    ("SRFBN-2x", "SRFBN-NA-x2-L1-NA-pretrained"),
    ("SwinIR-2x", "SwinIR-div2k-x2-L1-NA-pretrained"),
    ("DRN-4x", "DRN-div2k-x4-L1-NA-pretrained"),
    ("EDSR-4x", "EDSR-div2k-x4-L1-NA-pretrained"),
    ("LIIF-4x", "LIIF-div2k-x4-L1-NA-pretrained"),
    ("proSR-4x", "proSR-div2k-x4-L1-NA-pretrained"),
    ("RCAN-4x", "RCAN-div2k-x4-L1-NA-pretrained"),
    ("RDN-4x", "RDN-div2k-x4-L1-NA-pretrained"),
    ("SAN-4x", "SAN-div2k-x4-L1-NA-pretrained"),
    ("SRFBN-4x", "SRFBN-NA-x4-L1-NA-pretrained"),
    ("SwinIR-4x", "SwinIR-div2k-x4-L1-NA-pretrained"),
    ("E.GAN-4x", "ESRGAN-NA-x4-ESRGAN-NA-pretrained"),
    ("E.Net-4x", "EnhanceNet-NA-x4-EnhanceNet-NA-pretrained"),
    ("NCSR-4x", "NCSR-div2k-x4-NCSR_GAN-NA-pretrained"),
    ("proSR-4x", "proSR-div2k-x4-ProSRGAN-NA-pretrained"),
    ("R. E.GAN-4x", "Real_ESRGAN-div2k-x4-GAN-NA-pretrained"),
    ("SPSR-4x", "SPSR-div2k-x4-SPSR_GAN-NA-pretrained"),
    ("SwinIR-adv-4x", "SwinIR-div2k-x4-GAN-NA-pretrained"),
]

columns = [
    ("scale", "x2", "2x"),
    ("scale", "x4", "4x"),
    ("loss", "L1", "L₁"),  # TODO: subscript one
    ("loss", "VGG_GAN", "VGG+A"),
    ("loss", "ResNet_GAN", "R.Net+A"),
    ("architecture", "EDSR", "EDSR"),
    ("architecture", "RCAN", "RCAN"),
    ("architecture", "RDN", "RDN"),
]
scale_cols = [("x2", "2x"), ("x4", "4x")]
loss_cols = [("L1", "L₁"), ("VGG_GAN", "VGG+A."), ("ResNet_GAN", "R.Net+A.")]
arch_cols = [("EDSR", "EDSR"), ("RCAN", "RCAN"), ("RDN", "RDN")]

cmap_name = "viridis"
cmap = cm.get_cmap(cmap_name, lut=2)


def plot_param_predictions(ax, parameter, rows, cols, rowlabels=False, title=None):
    matrix = np.zeros((len(rows), len(cols)), dtype=int)
    for row_index, row in enumerate(rows):
        row_label, model = row
        for col_index, col in enumerate(cols):
            param_value, col_label = col
            value = table_data[f"all-pretrained-{parameter}"][model].get(param_value, 0)
            matrix[row_index, col_index] = value
    ax.imshow(matrix, aspect=3 / 4, cmap=cmap_name)

    # TODO: write the numbers
    for row_index, row in enumerate(rows):
        row_label, model = row
        for col_index, col in enumerate(cols):
            param_value, col_label = col
            value = table_data[f"all-pretrained-{parameter}"][model].get(param_value, 0)
            text_color = cmap(value < 50)
            offset = 0.35 if value == 100 else (0.3 if value > 9 else 0.1)
            ax.text(col_index - offset, row_index + 0.2, str(value), color=text_color)

    if title is not None:
        ax.set_title(title, y=-0.06)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([col_label for _, col_label in cols], rotation=60)
    ax.xaxis.tick_top()
    if rowlabels:
        ax.set_ylabel("Actual model")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row_label for row_label, _ in rows])
    else:
        ax.set_yticks([])


def draw_rectangle(ax, row, col, width=1, height=1):
    rect = plt.Rectangle(
        (col - 0.5, row - 0.5),
        width,
        height,
        linewidth=3,
        facecolor="none",
        edgecolor=(0, 1, 0),
    )
    ax.add_patch(rect)


fig, axs = plt.subplots(
    1,
    3,
    figsize=(len(columns) // 2, len(rows) // 2),
    gridspec_kw={"width_ratios": [2, 3, 3]},
)

plot_param_predictions(axs[0], "scale", rows, scale_cols, True, "Scale")
draw_rectangle(axs[0], 0, 0, 1, 7)
draw_rectangle(axs[0], 7, 1, 1, 16)

plot_param_predictions(axs[1], "loss", rows, loss_cols, False, "Loss")
draw_rectangle(axs[1], 0, 0, 1, 4)
draw_rectangle(axs[1], 5, 0, 1, 11)
# draw_rectangle(axs[1], 11, 1, 2, 5)

plot_param_predictions(axs[2], "architecture", rows, arch_cols, False, "bollocks")
axs[2].set_title("bollocks")
draw_rectangle(axs[2], 0, 0, 1, 1)
draw_rectangle(axs[2], 2, 1, 1, 1)
draw_rectangle(axs[2], 3, 2, 1, 1)
draw_rectangle(axs[2], 8, 0, 1, 1)
draw_rectangle(axs[2], 11, 1, 1, 1)
draw_rectangle(axs[2], 12, 2, 1, 1)

plt.subplots_adjust(wspace=0.1)

fig.suptitle("Predicted hyperparameter value", y=0.87, x=0.5)
plt.gca().xaxis.set_label_position("top")

plt.savefig(
    f"pretrained-model-parsing-table.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=200,
)
