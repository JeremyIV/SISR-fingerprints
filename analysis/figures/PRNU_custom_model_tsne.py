import database.api as db
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.decomposition import PCA, IncrementalPCA
from pathlib import Path
import gzip
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Generates a figure with t-sne "
    + "dimensionality reductions for custom-trained SISR models, split into "
    + "four quadrants based on scale and loss."
)

parser.add_argument(
    "--reduction",
    default="TSNE",
    help="Either TSNE or UMAP. UMAP is faster but less visually appealing.",
)

args = parser.parse_args()

if args.reduction == "TSNE":
    from sklearn.manifold import TSNE

    def embed(features):
        return TSNE(n_components=2).fit_transform(features)

elif args.reduction == "UMAP":
    import umap

    def embed(features):
        return umap.UMAP(n_components=2).fit_transform(features)

else:
    raise Exception(f"unrecognized reduction type: {args.reduction}")

SAVED_NOISE_REDIDUALS_PATH = Path("classification/classifiers/saved_noise_residuals")

classifier_name = "PRNU_SISR_custom_models"

data = db.read_and_decode_sql_query(
    f"""
    select actual, predicted, patch_hash, scale, loss
    from sisr_analysis
    where classifier_name = '{classifier_name}'
    and generator_name not like "%-pretrained"
    """
)
is_l1 = data.loss == "L1"
correct = data.actual == data.predicted

plt.rcParams.update({"font.size": 32})
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(10, 10)

for col, loss in enumerate(["L1", "Adv"]):
    for row, scale in enumerate([2, 4]):
        ax = axs[row, col]
        if col == 0:
            if scale == 2:
                ax.set_ylabel("2X")
            else:
                ax.set_ylabel("4X")
        if row == 0:
            if loss == "L1":
                ax.set_title("L₁")
            else:
                ax.set_title("Adv.")

        mask = (data.scale == scale) & (is_l1 == (loss == "L1"))
        accuracy = correct[mask].mean()
        print(f"scale: {scale}; loss: {loss}; acc: {accuracy}; count: {mask.sum()}")
        relevant_patch_hashes = data.patch_hash[mask]
        residuals = []
        for patch_hash in relevant_patch_hashes:
            noise_residual_filename = f"{patch_hash}.npy.gz"
            noise_residual_filepath = (
                SAVED_NOISE_REDIDUALS_PATH / noise_residual_filename
            )
            with gzip.GzipFile(noise_residual_filepath, "r") as f:
                residual = np.load(f)
                residuals.append(residual.flatten())
        residuals = np.array(residuals)
        relevant_features = IncrementalPCA(n_components=100).fit_transform(residuals)
        relevant_actual_labels = data.actual[mask]
        relevant_ordered_labels = sorted(set(relevant_actual_labels))
        colors = relevant_actual_labels.apply(relevant_ordered_labels.index)
        embedding = embed(relevant_features)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10, cmap="jet")
        ax.text(0.01, 0.01, f"{accuracy*100:.01f}%", transform=ax.transAxes)

for ax in fig.get_axes():
    ax.label_outer()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.02, hspace=0.02)

plt.savefig(
    "paper/figures/PRNU-model-tsne.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=400,
)
