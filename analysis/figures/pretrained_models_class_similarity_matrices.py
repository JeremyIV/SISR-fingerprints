import database.api as db
import numpy as np
from matplotlib import colors, cm
from matplotlib import rcParams
import matplotlib.pyplot as plt
import torch

ordered_classes = [
    "EDSR-div2k-x2-L1-NA-pretrained",
    "LIIF-div2k-x2-L1-NA-pretrained",
    "NLSN-div2k-x2-L1-NA-pretrained",
    "RCAN-div2k-x2-L1-NA-pretrained",
    "RDN-div2k-x2-L1-NA-pretrained",
    "SRFBN-NA-x2-L1-NA-pretrained",
    "SwinIR-div2k-x2-L1-NA-pretrained",
    "Real_ESRGAN-div2k-x2-GAN-NA-pretrained",
    "DRN-div2k-x4-L1-NA-pretrained",
    "EDSR-div2k-x4-L1-NA-pretrained",
    "LIIF-div2k-x4-L1-NA-pretrained",
    "NLSN-div2k-x4-L1-NA-pretrained",
    "RCAN-div2k-x4-L1-NA-pretrained",
    "RDN-div2k-x4-L1-NA-pretrained",
    "SAN-div2k-x4-L1-NA-pretrained",
    "SRFBN-NA-x4-L1-NA-pretrained",
    "SwinIR-div2k-x4-L1-NA-pretrained",
    "proSR-div2k-x4-L1-NA-pretrained",
    "ESRGAN-NA-x4-ESRGAN-NA-pretrained",
    "EnhanceNet-NA-x4-EnhanceNet-NA-pretrained",
    "Real_ESRGAN-div2k-x4-GAN-NA-pretrained",
    "SwinIR-div2k-x4-GAN-NA-pretrained",
    "NCSR-div2k-x4-NCSR_GAN-NA-pretrained",
    "proSR-div2k-x4-ProSRGAN-NA-pretrained",
    "SPSR-div2k-x4-SPSR_GAN-NA-pretrained",
]

to_nice_names = {
    "EDSR-div2k-x2-L1-NA-pretrained": "EDSR-2x",
    "LIIF-div2k-x2-L1-NA-pretrained": "LIIF-2x",
    "NLSN-div2k-x2-L1-NA-pretrained": "NLSN-2x",
    "RCAN-div2k-x2-L1-NA-pretrained": "RCAN-2x",
    "RDN-div2k-x2-L1-NA-pretrained": "RDN-2x",
    "SRFBN-NA-x2-L1-NA-pretrained": "SRFBN-2x",
    "SwinIR-div2k-x2-L1-NA-pretrained": "SwinIR-2x",
    "Real_ESRGAN-div2k-x2-GAN-NA-pretrained": "Real ESRGAN-2x",
    "DRN-div2k-x4-L1-NA-pretrained": "DRN-4x",
    "EDSR-div2k-x4-L1-NA-pretrained": "EDSR-4x",
    "LIIF-div2k-x4-L1-NA-pretrained": "LIIF-4x",
    "NLSN-div2k-x4-L1-NA-pretrained": "NLSN-4x",
    "RCAN-div2k-x4-L1-NA-pretrained": "RCAN-4x",
    "RDN-div2k-x4-L1-NA-pretrained": "RDN-4x",
    "SAN-div2k-x4-L1-NA-pretrained": "SAN-4x",
    "SRFBN-NA-x4-L1-NA-pretrained": "SRFBN-4x",
    "SwinIR-div2k-x4-L1-NA-pretrained": "SwinIR-4x",
    "proSR-div2k-x4-L1-NA-pretrained": "proSR-4x",
    "ESRGAN-NA-x4-ESRGAN-NA-pretrained": "ESRGAN-4x",
    "EnhanceNet-NA-x4-EnhanceNet-NA-pretrained": "EnhanceNet-4x",
    "Real_ESRGAN-div2k-x4-GAN-NA-pretrained": "Real ESRGAN-4x",
    "SwinIR-div2k-x4-GAN-NA-pretrained": "SwinIR-GAN-4x",
    "NCSR-div2k-x4-NCSR_GAN-NA-pretrained": "NCSR-4x",
    "proSR-div2k-x4-ProSRGAN-NA-pretrained": "proSR-4x",
    "SPSR-div2k-x4-SPSR_GAN-NA-pretrained": "SPSR-4x",
}
id_matrices = []
ood_matrices = []
for classifier_prefix in ["", "seed_2_", "seed_3_"]:
    custom_clsfr_data = db.read_and_decode_sql_query(
        f'''
        select actual, predicted, feature
        from sisr_analysis
        where classifier_name = '{classifier_prefix}ConvNext_CNN_SISR_custom_models'
        and generator_name like "%-pretrained"'''
    )
    ood_features = np.array(list(custom_clsfr_data.feature))

    pretrained_clsfr_data = db.read_and_decode_sql_query(
        f'''
        select actual, predicted, feature
        from sisr_analysis
        where classifier_name = '{classifier_prefix}ConvNext_CNN_SISR_pretrained_models'
        and generator_name like "%-pretrained"'''
    )

    id_features = np.array(list(pretrained_clsfr_data.feature))

    def avg_correlation(class_a_features, class_b_features, num_samples=1000):
        pass
        num_class_a_feats = len(class_a_features)
        num_class_b_feats = len(class_b_features)
        corr_coeffs = []
        for sample in range(num_samples):
            a_index = np.random.randint(num_class_a_feats)
            feature_a = class_a_features[a_index]
            b_index = np.random.randint(num_class_b_feats)
            feature_b = class_b_features[b_index]
            corr_coeff = np.corrcoef(feature_a, feature_b)
            corr_coeffs.append(corr_coeff)
        return np.mean(corr_coeffs)

    def distance(f1, f2):
        return np.sqrt((f1 - f2) ** 2)

    def intra_vs_inter_class_distance(class_a_features, class_b_features):
        # TODO
        # calculate the centroids
        centroid_a = class_a_features.mean(axis=0, keepdims=True)
        centroid_b = class_b_features.mean(axis=0, keepdims=True)
        # calculate mean distance from each point a to centroid a
        class_a_to_centroid_a = centroid_a - class_a_features
        class_a_to_centroid_b = centroid_b - class_a_features
        # calculate the mean distance from each point a to centroid b
        intra_class_distance = (class_a_to_centroid_a ** 2).sum(axis=1).mean()
        inter_class_distance = (class_a_to_centroid_b ** 2).sum(axis=1).mean()

        return intra_class_distance / inter_class_distance

    def intra_vs_inter_class_distance_legacy(
        class_a_features, class_b_features, num_samples=1000
    ):
        """Ratio of average intra-class vs. inter-class distance.
        If these classes are the same, the ratio should be approximately 1
        As these classes become more separable, the ratio should tend towards zero.
        """
        num_class_a_feats = len(class_a_features)
        num_class_b_feats = len(class_b_features)
        intra_class_distances = []
        inter_class_distances = []

        # TODO: calculate this exactly, instead of with random sampling.
        # Should be possible due to the linearity of expectations.
        for sample in range(num_samples):
            a_1_index = np.random.randint(num_class_a_feats)
            a_2_index = np.random.randint(num_class_a_feats)
            b_index = np.random.randint(num_class_b_feats)
            feature_a_1 = class_a_features[a_1_index]
            feature_a_2 = class_a_features[a_2_index]
            feature_b = class_b_features[b_index]

            intra_class_distance = distance(feature_a_1, feature_a_2)
            inter_class_distance = distance(feature_a_1, feature_b)
            intra_class_distances.append(intra_class_distance)
            inter_class_distances.append(inter_class_distance)

        return np.mean(intra_class_distances) / np.mean(inter_class_distances)

    def make_affinity_matrix(
        features, classes, ordered_classes, affinity_score=avg_correlation
    ):
        num_classes = len(ordered_classes)
        affinity_matrix = np.zeros((num_classes, num_classes))
        for i, class_a in enumerate(ordered_classes):
            # get the features in class a
            class_a_features = features[classes == class_a]
            # for each class b in this set
            for j, class_b in enumerate(ordered_classes):
                class_b_features = features[classes == class_b]
                affinity = affinity_score(class_a_features, class_b_features)
                affinity_matrix[i, j] += affinity
                affinity_matrix[j, i] += affinity

        return affinity_matrix / 2

    ordered_nice_names = [to_nice_names[c] for c in ordered_classes]
    id_affinity_matrix = make_affinity_matrix(
        id_features,
        pretrained_clsfr_data.actual,
        ordered_classes,
        affinity_score=intra_vs_inter_class_distance,
    )
    ood_affinity_matrix = make_affinity_matrix(
        ood_features,
        custom_clsfr_data.actual,
        ordered_classes,
        affinity_score=intra_vs_inter_class_distance,
    )

    id_matrices.append(id_affinity_matrix)
    ood_matrices.append(ood_affinity_matrix)

id_affinity_matrix = np.mean(id_matrices, axis=0)
ood_affinity_matrix = np.mean(ood_matrices, axis=0)
torch.save(
    {"id": id_affinity_matrix, "ood": ood_affinity_matrix},
    f"analysis/figures/pretrained_model_class_affinity_matrices.pt",
)

# TODO: in a notebook, load all three seeds in, average them,
# then run the code below to generate the aggregate figure.

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(21 * 0.75, 10 * 0.75), sharey=True)
id_ax = axs[0]
ood_ax = axs[1]
vmin = min(id_affinity_matrix.min(), ood_affinity_matrix.min())
im = id_ax.imshow(id_affinity_matrix, cmap="jet", vmin=vmin, vmax=1)
id_ax.set_xticks(np.arange(25))
id_ax.set_xticklabels(ordered_nice_names, rotation=90)
id_ax.set_yticks(np.arange(25))
id_ax.set_yticklabels(ordered_nice_names)
id_ax.set_title("Pretrained Model Classifier", fontsize=24)
ood_ax.imshow(ood_affinity_matrix, cmap="jet", vmin=vmin, vmax=1)
ood_ax.set_xticks(np.arange(25))
ood_ax.set_xticklabels(ordered_nice_names, rotation=90)
ood_ax.set_title("Custom Model Classifier", fontsize=24)
fig.subplots_adjust(wspace=0, right=0.9)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.savefig(
    f"paper/figures/pretrained-class-similarity-matrices.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=200,
)
