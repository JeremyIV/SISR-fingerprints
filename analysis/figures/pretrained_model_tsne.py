import database.api as db


pretrained_clsfr_data = db.read_sql_query(
    """
	select actual, predicted, feature
	from sisr_analysis
	where classifier_name = TODO"""
)

accuracy = (pretrained_clsfr_data.actual == pretrained_clsfr_data.predicted).mean()
embedding = TSNE(n_components=2).fit_transform(pretrained_clsfr_data.feature)
del pretrained_clsfr_data.feature

custom_clsfr_data = db.read_sql_query(
    '''
	select actual, predicted, feature
	from sisr_analysis
	where classifier_name = TODO
	and generator_name like "%-pretrained"'''
)
ood_embedding = TSNE(n_components=2).fit_transform(custom_clsfr_data.feature)
del custom_clsfr_data.feature

# TODO: update these lists.
classes_by_scale_then_loss_then_name = [
    "EDSR-div2k-x2-L1-NA-pretrained",
    "LIIF-div2k-x2-L1-NA-pretrained",
    "RCAN-div2k-x2-L1-NA-pretrained",
    "RDN-div2k-x2-L1-NA-pretrained",
    "SRFBN-NA-x2-L1-NA-pretrained",
    "SwinIR-div2k-x2-L1-NA-pretrained",
    "Real_ESRGAN-div2k-x2-GAN-NA-pretrained",
    "DRN-div2k-x4-L1-NA-pretrained",
    "EDSR-div2k-x4-L1-NA-pretrained",
    "LIIF-div2k-x4-L1-NA-pretrained",
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

class_markers = {
    "EDSR-div2k-x2-L1-NA-pretrained": "o",
    "LIIF-div2k-x2-L1-NA-pretrained": "o",
    "RCAN-div2k-x2-L1-NA-pretrained": "o",
    "RDN-div2k-x2-L1-NA-pretrained": "o",
    "SRFBN-NA-x2-L1-NA-pretrained": "o",
    "SwinIR-div2k-x2-L1-NA-pretrained": "o",
    "Real_ESRGAN-div2k-x2-GAN-NA-pretrained": "D",
    "DRN-div2k-x4-L1-NA-pretrained": "^",
    "EDSR-div2k-x4-L1-NA-pretrained": "^",
    "LIIF-div2k-x4-L1-NA-pretrained": "^",
    "RCAN-div2k-x4-L1-NA-pretrained": "^",
    "RDN-div2k-x4-L1-NA-pretrained": "^",
    "SAN-div2k-x4-L1-NA-pretrained": "^",
    "SRFBN-NA-x4-L1-NA-pretrained": "^",
    "SwinIR-div2k-x4-L1-NA-pretrained": "^",
    "proSR-div2k-x4-L1-NA-pretrained": "^",
    "ESRGAN-NA-x4-ESRGAN-NA-pretrained": "s",
    "EnhanceNet-NA-x4-EnhanceNet-NA-pretrained": "s",
    "Real_ESRGAN-div2k-x4-GAN-NA-pretrained": "s",
    "SwinIR-div2k-x4-GAN-NA-pretrained": "s",
    "NCSR-div2k-x4-NCSR_GAN-NA-pretrained": "s",
    "proSR-div2k-x4-ProSRGAN-NA-pretrained": "s",
    "SPSR-div2k-x4-SPSR_GAN-NA-pretrained": "s",
}

class_colors = {}
marker_groups = {}
for cls, marker in class_markers.items():
    if marker not in marker_groups:
        marker_groups[marker] = []
    marker_groups[marker].append(cls)

for group in marker_groups.values():
    norm = colors.Normalize(vmin=0, vmax=len(group), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="jet")
    for index, cls in enumerate(group):
        class_colors[cls] = mapper.to_rgba(index)

to_nice_names = {
    "EDSR-div2k-x2-L1-NA-pretrained": "EDSR-2x",
    "LIIF-div2k-x2-L1-NA-pretrained": "LIIF-2x",
    "RCAN-div2k-x2-L1-NA-pretrained": "RCAN-2x",
    "RDN-div2k-x2-L1-NA-pretrained": "RDN-2x",
    "SRFBN-NA-x2-L1-NA-pretrained": "SRFBN-2x",
    "SwinIR-div2k-x2-L1-NA-pretrained": "SwinIR-2x",
    "Real_ESRGAN-div2k-x2-GAN-NA-pretrained": "Real ESRGAN-2x",
    "DRN-div2k-x4-L1-NA-pretrained": "DRN-4x",
    "EDSR-div2k-x4-L1-NA-pretrained": "EDSR-4x",
    "LIIF-div2k-x4-L1-NA-pretrained": "LIIF-4x",
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


def plot_pretrained_tnse(ax, embedding, actual_class):
    for model in classes_by_scale_then_loss_then_name:
        color = class_colors[model]
        marker = class_markers[model]
        emb = embedding[actual_class == model]
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=np.array([color] * len(emb)),
            s=17,
            marker=marker,
            alpha=0.7,
            label=to_nice_names[model],
        )
    ax.set_xticks([])
    ax.set_yticks([])


fig, axs = plt.subplots(1, 2)
plt.rcParams.update({"font.size": 15})
fig.set_size_inches(10, 5)
axs[0].set_title("Pretrained model classifier")
plot_pretrained_tnse(axs[0], embedding, pretrained_clsfr_data.actual)
axs[0].text(0.01, 0.05, f"{accuracy*100:.01f}%", transform=axs[0].transAxes)
axs[1].set_title("Custom model classifier")
plot_pretrained_tnse(axs[1], ood_embedding, custom_clsfr_data.actual)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.0, hspace=0.0)
axs[1].legend(
    loc="upper center", fancybox=True, bbox_to_anchor=(0, 0.05), ncol=4, shadow=True
)
plt.savefig(
    "closed-vs-open-pretrained-tsne.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=200,
)
