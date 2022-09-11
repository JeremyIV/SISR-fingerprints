from PIL import Image
from pathlib import Path
import utils

dataset_dir = Path("classification/datasets/data/SISR")
img_filename = "3644634744_56aa1c7d32.png"

destination_dir = Path("paper/figures/SISR_patch_samples")
destination_dir.mkdir(parents=True, exist_ok=True)

# TODO: copy over LR_2x and LR_4x patches as well

width = 64
height = width
upper = 190
left = 340
right = left + width
lower = upper + height
crop_box = (left, upper, right, lower)

custom_sisr_models = utils.get_sisr_model_names(
    dataset=["div2k"],
    seed=[1],
)
print(custom_sisr_models)

pretrained_sisr_models = []
for sisr_model_dir in dataset_dir.iterdir():
    if not sisr_model_dir.is_dir():
        continue
    sisr_model = sisr_model_dir.stem
    if sisr_model.endswith("-pretrained"):
        pretrained_sisr_models.append(sisr_model)

sisr_models = custom_sisr_models + pretrained_sisr_models

for sisr_model in sisr_models:
    img_filepath = dataset_dir / sisr_model / img_filename
    img = Image.open(img_filepath)
    img = img.crop(crop_box)
    img_patch_filepath = destination_dir / f"{sisr_model}.png"
    img.save(img_patch_filepath)
