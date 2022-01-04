from collections import namedtuple
from database.api.common import get_unique_row, idempotent_insert_unique_row

CropCoords = namedtuple("CropCoords", ["left", "upper", "right", "lower"])


def get_image_patch_row(image_path=None, crop=None, image_patch_id=None):
    unique_identifiers = None
    if image_path is not None:
        unique_identifiers = {
            "image_path": image_path,
            "crop_left": crop.left,
            "crop_upper": crop.upper,
            "crop_right": crop.right,
            "crop_lower": crop.lower,
        }
    else:
        unique_identifiers = {"image_patch_id": image_patch_id}
    return get_unique_row("image_patch", unique_identifiers)


def idempotent_insert_image_patch(image_patch_edict):
    new_row = image_patch_edict.copy()
    crop = new_row.pop("crop")
    new_row.update(
        {
            "crop_left": crop.left,
            "crop_upper": crop.upper,
            "crop_right": crop.right,
            "crop_lower": crop.lower,
        }
    )
    return idempotent_insert_unique_row("image_patch", new_row)
