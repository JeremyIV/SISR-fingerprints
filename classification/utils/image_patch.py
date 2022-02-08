from skimage.metrics import peak_signal_noise_ratio
from lpips import LPIPS
from PIL import ImageOps
import cv2
import database.api as db
import numpy as np
from torchvision import transforms
import hashlib
from PIL import Image


def get_patch_hash(image):
    image = np.array(image)
    # we have about 100k images to hash.
    # 64 bits guarantees about 1e-9 probability of collision.
    return hashlib.shake_128(image.data.tobytes()).hexdigest(64)


def get_acutance(image):
    return cv2.Laplacian(np.array(ImageOps.grayscale(image)), cv2.CV_64F).var()


def get_psnr(image, reference):
    return peak_signal_noise_ratio(np.array(image), np.array(reference))


lpips_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

loss_fn_alex = LPIPS(net="alex")


def get_lpips(image, reference):
    lpips = loss_fn_alex(lpips_preprocess(image), lpips_preprocess(reference))
    lpips = float(lpips.squeeze().cpu().detach().numpy())
    return lpips


def get_generator_id(image_path):
    generator_name = image_path.parent.stem
    generator_row = db.get_unique_row("generator", {"name": generator_name})
    return generator_row.id


def update_image_patch(image, metadata):
    patch_hash = get_patch_hash(image)

    # If the image patch already exists, don't re-process it.
    old_patch = db.get_image_patch_row(
        image_path=metadata.image_path, crop=metadata.crop
    )
    if old_patch is not None:
        if old_patch.patch_hash == patch_hash:
            return old_patch.id
        else:
            raise Exception(
                f"Image patch {metadata} has changed hash! "
                + f"old: {old_patch.patch_hash:02x}, new: {patch_hash:02x}"
            )

    generator_id = get_generator_id(metadata.image_path)
    psnr = None
    lpips = None
    acutance = get_acutance(image)
    if metadata.ground_truth_path is not None:
        ground_truth = Image.open(metadata.ground_truth_path)
        ground_truth = ground_truth.crop(metadata.crop)
        psnr = get_psnr(image, ground_truth)
        lpips = get_lpips(image, ground_truth)
    return db.idempotent_insert_image_patch(
        {
            "generator_id": generator_id,
            "image_path": metadata.image_path,
            "patch_hash": patch_hash,
            "crop": metadata.crop,
            "acutance": acutance,
            "psnr": psnr,
            "lpips": lpips,
        }
    )
