from skimage.metrics import peak_signal_noise_ratio
from lpips import LPIPS
from PIL import ImageOps
import cv2


def get_acutance(image):
    return cv2.Laplacian(np.array(ImageOps.grayscale(img)), cv2.CV_64F).var()


def get_psnr(image, reference):
    psnr = peak_signal_noise_ratio(img, hr_img)


lpips_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

loss_fn_alex = LPIPS(net="alex")


def get_lpips(image, reference):
    lpips = loss_fn_alex(preprocess(image), preprocess(reference))
    lpips = float(lpips.squeeze().cpu().detach().numpy())
    return lpips


def update_image_patch(image, metadata):
    psnr = None
    lpips = None
    acutance = get_acutance(image)
    if metadata.ground_truth_path is not None:
        ground_truth = Image.open(metadata.ground_truth_path)
        ground_truth = ground_truth.crop(metadata.crop)
        psnr = get_psnr(image, ground_truth)
        lpips = get_lpips(image, ground_truth)
    return db.update_image_patch(
        generator=metadata.generator,
        image_path=metadata.image_path,
        crop=metadata.crop,
        acutance=acutance,
        psnr=psnr,
        lpips=lpips,
    )
