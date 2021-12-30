from collections import namedtuple

ImagePatchMetadata = namedtuple(
    "ImagePatchMetadata", ["image_path", "crop", "ground_truth_path"]
)
