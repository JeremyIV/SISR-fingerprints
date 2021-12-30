# sisr.py
from utils.registry import DATASET_REGISTRY
from torch.utils.data import Dataset


def is_in_dataset(sisr_model, is_test, test_param):
    pass
    # TODO


DATASET_REGISTRY.register()


class SISR_dataset(Dataset):
    def __init__(
        self,
        split_filepath=None,
        param_to_predict=None,
        is_test=False,
        random_crop=False,
        test_param=None,
    ):
        self.samples = []  # TODO
        # load the filenames from the split_filepath
        # for each SISR model in the SISR dataset
        # see if it's in the current SISR model split
        # if not, continue
        # get the label of this sisr model
        # for each image file in the SISR model directory:
        # if it's not an image, continue
        # if it's not in the split, continue
        # image_paths.append(image_path, label)
        self.ordered_labels = list(sorted(set(label for path, label in self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        # TODO
        # load the image
        # crop the image
        return image, label, image_path
