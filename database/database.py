from collections import namedtuple
import sqlite3
import json
import pandas as pd

con = sqlite3.connect("database/database.sqlite3")
cur = con.cursor()


def get_unique_row(query, params=None):
    result = pd.read_sql_query(query, con, params=params)
    assert len(result) <= 1, f"Multiple rows returned for {query}!"
    if not result:
        return None
    else:
        return result.iloc[0]


def insert_row(table, params):
    columns = ", ".join(params)
    params_str = ", ".join(f":{col}" for col in params)
    query = f"insert into {table}({columns}) values ({params_str})"
    cur.execute(query, params)


DatasetRow = namedtuple(
    "DatasetRow", ["id", "type", "name", "is_train", "ordered_labels", "opt"]
)


def get_dataset_row(name=None, dataset_id=None):
    """Takes in a dataset name XOR a dataset id,
    and returns an object with that dataset's information from the database.
    """
    assert (name is None) != (dataset_id is None), "must specify name XOR dataset ID."

    condition = "name = ?" if name is not None else "id = ?"
    fill_value = name or dataset_id
    query = f"""select 
        id,
        type,
        name,
        is_train,
        ordered_labels,
        opt
    from datasets
    where {condition}
    """
    result = get_unique_row(query, [fill_value])
    if result is None:
        return None
    else:
        ordered_labels = json.loads(result.ordered_labels)
        return DatasetRow(
            id=result.id,
            type=result.type,
            name=result.name,
            is_train=result.is_train,
            ordered_labels=ordered_labels,
            opt=json.loads(result.opt),
        )


def add_dataset(
    dataset_type=None, name=None, is_train=None, ordered_labels=None, opt=None
):
    """Adds the specified dataset to the database.
    If a dataset with this name already exists, assert that it is identical
    to the one we're attempting to add.
    Returns the ID of the dataset.
    """
    dataset_row = get_dataset_row(name=name)
    if dataset_row is None:
        insert_row(
            "datasets",
            {
                "type": dataset_type,
                "name": name,
                "is_train": is_train,
                "ordered_labels": json.dumps(ordered_labels),
                "opt": json.dumps(opt),
            },
        )
        return get_dataset_row(name=name).id
    else:
        new_row = DatasetRow(
            id=dataset_row.id,
            type=dataset_type,
            name=name,
            is_train=is_train,
            ordered_labels=ordered_labels,
            opt=opt,
        )
        assert (
            dataset_row == new_row
        ), f"New dataset ({new_row}) differs from existing row ({dataset_row})"
        return dataset_row.id


CropCoords = namedtuple("CropCoords", ["left", "upper", "right", "lower"])

ImagePatchRow = namedtuple(
    "ImagePatchRow",
    [
        "id",
        "generator_id",
        "image_path",
        "patch_hash",
        "crop_upper",
        "crop_left",
        "crop_lower",
        "crop_right",
        "acutance",
        "psnr",
        "lpips",
    ],
)


def get_image_patch_row(image_path=None, crop=None, image_patch_id=None):
    """Get the given image patch. Either by image_path and crop coords, or by id."""
    assertion_message = (
        "Specify either the image path and crop coords, XOR the image patch id"
    )
    condition = None
    parameters = None
    if image_patch_id is None:
        assert image_path is not None and crop is not None, assertion_message
        condition = """image_path = :image_path 
            and crop_upper = :crop_upper
            and crop_left = :crop_left
            and crop_lower = :crop_lower
            and crop_right = :crop_right"""
        parameters = {
            "crop_upper": crop.upper,
            "crop_left": crop.left,
            "crop_lower": crop.lower,
            "crop_right": crop.right,
        }
    else:
        assert image_path is None and crop is None, assertion_message
        condition = "id = :id"
        parameters = {"id": image_patch_id}

    query = f"""select
            id,
            generator_id,
            image_path,
            patch_hash,
            crop,
            acutance,
            psnr,
            lpips
        from image_patches
        where {condition}
    """

    result = get_unique_row(query, parameters)

    if result is None:
        return None
    else:
        return ImagePatchRow(
            id=result.image_patch_id,
            generator_id=result.generator_id,
            image_path=result.image_path,
            patch_hash=result.patch_hash,
            crop=CropCoords(
                upper=result.crop_upper,
                left=result.crop_left,
                lower=result.crop_lower,
                right=result.crop_right,
            ),
            acutance=result.acutance,
            psnr=result.psnr,
            lpips=result.lpips,
        )


def add_image_patch(
    generator=None,
    image_path=None,
    patch_hash=None,
    crop=None,
    acutance=None,
    psnr=None,
    lpips=None,
):
    generator_id = get_unique_row(
        "select id from generators where name = :name", {"name": generator}
    ).id
    row = get_image_patch(image_path=image_path, crop=crop)
    if row is None:
        insert_row(
            "image_patches",
            {
                "generator_id": generator_id,
                "image_path": image_path,
                "patch_hash": patch_hash,
                "crop_upper": crop.upper,
                "crop_left": crop.left,
                "crop_lower": crop.lower,
                "crop_right": crop.right,
                "acutance": acutance,
                "psnr": psnr,
                "lpips": lpips,
            },
        )

        return get_image_patch(image_path=image_path, crop=crop).id

    else:
        new_row = ImagePatchRow(
            id=row.id,
            generator_id=generator_id,
            image_path=image_path,
            patch_hash=patch_hash,
            crop=crop,
            acutance=acutance,
            psnr=psnr,
            lpips=lpips,
        )
        assert (
            generator_id == row.generator_id
            and image_path == row.image_path
            and crop == row.crop
            and patch_hash == row.hash
        ), f"New image patch ({new_row}) doesn't match existing image patch ({row})"
        return row.id


PredictionRow = namedtuple(
    "PredictionRow",
    [
        "id",
        "classifier_id",
        "image_patch_id",
        "dataset_id",
        "actual_label",
        "predicted_label",
        "feature",
        "label_probabilities",
    ],
)


def get_prediction(
    classifier_name=None,
    classifier_id=None,
    image_patch_id=None,
    image_path=None,
    crop=None,
):
    assert (classifier_name is None) != (
        classifier_id is None
    ), "must specify either classifier_name XOR classifier_id."
    if classifier_id is None:
        classifier_id = get_classifier(name=classifier_name).id

    if image_patch_id is None:
        image_patch_id = get_image_patch(image_path=image_path, crop=crop).id
    else:
        assert (
            image_path is None and crop is None
        ), "must specify either image_patch_id XOR (image_path and crop)"

    query = """select
            id,
            classifier_id,
            image_patch_id,
            dataset_id,
            actual_label,
            predicted_label,
            feature,
            class_probabilities
        from predictions
        where classifier_id = :classifier_id
          and image_patch_id = :image_patch_id
        """

    row = get_unique_row(
        query, params={"classifier_id": classifier_id, "image_patch_id": image_patch_id}
    )

    if row is None:
        return None
    else:
        feature = row.feature and pickle.loads(row.feature)
        label_probabilities = pickle.loads(row.label_probabilities)
        return PredictionRow(
            id=row.id,
            classifier_id=classifier_id,
            image_patch_id=image_patch_id,
            dataset_id=row.dataset_id,
            actual_label=row.actual_label,
            predicted_label=row.predicted_label,
            feature=feature,
            label_probabilities=label_probabilities,
        )


def add_prediction(
    classifier_name=None,
    image_patch_id=None,
    dataset_name=None,
    actual_label=None,
    predicted_label=None,
    class_probabilities=None,
    feature=None,
):

    classifier_id = get_classifier_row(name=classifier_name).id
    dataset_id = get_dataset_row(name=dataset_name).id
    feature_blob = pickle.dumps(feature)
    probs_blob = pickle.dumps(class_probabilities)

    insert_row(
        "predictions",
        {
            "classifier_id": classifier_id,
            "image_patch_id": image_patch_id,
            "dataset_id": dataset_id,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "feature": feature_blob,
            "class_probabilities": probs_blob,
        },
    )


def add_dataset_row(
    type=None,
    name=None,
    is_test=None,
    label_param=None,
    ordered_labels=None,
    test_param=None,
    opt=None,
):
    pass


def get_classifier_row(name):
    pass


def add_classifier_row(
    name=None,
    type=None,
    dataset_slice=None,
    test_param_value=None,
    param_to_predict=None,
    ordered_labels=None,
    opt=None,
):
    pass
