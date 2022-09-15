import database.api as db
import numpy as np


def fmt(acc):
    return f"{acc*100:.1f}"


def unfmt(acc_str):
    return float(acc_str) / 100


latex_friendly = {
    "2": "TwoX",
    "4": "FourX",
    "L1": "LOne",
    "div2k": "Div",
    "flickr2k": "Flickr",
    "quarter_div2k": "QuarterDiv",
    "quarter_flickr2k": "QuarterFlickr",
    "VGG_GAN": "VGG",
    "ResNet_GAN": "ResNet",
    "3": "three",
}


def latexify(string):
    string = str(string)
    return latex_friendly.get(string, string)


def get_acc_val(query):
    return fmt(db.read_sql_query(query).acc[0])


def aggregate_across_seeds(values_list):
    all_values = {}
    for seed_values in values_list:
        for key, value in seed_values.items():
            if key not in all_values:
                all_values[key] = []
            all_values[key].append(unfmt(value))

    aggregated_values = {}
    for key, vals in all_values.items():
        aggregated_values[f"{key}Mean"] = fmt(np.mean(vals))
        aggregated_values[f"{key}Std"] = fmt(np.std(vals))
    return aggregated_values
