import database.api as db


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
