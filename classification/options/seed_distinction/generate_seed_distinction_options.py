import yaml
import utils
from pathlib import Path

parser_opt_dir = Path("classification/options/seed_distinction")


def get_seed_triplets():
    models = utils.get_sisr_model_names(dataset=[utils.div2k])
    seed_triplets = {}
    for model in models:
        seedless_prefix = model[:-3]
        if seedless_prefix not in seed_triplets:
            seed_triplets[seedless_prefix] = []
        seed_triplets[seedless_prefix].append(model)
    return list(seed_triplets.keys())


seed_triplets = get_seed_triplets()
for seed_triplet in seed_triplets:
    classifier_name = f"PRNU_{seed_triplet}_seed_distinction"
    opt = {
        "classifier": {
            "type": "PRNU",
            "name": classifier_name,
        },
        "training_dataset": {
            "type": "SISR",
            "name": f"{classifier_name}_train",
            "random_crop": False,
            "include_pretrained": False,
            "include_custom_trained": True,
            "sisr_model_list": [
                f"{seed_triplet}-s1",
                f"{seed_triplet}-s2",
                f"{seed_triplet}-s3",
            ],
        },
        "validation_dataset": {
            "name": f"{classifier_name}_val",
            "type": "SISR",
            "random_crop": False,
            "include_pretrained": False,
            "include_custom_trained": True,
            "sisr_model_list": [
                f"{seed_triplet}-s1",
                f"{seed_triplet}-s2",
                f"{seed_triplet}-s3",
            ],
        },
        "evaluation_datasets": [
            {
                "name": f"{classifier_name}_test",
                "type": "SISR",
                "random_crop": False,
                "include_pretrained": False,
                "include_custom_trained": True,
                "sisr_model_list": [
                    f"{seed_triplet}-s1",
                    f"{seed_triplet}-s2",
                    f"{seed_triplet}-s3",
                ],
            }
        ],
    }
    opt_path = parser_opt_dir / f"{classifier_name}.yaml"
    with open(opt_path, "w") as f:
        yaml.dump(opt, f)
