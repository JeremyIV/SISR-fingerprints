import yaml
from yaml import Loader
import argparse

from analysis.values.values_registry import VALUES_REGISTRY
import analysis.values.model_parser_values
import analysis.values.architecture_comparison
import analysis.values.model_attribution_values
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Compute values from the analysis database, and store them in computed_values.tex"
)
parser.add_argument(
    "--value_functions",
    nargs="*",
    help="Only run these value-computing functions. Keep all other old values.",
)
args = parser.parse_args()

computed_values_yaml_path = Path("analysis/values/computed_values.yaml")
computed_values_tex_path = Path("paper/computed_values.tex")

only_compute_specific_values = (
    args.value_functions is not None and len(args.value_functions) > 0
)
computed_values = {}
if only_compute_specific_values and computed_values_yaml_path.exists():
    computed_values = yaml.load(open(computed_values_yaml_path), Loader=Loader)

##############################################################################
## Which values functions to compute?
##############################################################################

value_functions_to_run = VALUES_REGISTRY.keys()
if only_compute_specific_values:
    value_functions_to_run = args.value_functions

##############################################################################
## Compute values
##############################################################################

for value_function_name in value_functions_to_run:
    val_func = VALUES_REGISTRY.get(value_function_name)
    computed_values.update(val_func())

##############################################################################
## Save the results
##############################################################################
with open(computed_values_yaml_path, "w") as f:
    yaml.dump(computed_values, f)

with open(computed_values_tex_path, "w") as f:
    for name in sorted(computed_values):
        val = computed_values[name]
        f.write(f"\\newcommand{{\\val{name}}}{{{val}}}\n")
