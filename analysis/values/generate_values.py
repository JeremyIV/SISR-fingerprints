import yaml
import argparse
from analysis.values.values_registry import VALUES_REGISTRY
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Compute values from the analysis database, and store them in computed_values.tex"
)
parser.add_argument(
    "--value_functions",
    nargs="*",
    help="Only run these value-computing functions. Keep all other old values.",
)

computed_values_yaml_path = Path("analysis/values/computed_values.yaml")
computed_values_tex_path = Path("paper/computed_values.tex")

computed_values = {}
if computed_values_yaml_path.exists():
    computed_values = yaml.load(open(computed_values_yaml_path))

##############################################################################
## Which values functions to compute?
##############################################################################

value_functions_to_run = VALUES_REGISTRY.keys()
if len(args.value_functions) > 0:
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
    for name in sorted(values):
        val = values[name]
        f.write(f"\\newcommand{{\\val{name}}}{{{val}}}\n")
