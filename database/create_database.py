import sqlite3
import yaml
from yaml import Loader, Dumper
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    description="Create a fresh copy of the SISR fingerprint analysis database."
)
parser.add_argument(
    "-v", "--verbose"
    action="store_true",
    help="verbose mode: prints out all the sqlite commands "
    + "executed to create the database.",
)
args = parser.parse_args()

def vprint(*args, **kwargs):
    if args.v:
        print(*args, **kwargs)

db_path = Path("database/database.sqlite3")
if db_path.exists():
    print("database already exists. Are you sure you want to overwrite it? [y/N]: ")
    while True:
        response = input()
        if response.lower() == "y":
            db_path.unlink()
            break
        elif response.lower() == "n" or response.lower() == "":
            print("exiting.")
            exit(0)
        else:
            print(f"unrecognized response {response}. enter Y or N: ")

con = sqlite3.connect(db_path)
cur = con.cursor()

tables = yaml.load(open("database/tables.yml"), Loader=Loader)

special_types = {"JSON": "text", "PICKLE": "blob", "PATH": "text"}

for table, data in tables.items():
    elements = ["id integer primary key"]
    for col, col_type in data["cols"].items():
        if col_type in special_types:
            col_type = special_types[col_type]
        elements.append(f"{col} {col_type}")
    if "unique" in data:
        unique_cols_str = ", ".join(data["unique"])
        elements.append(f"constraint unq unique({unique_cols_str})")

    elements_str = ",\n".join(elements)
    command = f"create table {table}(\n{elements_str}\n);"
    vprint(command)
    cur.execute(command)
create_views_sql = open("database/create_views.sql").read()
vprint(create_views_sql)
cur.execute(create_views_sql)
con.commit()
