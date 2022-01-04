import sqlite3
import yaml
import json
import pickle
from yaml import Loader, Dumper
from collections import namedtuple
from easydict import EasyDict as edict
from pathlib import Path
import pandas as pd

con = sqlite3.connect("database/database.sqlite3")
cur = con.cursor()

tables = yaml.load(open("database/tables.yml"), Loader=Loader)
tables = {table: edict(table_data) for table, table_data in tables.items()}

EncoderDecoderPair = namedtuple("EncoderDecoderPair", ["encode", "decode"])

col_type_encodings = {
    "integer": EncoderDecoderPair(decode=int, encode=int),
    "text": EncoderDecoderPair(decode=str, encode=str),
    "boolean": EncoderDecoderPair(decode=bool, encode=bool),
    "JSON": EncoderDecoderPair(
        decode=json.loads, encode=lambda x: json.dumps(x, sort_keys=True)
    ),
    "PICKLE": EncoderDecoderPair(decode=pickle.loads, encode=pickle.dumps),
    "PATH": EncoderDecoderPair(decode=Path, encode=str),
}


def read_sql_query(query, params=None):
    # TODO: this turns ints into numpy ints.
    # need to convert into regular int for sql to handle correctly.
    return pd.read_sql_query(query, con, params=params)


def decode_row(table, row):
    """Decodes the special column types into pythonic objects.
    Returns the database row as an EasyDict.
    """
    cols = tables[table].cols
    row_edict = edict()
    row_edict["id"] = row["id"]
    for col, col_type in cols.items():
        value = row[col]
        if col_type in col_type_encodings:
            value = col_type_encodings[col_type].decode(value)
        row_edict[col] = value
    return row_edict


def get_unique_row(table, unique_identifiers):
    predicates = [f"{col} = :{col}" for col in unique_identifiers]
    predicate = " and ".join(predicates)
    columns = ", ".join(["id"] + list(tables[table].cols))
    query = f"select {columns} from {table} where {predicate}"

    unique_identifiers = encode_row(table, unique_identifiers)
    result = read_sql_query(query, params=unique_identifiers)
    assert len(result) <= 1, f"Multiple rows returned for {query}!"
    if len(result) == 0:
        return None
    else:
        row = result.iloc[0]
        return decode_row(table, row)


def rows_are_equivalent(table, old_row, new_values):
    cols_to_ignore = tables[table].get("volatile_cols", set())
    cols_to_compare = set(tables[table].cols) - set(cols_to_ignore)
    for col in cols_to_compare:
        if old_row[col] != new_values[col]:
            return False
    return True


def get_unique_idenfiers(table, row):
    return {col: row[col] for col in tables[table].unique}


def idempotent_insert_unique_row(table, new_row):
    unique_identifiers = get_unique_idenfiers(table, new_row)
    old_row = get_unique_row(table, unique_identifiers)

    if old_row is None:
        insert_row(table, new_row)
        return get_unique_row(table, unique_identifiers).id
    else:
        new_row_with_old_id = new_row.copy()
        new_row_with_old_id["id"] = old_row.id
        if rows_are_equivalent(table, old_row, new_row):
            return old_row.id
        else:
            raise Exception(
                f"Attempting to insert non-unique row "
                + f"{new_row_with_old_id} which differs from existing row "
                + f"{old_row}. If you intend to overwrite the old rows, use "
                + "the --overwrite argument."
            )


def encode_row(table, row):
    encoded_row = {}
    table_cols = tables[table].cols
    for col, value in row.items():
        if col == "id":
            encoded_row["id"] = int(value)
            continue
        col_type = table_cols[col]
        if col_type in col_type_encodings:
            value = col_type_encodings[col_type].encode(value)
        encoded_row[col] = value
    return encoded_row


def insert_row(table, row):
    columns = ", ".join(row.keys())
    params_str = ", ".join(f":{col}" for col in row.keys())

    encoded_row = encode_row(table, row)

    query = f"insert into {table}({columns}) values ({params_str})"
    cur.execute(query, encoded_row)
