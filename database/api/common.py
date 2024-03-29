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
    return pd.read_sql_query(query, con, params=params)


default_col_types = {
    "generator_name": "text",
    "classifier_name": "text",
    "classifier_path": "PATH",
    "classifier_type": "text",
    "classifier_opt": "JSON",
    "dataset_type": "text",
    "dataset_name": "text",
}

for table_schema in tables.values():
    for col, col_type in table_schema.cols.items():
        default_col_types[col] = col_type


def read_and_decode_sql_query(query, params=None, col_types=default_col_types):
    raw_result = read_sql_query(query, params)

    decoded_result = {}
    for column, values in raw_result.iteritems():
        if column in col_types:
            col_type = col_types[column]
            decode = col_type_encodings[col_type].decode
            values = values.apply(decode)
        decoded_result[column] = values

    return pd.DataFrame(data=decoded_result)


def decode_value(value, col_type):
    if col_type in col_type_encodings and value is not None:
        return col_type_encodings[col_type].decode(value)
    return value


def decode_row(table, row):
    """Decodes the special column types into pythonic objects.
    Returns the database row as an EasyDict.
    """
    cols = tables[table].cols
    row_edict = edict()
    row_edict["id"] = row["id"]
    for col, col_type in cols.items():
        value = row[col]
        row_edict[col] = decode_value(value, col_type)
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


def equality_operator(a, b):
    return a == b


def filter_none_vals(d):
    d2 = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            d2[k] = filter_none_vals[v]
        else:
            d2[k] = v
    return d2


def dict_equality_operator_ignoring_none_vals(a, b):
    if a == b:
        return True
    filtered_a = filter_none_vals(a)
    filtered_b = filter_none_vals(b)
    return filtered_a == filtered_b


def get_equality_operator_for(table, col):
    if col == "opt":
        return dict_equality_operator_ignoring_none_vals
    else:
        return equality_operator


def rows_are_equivalent(table, old_row, new_values):
    cols_to_ignore = tables[table].get("volatile_cols", set())
    cols_to_compare = set(tables[table].cols) - set(cols_to_ignore)
    for col in cols_to_compare:
        equal = get_equality_operator_for(table, col)
        if not equal(old_row[col], new_values.get(col)):
            print(f"{col} did not match")
            return False
    return True


def get_unique_idenfiers(table, row):
    return {col: row[col] for col in tables[table].unique}


def idempotent_insert_unique_row(table, new_row):
    table_schema = tables[table]
    if "extends" in table_schema:
        # recursively create a parent row first
        parent_table = table_schema.extends
        parent_schema = tables[parent_table]
        parent_row = {}
        child_row = {}
        for col, val in new_row.items():
            if col in parent_schema.cols:
                parent_row[col] = val
            else:
                child_row[col] = val
        parent_id = idempotent_insert_unique_row(parent_table, parent_row)
        child_row[f"{parent_table}_id"] = parent_id
        new_row = child_row

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
        if col_type in col_type_encodings and value is not None:
            value = col_type_encodings[col_type].encode(value)
        encoded_row[col] = value
    return encoded_row


def insert_row(table, row):
    columns = ", ".join(row.keys())
    params_str = ", ".join(f":{col}" for col in row.keys())

    encoded_row = encode_row(table, row)

    query = f"insert into {table}({columns}) values ({params_str})"
    cur.execute(query, encoded_row)
