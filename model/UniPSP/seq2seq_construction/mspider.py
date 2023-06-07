import os
import torch
import random
import re
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from third_party.miscs.bridge_content_encoder import get_database_matches

"""
This part of seq2seq construction of spider dataset was partly borrowed from PICARD model.
https://github.com/ElementAI/picard

And we followed their configuration of normalization and serialization.
their configuration is as followed:
{
    "source_prefix": "",
    "schema_serialization_type": "peteshaw",
    "schema_serialization_randomized": false,
    "schema_serialization_with_db_id": true,
    "schema_serialization_with_db_content": true,
    "normalize_query": true,
    "target_with_db_id": true,
}
"""


def spider_get_input(
        question: str,
        serialized_schema: str,
        prefix: str,
    ) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()


def spider_get_target(
        query: str,
        db_id: str,
        normalize_query: bool,
        target_with_db_id: bool,
    ) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def spider_add_serialized_schema(ex: dict, args) -> dict:
    serialized_schema = serialize_schema(
        question=ex["question"][args.dataset.language],
        db_path=ex["db_path"],
        db_id=ex["db_id"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        schema_serialization_type="peteshaw",
        schema_serialization_randomized=False,
        schema_serialization_with_db_id=True,
        schema_serialization_with_db_content=True,
        normalize_query=True,
    )
    return {"serialized_schema": serialized_schema}


def spider_pre_process_function(batch: dict, args):
    prefix = ""

    inputs = [
        spider_get_input(
            question=question, serialized_schema=serialized_schema, prefix=prefix
        )
        for question, serialized_schema in zip(
            batch["question"][args.dataset.language], batch["serialized_schema"]
        )
    ]

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=True,
            target_with_db_id=args.seq2seq.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    return zip(inputs, targets)


def spider_pre_process_one_function(item: dict, args):
    prefix = ""

    seq_out = spider_get_target(
        query=item["query"],
        db_id=item["db_id"],
        normalize_query=True,
        target_with_db_id=args.seq2seq.target_with_db_id,
    )

    return prefix + item["question"][args.dataset.language].strip(), seq_out


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(
            r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s
        )

    return comma_fix(white_space_fix(lower(query)))


def serialize_schema(
        question: str,
        db_path: str,
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        schema_serialization_type: str = "peteshaw",
        schema_serialization_randomized: bool = False,
        schema_serialization_with_db_id: bool = True,
        schema_serialization_with_db_content: bool = False,
        normalize_query: bool = True,
    ) -> str:
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError

    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(
                    column=column_name_str, values=value_sep.join(matches)
                )
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema


def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex["db_table_names"],
                "db_column_names": ex["db_column_names"],
                "db_column_types": ex["db_column_types"],
                "db_primary_keys": ex["db_primary_keys"],
                "db_foreign_keys": ex["db_foreign_keys"],
            }
    return schemas


"""
    Wrap the raw dataset into the seq2seq one.
    And the raw dataset item is formatted as
    {
        "query": sample["query"],
        "question": sample["question"],
        "db_id": db_id,
        "db_path": db_path,
        "db_table_names": schema["table_names_original"],
        "db_column_names": [
            {"table_id": table_id, "column_name": column_name}
            for table_id, column_name in schema["column_names_original"]
        ],
        "db_column_types": schema["column_types"],
        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
        "db_foreign_keys": [
            {"column_id": column_id, "other_column_id": other_column_id}
            for column_id, other_column_id in schema["foreign_keys"]
        ],
    }
    """


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 2:
            raise AssertionError("Train, Dev sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets["validation"])

        return train_dataset, dev_dataset


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'spider_train.cache')
        if os.path.exists(cache_path):
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in self.raw_datasets:
                extend_data = deepcopy(raw_data)
                extend_data.update(spider_add_serialized_schema(extend_data, args=self.args))

                question, seq_out = spider_pre_process_one_function(extend_data, args=self.args)
                extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                    "text_in": question,
                                    "seq_out": seq_out})
                self.extended_data.append(extend_data)
            torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets):
        self.args = args
        self.raw_datasets = raw_datasets

        self.extended_data = []
        for raw_data in self.raw_datasets:
            extend_data = deepcopy(raw_data)
            extend_data.update(spider_add_serialized_schema(extend_data, args=self.args))

            question, seq_out = spider_pre_process_one_function(extend_data, args=self.args)
            extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                "text_in": question,
                                "seq_out": seq_out})
            self.extended_data.append(extend_data)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
