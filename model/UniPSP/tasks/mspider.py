# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks"""


import json
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """
"""

_DESCRIPTION = """
"""

_HOMEPAGE = ""

_LICENSE = "CC BY-SA 4.0"


class MSpider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="spider",
            version=VERSION,
            description="",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        # print(args)
        # print(kwargs)
        # exit()

    def _info(self):
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": {
                    'en': datasets.Value("string"),
                    'zh': datasets.Value("string"),
                    'vi': datasets.Value("string"),
                },
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        filepath = "/home/yfz5488/xsp/dataset/mspider/"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": filepath + "train.json",
                    "db_path": "/home/rmz5227/xsp/dataset/spider/spider/database",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": filepath + "dev.json",
                    "db_path": "/home/rmz5227/xsp/dataset/spider/spider/database",
                },
            ),
        ]

    def _generate_examples(self, data_filepath, db_path):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            spider = json.load(f)
            for idx, sample in enumerate(spider):
                db_id = sample["db_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                    )
                schema = self.schema_cache[db_id]
                yield idx, {
                    "query": sample["mr"]["sql"],
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
