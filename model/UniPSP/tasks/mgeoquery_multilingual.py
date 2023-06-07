import json
import datasets

_DESCRIPTION = """"""
_CITATION = """"""

class MGeoquery(datasets.GeneratorBasedBuilder):

    def _info(self):
        features=datasets.Features(
            {
                "question": {
                    "multilingual": datasets.Value("string"),
                },
                "mr": {
                    "sql": datasets.Value("string"),
                    "prolog": datasets.Value("string"),
                    "lambda": datasets.Value("string"),
                    "funql": datasets.Value("string"),
                },
                "language": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        filepath = "/data/yfz5488/xsp/multilingual_dataset/mgeoquery/"

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepath+'train.json'}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": filepath+'dev.json'}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": filepath+'test.json'}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            reader = json.load(f)
            for example_idx, example in enumerate(reader):
                yield example_idx, {
                    "question": example["question"],
                    "mr": example["mr"],
                    "language": example["language"]
                }
