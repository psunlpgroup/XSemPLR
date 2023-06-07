import json
import datasets

_DESCRIPTION = """"""
_CITATION = """"""

class MSchema2qa(datasets.GeneratorBasedBuilder):

    def _info(self):
        features=datasets.Features(
            {
                "question": {
                    "multilingual": datasets.Value("string"),
                },
                "mr": {
                    "thingtalk": {
                        "multilingual": datasets.Value("string"),
                    },
                    "thingtalk_no_value": datasets.Value("string"),
                },
                "id": datasets.Value("string"),
                "domain": datasets.Value("string"),
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
        filepath = "/home/rmz5227/xsp/multilingual_dataset/mschema2qa/"

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepath+'train.json'}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": filepath+'test.json'}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            reader = json.load(f)
            for example_idx, example in enumerate(reader):
                yield example_idx, {
                    "question": example["question"],
                    "mr": example["mr"],
                    "id": example["id"],
                    "domain": example["domain"],
                    "language": example["language"]
                }
