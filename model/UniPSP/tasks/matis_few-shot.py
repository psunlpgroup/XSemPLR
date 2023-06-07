import json
import datasets

_DESCRIPTION = """"""
_CITATION = """"""


class MAtis(datasets.GeneratorBasedBuilder):

    def _info(self):
        features=datasets.Features(
            {
                "question": {
                    "en": datasets.Value("string"),
                    "es": datasets.Value("string"),
                    "de": datasets.Value("string"),
                    "fr": datasets.Value("string"),
                    "pt": datasets.Value("string"),
                    "zh": datasets.Value("string"),
                },
                "mr": {
                    "sql": datasets.Value("string"),
                }
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
        filepath = "/home/yfz5488/xsp/few-shot_dataset/matis/"

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
                }
