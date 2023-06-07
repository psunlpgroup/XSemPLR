# This folder aims to create a multilingual dataset that can be fed into the dataset
import os
import json
import copy
from tqdm import tqdm
SRC = "../dataset"
DEST = "../multilingual_dataset"


def main():
    for name in os.listdir(SRC):
        print("Processing dataset: {}".format(name))

        if name != "mspider":
            continue
        if name == 'mcwq':
            name = 'mcwq/mcd3'
        for split in ['train','test','dev']:
            if name == 'mcwq':
                file_path = os.path.join(f"{SRC}/{name}/mcd3/{split}.json")
                dest_path = os.path.join(f"{DEST}/{name}/mcd3/{split}.json")
            else:
                file_path = os.path.join(f"{SRC}/{name}/{split}.json")
                dest_path = os.path.join(f"{DEST}/{name}/{split}.json")
            print("Processing file: {}".format(file_path))
            if not os.path.exists(file_path):
                continue
            data = json.load(open(file_path))
            new_data = []

            if name == "mspider":
                count = 0
                database_id = ""

            for sample in tqdm(data):
                if name == "mspider":
                    if database_id != sample["db_id"]:
                        count = 0
                        database_id = sample["db_id"]
                    else:
                        count += 1
                for lang, question in sample['question'].items():
                    new_sample = copy.deepcopy(sample)
                    new_sample['question'] = {}
                    new_sample['question']['multilingual'] = question
                    new_sample['language'] = lang

                    if name == 'mspider':
                        assert new_sample['interaction_id'] == count

                    if name == 'mtop':
                        mr = new_sample['mr']["slot_intent"][lang]
                        new_sample['mr']["slot_intent"] = {}
                        new_sample['mr']["slot_intent"]['multilingual'] = mr

                    elif name == 'mschema2qa':
                        mr = new_sample['mr']["thingtalk"][lang]
                        new_sample['mr']["thingtalk"] = {}
                        new_sample['mr']["thingtalk"]['multilingual'] = mr

                    new_data.append(new_sample)
            if name == 'mcwq':
                if not os.path.exists(f"{DEST}/{name}/mcd3"):
                    os.makedirs(f"{DEST}/{name}/mcd3")
            else:
                if not os.path.exists(f"{DEST}/{name}"):
                    os.makedirs(f"{DEST}/{name}")

            with open(dest_path, 'w', encoding='utf-8') as file:
                json.dump(new_data, file, indent=2)


if __name__ == '__main__':
    main()
