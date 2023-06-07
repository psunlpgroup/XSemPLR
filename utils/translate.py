# This folder aims to create a multilingual dataset that can be fed into the dataset
import os
import json
import copy
import random

from tqdm import tqdm
from random import choices
SRC = "../dataset"
DEST = "../translated_dataset"
TRANS_SRC = "../translated_testset"
def main():
    for name in os.listdir(SRC):
        print("Processing dataset: {}".format(name))
        if name == 'mconala':
            continue

        for split in ['test','train','dev']:
            if name == 'mcwq':
                name = 'mcwq/mcd3'
            file_path = os.path.join(f"{SRC}/{name}/{split}.json")
            dest_path = os.path.join(f"{DEST}/{name}/{split}.json")

            print("Processing file: {}".format(file_path))
            if not os.path.exists(file_path):
                continue
            data = json.load(open(file_path))
            new_data = []
            if split == 'test' or \
                ( split == 'dev' and not os.path.exists(f"{TRANS_SRC}/{name}/test.json") ):
                trans_path = os.path.join(f"{TRANS_SRC}/{name}/{split}.json")
                trans_questions = json.load(open(trans_path))
                for sample, trans in zip(data, trans_questions):
                    new_sample = copy.deepcopy(sample)
                    for lang in sample['question'].keys():
                        if lang not in trans.keys():
                            print("error:", sample)
                        else:
                            new_sample['question'][lang] = trans[lang]
                    new_data.append(new_sample)
            else:
                new_data = data


            if not os.path.exists(f"{DEST}/{name}"):
                os.makedirs(f"{DEST}/{name}")

            with open(dest_path, 'w', encoding='utf-8') as file:
                json.dump(new_data, file)

if __name__ == '__main__':
    main()
