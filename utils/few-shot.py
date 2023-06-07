# This folder aims to create a multilingual dataset that can be fed into the dataset
import os
import json
import copy
import random

from tqdm import tqdm
from random import choices
SRC = "../dataset"
DEST = "../few-shot_dataset"
random.seed(42)
SAMPLE_RATIO = 0.1

def main():
    for name in os.listdir(SRC):
        print("Processing dataset: {}".format(name))
        for split in ['train','test','dev']:
            if name == 'mcwq':
                if not os.path.exists(f"{DEST}/{name}/mcd3/"):
                    os.makedirs(f"{DEST}/{name}/mcd3/")
                file_path = os.path.join(f"{SRC}/{name}/mcd3/{split}.json")
                dest_path = os.path.join(f"{DEST}/{name}/mcd3/{split}.json")
            else:
                file_path = os.path.join(f"{SRC}/{name}/{split}.json")
                dest_path = os.path.join(f"{DEST}/{name}/{split}.json")
            print("Processing file: {}".format(file_path))
            if not os.path.exists(file_path):
                continue
            data = json.load(open(file_path))
            new_data = random.sample(data, int(len(data) * SAMPLE_RATIO)) if split != 'test' and split != 'dev' else data

            if not os.path.exists(f"{DEST}/{name}"):
                os.makedirs(f"{DEST}/{name}")

            with open(dest_path, 'w', encoding='utf-8') as file:
                print(len(new_data))
                json.dump(new_data, file, indent=2)

if __name__ == '__main__':
    main()
