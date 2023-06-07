gold_path = "/data/yfz5488/xsp/multilingual_dataset/mspider/dev.json"
pred_path = "/data/yfz5488/xsp/model/seq2seqPTR/output/mspider_multilingual_xlm-roberta-large_multilingual_20230120/lr0.00005_batch8/dev_output.txt"
import json
with open(gold_path, 'r') as file:
    data = json.load(file)

from collections import defaultdict

results = defaultdict(list)

def exact_match(gold, pred):
    gold = gold.strip()
    pred = pred.strip()
    return gold == pred

def find_data(pred_input, data):
    pred_input = pred_input.split("</s>")[0]
    pred_input = pred_input.replace("<s>","").replace("</s>","").replace(" ","").strip().lower()
    for i, sample in enumerate(data):
        gold_input = sample['question']['multilingual'].replace(" ","").strip().lower()
        if gold_input == pred_input:
            return i
    print(pred_input, "Not Found!!")
    return -1

with open(pred_path) as file:
    cur = []
    for line in file:
        if len(line.strip()):
            cur.append(line)
        else:
            base = 0
            if "mspider" in pred_path:
                base = 1
            count = find_data(cur[0 + base],data)
            cur_dict = results[data[count]["language"]]
            cur_dict.append(
                {
                    "prediction": cur[3 + base],
                    "gold": cur[5 + base],
                    "data": data[count],
                    "score":exact_match(cur[3 + base].replace("<s>","").replace("</s>",""),cur[5 + base])
                })
            cur = []
    if len(cur):
        count = find_data(cur[0], data)
        cur_dict = results[data[count]["language"]]
        cur_dict.append(
            {
                "prediction": cur[3],
                "gold": cur[5],
                "data": data[count],
                "score": exact_match(cur[3].replace("<s>", "").replace("</s>", ""), cur[5])
            })
        cur = []

    all = 0
    for lang, lang_res in results.items():
        single_lang_score = sum(result['score'] for result in lang_res)/len(lang_res)
        all += single_lang_score
        print(lang, single_lang_score)

    print("total:", all/len(results.items()))