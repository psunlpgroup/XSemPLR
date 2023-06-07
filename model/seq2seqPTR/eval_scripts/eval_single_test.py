gold_path = "/data/yfz5488/xsp/model/seq2seqPTR/data/mtop/xlm-roberta-large/en/test.tsv"
pred_path = "/data/yfz5488/xsp/model/seq2seqPTR/output/mtop_en_xlm-roberta-large_monofew_20230101/lr0.00005_batch24/test_output.txt"

import csv

def exact_match(gold, pred):
    gold = gold.strip()
    pred = pred.strip()
    return gold == pred

# read gold
gold_data = []
tsv = csv.reader(open(gold_path), delimiter='\t')
for line in tsv:
    gold_data.append(line)

# read prediction
pred_data = []
with open(pred_path) as file:
    cur = []
    count = 0
    for line in file:
        if len(line.strip()):
            cur.append(line)
        else:
            count += 1
            pred_data.append(
                {
                    "prediction": cur[3],
                    "gold": cur[5],
                    "score":exact_match(cur[3].replace("<s>","").replace("</s>",""),cur[5])
                })
            cur = []
    if len(cur):
        count += 1
        pred_data.append(
            {
                "prediction": cur[3],
                "gold": cur[5],
                "score": exact_match(cur[3].replace("<s>", "").replace("</s>", ""), cur[5])
            })
        cur = []

# Computer scores
score = sum(result['score'] for result in pred_data)/len(pred_data)
print(pred_path)
print(score)