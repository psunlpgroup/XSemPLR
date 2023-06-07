import argparse,os,json
if __name__ == '__main__':
    ############################# Parameters ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mconala')
    parser.add_argument('--data_folder', type=str, default='../../dataset/')
    parser.add_argument('--language', type=str, default='mix')
    parser.add_argument('--mr', type=str, default='python')
    parser.add_argument('--crosslingual', action='store_true', default=True)
    args = parser.parse_args()
    ##############################################################################

    # Prepare the parameters
    dataset = args.dataset
    data_folder = f"{args.data_folder}{dataset}/"
    language = args.language
    mr = args.mr
    cross_lingual = args.crosslingual
    if dataset == 'mcwq':
        data_folder = data_folder + "mcd3/"

    # read predictions
    if cross_lingual:
        save_folder = f"results/crosslingual/{dataset}/"
    else:
        save_folder = f"results/monolingual/{dataset}/"

    if dataset == 'mconala' and cross_lingual:
        save_file = save_folder + f"mix_{mr}.txt"
    else:
        save_file = save_folder + f"{language}_{mr}.txt"
    preds = []
    with open(save_file) as file:
        for line in file:
            line = line.strip()
            preds.append(line)

    # read test dataset
    if dataset=='mconala':
        if cross_lingual:
            test_path = data_folder+"test.json"
        else:
            test_path = data_folder+"dev.json"
        test_data = json.load(open(test_path))

    else:
        test_path = data_folder + "test.json"
        if not os.path.exists(test_path):
            test_path = data_folder + "dev.json"
        test_data = json.load(open(test_path))

    # compute scores
    scores = []
    for pred, sample in zip(preds, test_data):
        if dataset == 'mconala':
            if language not in sample['question'].keys():
                continue

        ## Alignment 2: replace ' '
        pred = pred.replace(' ', '')
        if isinstance(sample['mr'][mr], str):
            gold = sample['mr'][mr].replace(' ', '')
        else:
            gold = sample['mr'][mr][language].replace(' ', '')
        # print("pred:",pred)
        # print("gold:", gold)
        # print("match:", pred==gold)

        scores.append(pred.lower()==gold.lower())

    print("Dataset:", dataset, "Language:", language, "MR:", mr)
    print("score is:", sum(scores)/len(scores))