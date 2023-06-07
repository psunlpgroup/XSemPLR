import argparse, random, os, csv, logging, json, copy, math, operator, re, calendar, num2words, airportsdata, us

from transformers import BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, BartTokenizer, MBart50Tokenizer, \
    T5Tokenizer, MT5Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "../../dataset/"
def get_tokenizer(model):
    """
    Gets tokenizer from models, now supporting 7 groups of models: BERT, RoBERTa, XLM-R, BART, mBART50, T5, mT5
    """
    if model in ['bert-base-cased', 'bert-base-multilingual-cased']:
        tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
    elif model in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(model, add_prefix_space=True)
    elif model in ['xlm-roberta-base', 'xlm-roberta-large']:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model)
    elif model in ['facebook/bart-base', 'facebook/bart-large']:
        tokenizer = BartTokenizer.from_pretrained(model, add_prefix_space=True)
    elif model in ['facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
        tokenizer = MBart50Tokenizer.from_pretrained(model)
    elif model in ['t5-large', 't5-base', 't5-small']:
        tokenizer = T5Tokenizer.from_pretrained(model)
    elif model in ['google/mt5-large']:
        tokenizer = MT5Tokenizer.from_pretrained(model)

    if not tokenizer.cls_token:
        tokenizer.cls_token = tokenizer.pad_token
    if not tokenizer.sep_token:
        tokenizer.sep_token = tokenizer.eos_token

    return tokenizer


def get_utterances_mrs(file_path, language):
    utterances = []
    mrs = {"sql": [], "prolog": [], "lambda": [], "funql": []}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for piece in data:
            utterance = piece["question"][language]
            if language == 'zh':
                utterance = utterance.replace(' ', '')
            utterances.append(utterance)
            mrs["sql"].append(piece["mr"]["sql"])
            mrs["prolog"].append(piece["mr"]["prolog"])
            mrs["lambda"].append(piece["mr"]["lambda"])
            mrs["funql"].append(piece["mr"]["funql"])
    return utterances, mrs


def get_utterances_mr(file_path, language, mr):
    """
    Extracts the utterance-mr pairs for one language, one meaning representation in a specific dataset.
    """
    utterances = []
    mrs = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for piece in data:
            utterance = piece["question"][language]
            if language == 'zh':
                if mr in ['thingtalk', 'sparql']:
                    utterance = utterance
                else:
                    utterance = utterance.replace(' ', '')
            utterances.append(utterance)
            if mr in ['thingtalk', 'slot_intent']:
                mrs.append(piece["mr"][mr][language])
            else:
                mrs.append(piece["mr"][mr])
    return utterances, mrs


def read_mtop(model, output_dir):
    """
    Uses the tokenizer from model to preprocess the MTOP dataset, and saves it in output_dir.
    For each language, the output data folder contains 4 files:
        dev/test/train.tsv: each line indicates a sample, i.e. utterance \t tokenized utterance \t mr with pointer
        output_vocab.txt: each line indicates a possible token in output. @ptr{} are pointers to input, the others are
            operators in this meaning representation language. (train/test/dev share the same output vocab)
    """
    languages = ['en', 'de', 'fr', 'th', 'es', 'hi']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['multilingual']
    tokenizer = get_tokenizer(model)

    for language in languages:
        output_dir_language = os.path.join(output_dir, 'mtop', model.replace('/', '_'), language)
        if not os.path.exists(output_dir_language):
            os.makedirs(output_dir_language)

        output_vocab = set()
        len_list = []

        bad = 0
        with open(os.path.join(output_dir_language, 'train.tsv'), 'w') as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mtop/train.json", language, "slot_intent")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterance_tokenized = utterance.split()
                len_list.append(len(utterance_tokenized))

                query_pointers = []
                for token in mr.split():
                    if '[' in token or ']' in token:
                        output_vocab.add(token)
                        query_pointers.append(token)
                    else:
                        span_idx = utterance_tokenized.index(token)
                        assert (span_idx >= 1)
                        query_pointers.append('@ptr{}'.format(span_idx))

                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_language, 'dev.tsv'), 'w') as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mtop/dev.json", language, "slot_intent")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterance_tokenized = utterance.split()
                len_list.append(len(utterance_tokenized))

                query_pointers = []
                for token in mr.split():
                    if '[' in token or ']' in token:
                        output_vocab.add(token)
                        query_pointers.append(token)
                    else:
                        span_idx = utterance_tokenized.index(token)
                        assert (span_idx >= 1)
                        query_pointers.append('@ptr{}'.format(span_idx))

                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_language, 'test.tsv'), 'w') as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mtop/test.json", language, "slot_intent")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterance_tokenized = utterance.split()
                len_list.append(len(utterance_tokenized))

                query_pointers = []
                for token in mr.split():
                    if '[' in token or ']' in token:
                        output_vocab.add(token)
                        query_pointers.append(token)
                    else:
                        span_idx = utterance_tokenized.index(token)
                        assert (span_idx >= 1)
                        query_pointers.append('@ptr{}'.format(span_idx))

                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_language, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab)):
                f.write(token + '\n')
    return

def read_mconala(model, output_dir):
    """
    Uses the tokenizer from model to preprocess the MTOP dataset, and saves it in output_dir.
    For each language, the output data folder contains 4 files:
        dev/test/train.tsv: each line indicates a sample, i.e. utterance \t tokenized utterance \t mr with pointer
        output_vocab.txt: each line indicates a possible token in output. @ptr{} are pointers to input, the others are
            operators in this meaning representation language. (train/test/dev share the same output vocab)
    """
    # We only consider the monolingual setting
    languages = ['en']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['multilingual']
    tokenizer = get_tokenizer(model)

    for language in languages:
        output_dir_language = os.path.join(output_dir, 'mconala', model.replace('/', '_'), language)
        if not os.path.exists(output_dir_language):
            os.makedirs(output_dir_language)

        output_vocab = set()
        len_list = []

        bad = 0
        with open(os.path.join(output_dir_language, 'train.tsv'), 'w') as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mconala/train.json", language, "slot_intent")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterance_tokenized = utterance.split()
                len_list.append(len(utterance_tokenized))

                query_pointers = []
                for token in mr.split():
                    if '[' in token or ']' in token:
                        output_vocab.add(token)
                        query_pointers.append(token)
                    else:
                        span_idx = utterance_tokenized.index(token)
                        assert (span_idx >= 1)
                        query_pointers.append('@ptr{}'.format(span_idx))

                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_language, 'dev.tsv'), 'w') as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mtop/dev.json", language, "slot_intent")
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mtop/test.json", language, "slot_intent")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterance_tokenized = utterance.split()
                len_list.append(len(utterance_tokenized))

                query_pointers = []
                for token in mr.split():
                    if '[' in token or ']' in token:
                        output_vocab.add(token)
                        query_pointers.append(token)
                    else:
                        span_idx = utterance_tokenized.index(token)
                        assert (span_idx >= 1)
                        query_pointers.append('@ptr{}'.format(span_idx))

                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_language, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab)):
                f.write(token + '\n')
    return


def read_mtop_old(model, output_dir):
    # languages = ['en','de','fr','th','es','hi']
    languages = ['en']

    if model in ['bert-base-cased', 'bert-base-multilingual-cased']:
        tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
    elif model in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(model, add_prefix_space=True)
    elif model in ['xlm-roberta-base', 'xlm-roberta-large']:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model)
    elif model in ['facebook/bart-base', 'facebook/bart-large']:
        tokenizer = BartTokenizer.from_pretrained(model, add_prefix_space=True)
    elif model in ['facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
        tokenizer = MBart50Tokenizer.from_pretrained(model)
    elif model in ['t5-large']:
        tokenizer = T5Tokenizer.from_pretrained(model)
    elif model in ['google/mt5-large']:
        tokenizer = MT5Tokenizer.from_pretrained(model)

    output_vocab = set()
    output_ptr = set()
    len_list = []

    def process_line(line):
        uid, intent, slot, utterance, domain, locale, query, token = line.strip().split('\t')
        ### use tokens to build utterance
        token = json.loads(token)
        utterance = ' '.join(token['tokens'])
        utterance_tokenized = tokenizer.tokenize(utterance)
        len_list.append(len(utterance_tokenized))
        # print(utterance)
        # print(query)

        query_pointers = []
        value = []
        for token in query.split():
            if '[' in token or ']' in token:
                output_vocab.add(token)
                if value:
                    value_tokenized = tokenizer.tokenize(' '.join(value))
                    # print(value)
                    # print(value_tokenized)
                    # print(utterance_tokenized)
                    found = False
                    for i in range(len(utterance_tokenized)):
                        if utterance_tokenized[i:i + len(value_tokenized)] == value_tokenized:
                            pointers = ['@ptr{}'.format(idx) for idx in range(i, i + len(value_tokenized))]
                            # print(i, pointers)
                            query_pointers += pointers
                            output_ptr.update(pointers)
                            found = True
                            break
                    assert (found)
                    # exit()
                    # if not found:
                    #     print(' '.join(value), value_tokenized)
                    #     print(utterance, utterance_tokenized)
                    #     exit()
                    value = []
                query_pointers.append(token)
            else:
                if token in ['am', 'pm', 'AM', 'PM']:
                    if value and value[-1] + token in utterance:
                        value[-1] = value[-1] + token
                    else:
                        value.append(token)
                else:
                    value.append(token)
                # token_tokenized = tokenizer.tokenize(token)
                # for i in range(len(utterance_tokenized)):
                #     if utterance_tokenized[i:i+len(token_tokenized)] == token_tokenized:
                #         pointers = ['@ptr{}'.format(idx) for idx in range(i,i+len(token_tokenized))]
                #         query_pointers += pointers
                #         output_ptr.update(pointers)
                #         break

        utterance_tokenized = ' '.join(utterance_tokenized)
        query_pointers = ' '.join(query_pointers)
        return utterance, utterance_tokenized, query_pointers, query

    for language in languages:
        output_dir = os.path.join(output_dir, 'mtop', model.replace('/', '_'), language)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        f_train = open(os.path.join(output_dir, 'train.tsv'), 'w')
        f_dev = open(os.path.join(output_dir, 'dev.tsv'), 'w')
        f_test = open(os.path.join(output_dir, 'test.tsv'), 'w')

        train_language = '../../dataset/mtop/{}/train.txt'.format(language)
        cnt = 0
        with open(train_language) as f:
            for line in f.readlines():
                cnt += 1
                # print(cnt)
                utterance, utterance_tokenized, query_pointers, query = process_line(line)
                # print(utterance)
                # print(utterance_tokenized)
                # print(query_pointers)
                # print(query)
                # print()
                f_train.write('\t'.join([utterance, utterance_tokenized, query_pointers]) + '\n')
                # if cnt == 13:
                #     exit()
        f_train.close()

        dev_language = '../../dataset/mtop/{}/eval.txt'.format(language)
        with open(dev_language) as f:
            for line in f.readlines():
                utterance, utterance_tokenized, query_pointers, query = process_line(line)
                f_dev.write('\t'.join([utterance, utterance_tokenized, query_pointers]) + '\n')
        f_dev.close()

        test_language = '../../dataset/mtop/{}/test.txt'.format(language)
        with open(test_language) as f:
            for line in f.readlines():
                utterance, utterance_tokenized, query_pointers, query = process_line(line)
                f_test.write('\t'.join([utterance, utterance_tokenized, query_pointers]) + '\n')
        f_test.close()

    # print('len_list', max(len_list), 'output_ptr', len(output_ptr))
    with open(os.path.join(output_dir, 'output_vocab.txt'), 'w') as f:
        # for token in sorted(list(output_ptr), key=lambda x: int(x[4:])):
        for idx in range(max(len_list)):
            f.write('@ptr{}'.format(idx) + '\n')
        for token in sorted(list(output_vocab)):
            f.write(token + '\n')
    return


def read_mgeoquery(model, output_dir):
    """
    Uses the tokenizer from model to preprocess the MGeoQuery dataset, and saves it in output_dir.
    For each source language e.g. 'en', each target language e.g. SQL, the output data folder contains 4 files:
        dev/test/train.tsv: each line indicates a sample, i.e. utterance \t tokenized utterance \t mr with pointer
        output_vocab.txt: each line indicates a possible token in output. @ptr{} are pointers to input, the others are
            operators in this meaning representation language. (train/test/dev share the same output vocab)
    """
    languages = ['en', 'de', 'el', 'fa', 'id', 'sv', 'th', 'zh']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['multilingual']
    # languages = ['en']
    tokenizer = get_tokenizer(model)

    for language in languages:
        output_dir_language = os.path.join(output_dir, 'mgeoquery', model.replace('/', '_'), language)
        output_dir_prolog = os.path.join(output_dir_language, "prolog")
        output_dir_sql = os.path.join(output_dir_language, "sql")
        output_dir_funql = os.path.join(output_dir_language, "funql")
        output_dir_lambda = os.path.join(output_dir_language, "lambda")
        if not os.path.exists(output_dir_prolog):
            os.makedirs(output_dir_prolog)
        if not os.path.exists(output_dir_sql):
            os.makedirs(output_dir_sql)
        if not os.path.exists(output_dir_funql):
            os.makedirs(output_dir_funql)
        if not os.path.exists(output_dir_lambda):
            os.makedirs(output_dir_lambda)

        PREDS_PROLOG = [
            'cityid', 'countryid', 'placeid', 'riverid', 'stateid',
            'capital', 'city', 'lake', 'major', 'mountain', 'place', 'river',
            'state', 'area', 'const', 'density', 'elevation', 'high_point',
            'higher', 'loc', 'longer', 'low_point', 'lower', 'len', 'next_to',
            'population', 'size', 'traverse',
            'answer', 'largest', 'smallest', 'highest', 'lowest', 'longest',
            'shortest', 'count', 'most', 'fewest', 'sum',
            'not']
        PREDS_LAMBDA = ['$0', '$1', '$2', '$3', '$4', '(', ')', '0', ':<<e,t>,<<e,i>,e>>', ':<<e,t>,<<e,i>,i>>',
                        ':<<e,t>,e>', ':<<e,t>,i>', ':<<e,t>,t>', ':<c,t>', ':<e,<e,t>>', ':<e,<n,t>>', ':<i,<i,t>>',
                        ':<l,t>', ':<lo,<i,t>>', ':<lo,<lo,t>>', ':<lo,i>', ':<lo,t>', ':<m,t>', ':<p,t>', ':<r,i>',
                        ':<r,t>', ':<s,<c,t>>', ':<s,c>', ':<s,t>', ':<t*,t>', ':<t,t>', ':c', ':co', ':e', ':i', ':lo',
                        ':m', ':n', ':p', ':r', ':s', '<', '=', '>', 'and', 'area', 'argmax', 'argmin', 'capital',
                        'capital2', 'city', 'count', 'density', 'elevation', 'equals', 'exists', 'high_point', 'in',
                        'lake', 'lambda', 'len', 'loc', 'major', 'mountain', 'named', 'next_to', 'not', 'or', 'place',
                        'population', 'river', 'size', 'state', 'sum', 'the', 'town']
        CONJ = ["(", ")", ",", ".", "'", "_"]
        output_vocab_prolog = set()
        len_list_prolog = []
        output_vocab_sql = set()
        len_list_sql = []
        output_vocab_funql = set()
        len_list_funql = []
        output_vocab_lambda = set()
        len_list_lambda = []

        utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mgeoquery/dev.json", language)

        with open(os.path.join(output_dir_prolog, "dev.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["prolog"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_prolog.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace(
                    ".", " . ").replace("   ", " ").replace("  ", " ")

                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                # elif re.search(r"\bsea level\b", utterance):
                #     mr_padding = mr_padding.replace("0", "sea level").split()
                # else:
                #     mr_padding = mr_padding.split()
                mr_padding = mr_padding.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token not in PREDS_PROLOG and mr_token not in CONJ and not (
                            mr_token >= "A" and mr_token <= "Z"):
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_prolog.add(mr_token)
                            query_pointers.append(mr_token)
                            # All unknown tokens (not input, not keywords), are operators
                    else:
                        output_vocab_prolog.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_sql, "dev.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["sql"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_sql.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace(
                    '"', ' " ').replace("   ", " ").replace("  ", " ")
                # if re.search(r"\bga\b", utterance):
                #     mr_padding = mr_padding.replace("georgia", "ga").split()
                # else:
                #     mr_padding = mr_padding.split()
                mr_padding = mr_padding.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token.islower():
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_sql.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_sql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_funql, "dev.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["funql"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_funql.append(len(utterances_tokenized))

                pattern = re.compile("'(.*?)'")

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace(
                    '"', ' " ').replace("   ", " ").replace("  ", " ")
                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                #     res = pattern.findall(mr.replace("usa", "united states"))
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                #     res = pattern.findall(mr.replace("usa", "us"))
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                #     res = pattern.findall(mr.replace("usa", "america"))
                # elif re.search(r"\bsea level\b", utterance):
                #     mr_padding = mr_padding.replace("0", "sea level").split()
                #     res = pattern.findall(mr.replace("0", "sea level"))
                # else:
                #     mr_padding = mr_padding.split()
                #     res = pattern.findall(mr)
                mr_padding = mr_padding.split()
                res = pattern.findall(mr)
                res_final = []
                for token in res:
                    res_final.extend(token.split())
                # res_final.extend(["sea", "level"])

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res_final:
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_funql.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_funql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_lambda, "dev.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["lambda"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(":", " :").replace("   ", " ").replace(
                    "  ", " ")
                # if re.search(r"\balbany_ny\b", mr_padding):
                #     mr_padding = mr_padding.replace("albany_ny", "albany")
                # elif re.search(r"\bdallas_tx\b", mr_padding):
                #     mr_padding = mr_padding.replace("dallas_tx", "dallas")
                # elif re.search(r"\bsan_diego_ca\b", mr_padding):
                #     mr_padding = mr_padding.replace("san_diego_ca", "san diego")
                # elif re.search(r"\bchicago_il\b", mr_padding):
                #     mr_padding = mr_padding.replace("chicago_il", "chicago")

                # pattern = re.compile(r'[a-zA-Z]+_[a-zA-Z]+')
                # res = pattern.findall(mr_padding)
                # for token in res:
                #     if token != "next_to":
                #         mr_padding = mr_padding.replace(token, " ".join(token.split("_")))
                #         print(raw_utterance)
                #         print(mr)
                #         print(token)
                #         print(res)
                #         exit()

                # if re.search(r"'(.*?)'", mr_padding):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                # elif re.search(r"\bsea level\b", utterance):
                #     mr_padding = mr_padding.replace("0", " sea level").split()
                # else:
                #     mr_padding = mr_padding.split()
                mr_padding = mr_padding.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in PREDS_LAMBDA:
                        output_vocab_lambda.add(mr_token)
                        query_pointers.append(mr_token)
                    else:
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_lambda.add(mr_token)
                            query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mgeoquery/test.json", language)

        with open(os.path.join(output_dir_prolog, "test.tsv"), "w") as f:
            utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mgeoquery/test.json", language)
            for utterance, mr in zip(utterances, mrs["prolog"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_prolog.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace(
                    ".", " . ").replace("   ", " ").replace("  ", " ")

                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                # elif re.search(r"\bmn\b", mr_padding):
                #     mr_padding = mr_padding.replace("mn", "minnesota").split()
                # elif re.search(r"\bpa\b", mr_padding):
                #     mr_padding = mr_padding.replace("pa", "pennsylvania").split()
                # elif re.search(r"\baz\b", mr_padding):
                #     mr_padding = mr_padding.replace("az", "arizona").split()
                # else:
                #     mr_padding = mr_padding.split()
                mr_padding = mr_padding.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token not in PREDS_PROLOG and mr_token not in CONJ and not (
                            mr_token >= "A" and mr_token <= "Z"):
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_prolog.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_prolog.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_sql, "test.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["sql"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_sql.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace(
                    '"', ' " ').replace("   ", " ").replace("  ", " ").split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token.islower():
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_sql.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_sql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_funql, "test.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["funql"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_funql.append(len(utterances_tokenized))

                pattern = re.compile("'(.*?)'")
                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace(
                    '"', ' " ').replace("   ", " ").replace("  ", " ")
                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                #     res = pattern.findall(mr.replace("usa", "united states"))
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                #     res = pattern.findall(mr.replace("usa", "us"))
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                #     res = pattern.findall(mr.replace("usa", "america"))
                # elif re.search(r"\bmn\b", mr_padding):
                #     mr_padding = mr_padding.replace("mn", "minnesota").split()
                #     res = pattern.findall(mr.replace("mn", "minnesota"))
                # elif re.search(r"\bpa\b", mr_padding):
                #     mr_padding = mr_padding.replace("pa", "pennsylvania").split()
                #     res = pattern.findall(mr.replace("pa", "pennsylvania"))
                # elif re.search(r"\baz\b", mr_padding):
                #     mr_padding = mr_padding.replace("az", "arizona").split()
                #     res = pattern.findall(mr.replace("az", "arizona"))
                # else:
                #     mr_padding = mr_padding.split()
                #     res = pattern.findall(mr)
                mr_padding = mr_padding.split()
                res = pattern.findall(mr)
                res_final = []
                for token in res:
                    res_final.extend(token.split())

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res_final:
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_funql.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_funql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_lambda, "test.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["lambda"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(":", " :").replace("   ", " ").replace("  ", " ")
                # if re.search(r"\bsalt_lake_city\b", mr_padding):
                #     mr_padding = mr_padding.replace("salt_lake_city", "salt lake city")

                # pattern = re.compile(r'\b[a-zA-Z\_]+_[a-z][a-z]\b')
                # res = pattern.findall(mr_padding)
                # for token in res:
                #     if token != "next_to":
                #         mr_padding = mr_padding.replace(token, token.split("_")[0])

                # pattern = re.compile(r'\b[a-z\_]+_[a-z]+\b')
                # res = pattern.findall(mr_padding)
                # for token in res:
                #     if token != "next_to":
                #         mr_padding = mr_padding.replace(token, " ".join(token.split("_")))


                # pattern = re.compile(r'[a-zA-Z]+_[a-zA-Z]+')
                # res = pattern.findall(mr_padding)
                # for token in res:
                #     if token != "next_to":
                #         mr_padding = mr_padding.replace(token, " ".join(token.split("_")))

                # if re.search(r"'(.*?)'", mr_padding):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                # elif re.search(r"\bamerican\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "american").split()
                # elif re.search(r"\bsea level\b", utterance):
                #     mr_padding = mr_padding.replace("0", " sea level").split()
                # else:
                #     mr_padding = mr_padding.split()
                mr_padding = mr_padding.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in PREDS_LAMBDA:
                        output_vocab_lambda.add(mr_token)
                        query_pointers.append(mr_token)
                    else:
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_lambda.add(mr_token)
                            query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mgeoquery/train.json", language)

        with open(os.path.join(output_dir_prolog, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mgeoquery/train.json", language)
            for utterance, mr in zip(utterances, mrs["prolog"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_prolog.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace(
                    ".", " . ").replace("   ", " ").replace("  ", " ")

                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                # elif re.search(r"\bmn\b", mr_padding):
                #     mr_padding = mr_padding.replace("mn", "minnesota").split()
                # elif re.search(r"\bpa\b", mr_padding):
                #     mr_padding = mr_padding.replace("pa", "pennsylvania").split()
                # elif re.search(r"\bma\b", mr_padding):
                #     mr_padding = mr_padding.replace("ma", "massachusetts").split()
                # elif re.search(r"\bmo\b", mr_padding):
                #     mr_padding = mr_padding.replace("mo", "missouri").split()
                # elif re.search(r"\bsd\b", mr_padding):
                #     mr_padding = mr_padding.replace("sd", "south dakota").split()
                # elif re.search(r"\btx\b", mr_padding):
                #     mr_padding = mr_padding.replace("tx", "texas").split()
                # elif re.search(r"\bme\b", mr_padding):
                #     mr_padding = mr_padding.replace("me", "maine").split()
                # elif re.search(r"\bwa\b", mr_padding):
                #     mr_padding = mr_padding.replace("wa", "washington").split()
                # elif re.search(r"\busa\b", mr_padding) and re.search(r"\bcountry\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "country").split()
                # else:
                #     mr_padding = mr_padding.split()
                mr_padding = mr_padding.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token not in PREDS_PROLOG and mr_token not in CONJ and not (mr_token >= "A" and mr_token <= "Z"):
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_prolog.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_prolog.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_lambda, "train.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["lambda"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(":", " :").replace("   ", " ").replace(
                    "  ", " ")
                # if re.search(r"\bsalt_lake_city\b", mr_padding):
                #     mr_padding = mr_padding.replace("salt_lake_city", "salt lake city")

                # pattern = re.compile(r'\b[a-zA-Z\_]+_[a-z][a-z]\b')
                # res = pattern.findall(mr_padding)
                # for token in res:
                #     if token != "next_to":
                #         mr_padding = mr_padding.replace(token, token.split("_")[0])

                # pattern = re.compile(r'\b[a-z\_]+_[a-z]+\b')
                # res = pattern.findall(mr_padding)
                # for token in res:
                #     if token != "next_to" and token not in PREDS_LAMBDA:
                #         mr_padding = mr_padding.replace(token, " ".join(token.split("_")))

                # if re.search(r"'(.*?)'", mr_padding):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                # elif re.search(r"\bamerican\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "american").split()
                # elif re.search(r"\bcountry\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "country").split()
                # elif re.search(r"\bstate\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "state").split()
                # elif re.search(r"\bstates\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "states").split()
                # elif re.search(r"\bsea level\b", utterance):
                #     mr_padding = mr_padding.replace("0", " sea level").split()
                # else:
                #     mr_padding = mr_padding.split()
                mr_padding = mr_padding.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in PREDS_LAMBDA:
                        output_vocab_lambda.add(mr_token)
                        query_pointers.append(mr_token)
                    else:
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_lambda.add(mr_token)
                            query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_sql, "train.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["sql"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_sql.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace(
                    '"', ' " ').replace("   ", " ").replace("  ", " ").split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token.islower():
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_sql.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_sql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_funql, "train.tsv"), "w") as f:
            for utterance, mr in zip(utterances, mrs["funql"]):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_funql.append(len(utterances_tokenized))

                pattern = re.compile("'(.*?)'")

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(",", " , ").replace("'", " ' ").replace('"', ' " ').replace("   ", " ").replace("  ", " ")
                # if re.search(r"\bunited states\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "united states").split()
                #     res = pattern.findall(mr.replace("usa", "united states"))
                # elif re.search(r"\bus\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "us").split()
                #     res = pattern.findall(mr.replace("usa", "us"))
                # elif re.search(r"\bamerica\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "america").split()
                #     res = pattern.findall(mr.replace("usa", "america"))
                # elif re.search(r"\bmn\b", mr_padding):
                #     mr_padding = mr_padding.replace("mn", "minnesota").split()
                #     res = pattern.findall(mr.replace("mn", "minnesota"))
                # elif re.search(r"\bpa\b", mr_padding):
                #     mr_padding = mr_padding.replace("pa", "pennsylvania").split()
                #     res = pattern.findall(mr.replace("pa", "pennsylvania"))
                # elif re.search(r"\bma\b", mr_padding):
                #     mr_padding = mr_padding.replace("ma", "massachusetts").split()
                #     res = pattern.findall(mr.replace("ma", "massachusetts"))
                # elif re.search(r"\bmo\b", mr_padding):
                #     mr_padding = mr_padding.replace("mo", "missouri").split()
                #     res = pattern.findall(mr.replace("mo", "missouri"))
                # elif re.search(r"\bsd\b", mr_padding):
                #     mr_padding = mr_padding.replace("sd", "south dakota").split()
                #     res = pattern.findall(mr.replace("sd", "south dakota"))
                # elif re.search(r"\btx\b", mr_padding):
                #     mr_padding = mr_padding.replace("tx", "texas").split()
                #     res = pattern.findall(mr.replace("tx", "texas"))
                # elif re.search(r"\bme\b", mr_padding):
                #     mr_padding = mr_padding.replace("me", "maine").split()
                #     res = pattern.findall(mr.replace("me", "maine"))
                # elif re.search(r"\bwa\b", mr_padding):
                #     mr_padding = mr_padding.replace("wa", "washington").split()
                #     res = pattern.findall(mr.replace("wa", "washington"))
                # elif re.search(r"\busa\b", mr_padding) and re.search(r"\bcountry\b", utterance):
                #     mr_padding = mr_padding.replace("usa", "country").split()
                #     res = pattern.findall(mr.replace("usa", "country"))
                # else:
                #     mr_padding = mr_padding.split()
                #     res = pattern.findall(mr)
                mr_padding = mr_padding.split()
                res = pattern.findall(mr)
                res_final = []
                for token in res:
                    res_final.extend(token.split())

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res_final:
                        # tokens = tokenizer.tokenize(mr_token)
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        if mr_token in utterances_tokenized:
                            query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                        else:
                            output_vocab_funql.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_funql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')


        with open(os.path.join(output_dir_prolog, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_prolog)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_prolog)):
                f.write(token + '\n')

        with open(os.path.join(output_dir_sql, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_sql)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_sql)):
                f.write(token + '\n')

        with open(os.path.join(output_dir_funql, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_funql)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_funql)):
                f.write(token + '\n')

        with open(os.path.join(output_dir_lambda, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_lambda)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_lambda)):
                f.write(token + '\n')


def read_mspider_old(model, output_dir):
    languages = ['en']

    if model in ['bert-base-cased', 'bert-base-multilingual-cased']:
        tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
    elif model in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(model, add_prefix_space=True)
    elif model in ['xlm-roberta-base', 'xlm-roberta-large']:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model)
    elif model in ['facebook/bart-base', 'facebook/bart-large']:
        tokenizer = BartTokenizer.from_pretrained(model, add_prefix_space=True)
    elif model in ['facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
        tokenizer = MBart50Tokenizer.from_pretrained(model)
    elif model in ['t5-large']:
        tokenizer = T5Tokenizer.from_pretrained(model)
    elif model in ['google/mt5-large']:
        tokenizer = MT5Tokenizer.from_pretrained(model)

    sql_keywords = ['.', 't1', 't2', '=', 'select', 'as', 'join', 'on', ')', '(', 'where', 't3', 'by', ',', 'group',
                    'distinct', 't4', 'and', 'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<', 'sum', 't5',
                    'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!', 'union', 'between', 't6', '-', 't7',
                    '+', '/']
    sql_keywords += ['count', 'from', 'value', 'order']
    sql_keywords += ['group_by', 'order_by', 'limit_value', '!=']   # TODO: check here
    sql_keywords += ['*']
    len_list = []
    cnt_bad = []

    def process_data(data, table):
        # Read Schema
        schema = {}
        for table_idx, table_name in enumerate(table['table_names_original']):
            schema[table_idx] = [table_name.lower()]
        for table_idx, column_name in table['column_names_original']:
            if table_idx != -1:
                schema[table_idx].append(column_name.lower())

        schema_list = []
        for table_idx in range(len(table['table_names'])):
            schema_list.append(tokenizer.sep_token)
            schema_list += schema[table_idx]

        # Read utterance
        utterance = data['question']['en']

        # concat with tokenizer.sep_token
        utterance_schema =  utterance + ' ' + ' '.join(schema_list)
        utterance_tokenized = tokenizer.tokenize(utterance_schema)
        len_list.append(len(utterance_tokenized))

        name2span = {}
        utterance_tokenized_2 = tokenizer.tokenize(utterance)
        for table_idx in range(len(table['table_names'])):
            utterance_tokenized_2 += tokenizer.tokenize(tokenizer.sep_token)
            for name in schema[table_idx]:
                start_idx = len(utterance_tokenized_2)
                utterance_tokenized_2 += tokenizer.tokenize(name)
                end_idx = len(utterance_tokenized_2)
                name2span[name] = list(range(start_idx, end_idx))

        # print(schema)
        # print(utterance)
        # print(utterance_schema)
        # print(utterance_tokenized)
        # print(utterance_tokenized_2)
        # assert(utterance_tokenized == utterance_tokenized_2)
        # return '', '', '', ''

        query = data['mr']['sql_toks_no_value']
        # print(query)
        if utterance == 'List roles that have more than one employee. List the role description and number of employees.':
            query = ['select', 't1', '.', 'role_description', ',', 'count', '(', 't2', '.', 'employee_id', ')', 'from',
                     'roles', 'as', 't1', 'join', 'employees', 'as', 't2', 'on', 't2', '.', 'role_code', '=', 't1', '.',
                     'role_code', 'group', 'by', 't2', '.', 'role_code', 'having', 'count', '(', 't2', '.',
                     'employee_id', ')', '>', 'value']
            # print(' '.join(query))
        elif utterance == 'What is the document status description of the document with id 1?':
            query = ['select', 't1', '.', 'document_status_description', 'from', 'ref_document_status', 'as', 't1',
                     'join', 'documents', 'as', 't2', 'on', 't2', '.', 'document_status_code', '=', 't1', '.',
                     'document_status_code', 'where', 't2', '.', 'document_id', '=', 'value']
        elif utterance == 'What is the name of the shipping agent of the document with id 2?':
            query = ['select', 't1', '.', 'shipping_agent_name', 'from', 'ref_shipping_agents', 'as', 't1', 'join',
                     'documents', 'as', 't2', 'on', 't2', '.', 'shipping_agent_code', '=', 't1', '.',
                     'shipping_agent_code', 'where', 't2', '.', 'document_id', '=', 'value']
        elif utterance == 'How many documents were shipped by USPS?':
            query = ['select', 'count', '(', '*', ')', 'from', 'ref_shipping_agents', 'as', 't1', 'join', 'documents',
                     'as', 't2', 'on', 't2', '.', 'shipping_agent_code', '=', 't1', '.', 'shipping_agent_code', 'where',
                     't1', '.', 'shipping_agent_name', '=', 'value']
        query_pointers = []
        value = []
        for token in query:
            if token in sql_keywords:
                query_pointers.append(token)
            else:
                # Then, token is a table_name or column_name
                if token not in name2span:
                    cnt_bad.append(1)
                    query_pointers = []
                    print(utterance)
                    print(query)
                    print()
                    break
                for span_idx in name2span[token]:
                    query_pointers.append('@ptr{}'.format(span_idx))

        utterance_tokenized = ' '.join(utterance_tokenized)
        query_pointers = ' '.join(query_pointers)

        # print(query_pointers)
        # print()
        return utterance, utterance_tokenized, query_pointers, query

    for language in languages:
        output_dir = os.path.join(output_dir, 'mspider', model.replace('/', '_'), language)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        table_f = '../../dataset/mspider/tables.json'
        with open(table_f) as f:
            tables = json.load(f)
        db2table = {}
        for table in tables:
            db2table[table['db_id']] = table

        train_f = '../../dataset/mspider/train.json'
        with open(train_f) as f:
            train_data = json.load(f)
        with open(os.path.join(output_dir, 'train.tsv'), 'w') as f:
            for data in train_data:
                table = db2table[data['db_id']]
                utterance, utterance_tokenized, query_pointers, query = process_data(data, table)
                if query_pointers:
                    f.write('\t'.join([utterance, utterance_tokenized, query_pointers]) + '\n')

        print(sum(cnt_bad))

        dev_f = '../../dataset/mspider/dev.json'
        with open(dev_f) as f:
            dev_data = json.load(f)
        with open(os.path.join(output_dir, 'dev.tsv'), 'w') as f:
            for data in dev_data:
                table = db2table[data['db_id']]
                utterance, utterance_tokenized, query_pointers, query = process_data(data, table)
                f.write('\t'.join([utterance, utterance_tokenized, query_pointers]) + '\n')

        print(sum(cnt_bad))

    with open(os.path.join(output_dir, 'output_vocab.txt'), 'w') as f:
        for idx in range(max(len_list)):
            f.write('@ptr{}'.format(idx) + '\n')
        for token in sorted(sql_keywords):
            f.write(token + '\n')
    return


def read_database_schema(database_schema_filename):
    with open(database_schema_filename, "r") as f:
        database_schema = json.load(f)

    database_schema_dict = {}
    column_names_surface_form = {}
    column_names_embedder_input = {}
    for table_schema in database_schema:
        db_id = table_schema['db_id']
        database_schema_dict[db_id] = table_schema
        column_names_surface_form[db_id] = []
        column_names_embedder_input[db_id] = []

        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']

        prev_table_id = -1
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                if table_id != prev_table_id:
                    # add table_name.*
                    prev_table_id = table_id
                    table_name = table_names_original[table_id]
                    column_names_surface_form[db_id].append('{}.*'.format(table_name.lower()))
                table_name = table_names_original[table_id]
                column_name = column_name.replace(' ', '_')  # TODO: This will change the output format as well?
                assert (' ' not in '{}.{}'.format(table_name, column_name))
                column_name_surface_form = '{}.{}'.format(table_name, column_name)
            else:
                column_name_surface_form = column_name
            column_names_surface_form[db_id].append(column_name_surface_form.lower())

        prev_table_id = -1
        for i, (table_id, column_name) in enumerate(column_names):
            if table_id >= 0:
                if table_id != prev_table_id:
                    prev_table_id = table_id
                    table_name = table_names[table_id]
                    column_name_embedder_input = table_name + ' . *'
                    column_names_embedder_input[db_id].append(column_name_embedder_input)
                table_name = table_names[table_id]
                column_name_embedder_input = table_name + ' . ' + column_name
            else:
                column_name_embedder_input = column_name
            column_names_embedder_input[db_id].append(column_name_embedder_input)

    return database_schema_dict, column_names_surface_form, column_names_embedder_input


def read_mspider(model, output_dir):
    languages = ['en', 'zh', 'vi']
    if MULTILINGUAL:
        languages = ['multilingual']

    tokenizer = get_tokenizer(model)

    # sql_keywords = ['select', 'value', ')', '(', 'where', '=', ',', 'count', 'group_by', 'order_by', 'limit_value', 'desc', '>', 'distinct', 'avg', 'and', 'having', '<', 'in', 'max', 'sum', 'asc', 'like', 'not', 'or', 'min', 'intersect', 'except', '!=', 'union', 'between', '-', '+']
    sql_keywords = ['=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group_by', 'order_by', 'distinct', 'and',
                    'limit_value', 'desc', '>', 'avg', 'having', 'max', 'in', '<', 'sum', 'intersect', 'not', 'min',
                    'except', 'or', 'asc', 'like', '!=', 'union', 'between', '-', '+']

    database_schema, column_names_surface_form, column_names_embedder_input = read_database_schema(
        'data/spider_data_removefrom/tables.json')

    # def process_data(data):
    #     database_id = data['database_id']
    #     # Read Schema
    #     schema_list = []
    #     prev_table = None
    #     for column_name in column_names_surface_form[database_id]:
    #         current_table = column_name.split('.')[0]
    #         if current_table != prev_table:
    #             schema_list.append(tokenizer.sep_token)
    #             prev_table = current_table
    #         schema_list.append(column_name)

    #     # Read utterance
    #     utterance = data['final']['utterance'].lower()

    #     # Form the utterance_schema input
    #     utterance_schema = tokenizer.cls_token + ' ' + utterance + ' ' + ' '.join(schema_list) + ' ' + tokenizer.sep_token
    #     utterance_tokenized = utterance_schema.split()
    #     len_list.append(len(utterance_tokenized))

    #     query = data['final']['sql'].split()
    #     query_pointers = []
    #     for token in query:
    #         if token in sql_keywords:
    #             query_pointers.append(token)
    #             sql_keywords_appear.add(token)
    #         else:
    #             # Then, token is a table_name or column_name
    #             span_idx = utterance_tokenized.index(token)
    #             query_pointers.append('@ptr{}'.format(span_idx))

    #     query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token

    #     # print(query)
    #     # print(query_pointers)
    #     # for i, token in enumerate(utterance_tokenized):
    #     #     print(i, token)
    #     # print()

    #     return utterance, utterance_schema, query_pointers, query, database_id

    removefrom_sql = {}  # No longer one-one mapping after applying multilingual setting
    with open("data/spider_data_removefrom/train.json", "r", encoding="utf-8") as f_in:
        examples = json.load(f_in)
    for data in examples:
        database_id = data['database_id']
        interaction_id = data['interaction_id']
        if database_id not in removefrom_sql:
            removefrom_sql[database_id] = {}
        removefrom_sql[database_id][interaction_id] = data['final']['sql']
    with open("data/spider_data_removefrom/dev.json", "r", encoding="utf-8") as f_in:
        examples = json.load(f_in)
    for data in examples:
        database_id = data['database_id']
        interaction_id = data['interaction_id']
        if database_id not in removefrom_sql:
            removefrom_sql[database_id] = {}
        removefrom_sql[database_id][interaction_id] = data['final']['sql']

    for language in languages:
        output_dir_language = os.path.join(output_dir, 'mspider', model.replace('/', '_'), language)
        if not os.path.exists(output_dir_language):
            os.makedirs(output_dir_language)

        len_list = []
        sql_keywords_appear = set()

        count = {}

        with open(os.path.join(output_dir_language, 'train.tsv'), 'w') as f:
            with open(f"{DATASET_PATH}mspider/train.json", "r", encoding="utf-8") as f_in:
                examples = json.load(f_in)
            for data in examples:
                database_id = data['db_id']
                # if database_id not in count:
                #     count[database_id] = -1
                # count[database_id] += 1
                # interaction_id = count[database_id]
                # if MULTILINGUAL:
                interaction_id = data['interaction_id'] # We maintain the interaction_id in multilingual examples

                if removefrom_sql[database_id][interaction_id] == 'SKIP':
                    print('skip', database_id)
                    continue
                # Read Schema
                schema_list = []
                prev_table = None
                for column_name in column_names_surface_form[database_id]:
                    current_table = column_name.split('.')[0]
                    if current_table != prev_table:
                        schema_list.append(tokenizer.sep_token)
                        prev_table = current_table
                    schema_list.append(column_name)

                # Read utterance
                utterance = data['question'][language].lower()
                raw_utterance = utterance

                # Form the utterance_schema input
                utterance_schema = tokenizer.cls_token + ' ' + utterance + ' ' + ' '.join(
                    schema_list) + ' ' + tokenizer.sep_token
                utterance_tokenized = utterance_schema.split()
                len_list.append(len(utterance_tokenized))

                mr = removefrom_sql[database_id][interaction_id].split()

                query_pointers = []
                for token in mr:
                    if token in sql_keywords:
                        query_pointers.append(token)
                        sql_keywords_appear.add(token)
                    else:
                        assert (token in schema_list)
                        span_idx = schema_list.index(token) + len((tokenizer.cls_token + ' ' + utterance).split())
                        assert (token == utterance_schema.split()[span_idx])
                        query_pointers.append('@ptr{}'.format(span_idx))
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance_schema, query_pointers, database_id]) + '\n')

        with open(os.path.join(output_dir_language, 'dev.tsv'), 'w') as f:
            with open(f"{DATASET_PATH}mspider/dev.json", "r", encoding="utf-8") as f_in:
                examples = json.load(f_in)
            for data in examples:
                database_id = data['db_id']
                # if database_id not in count:
                #     count[database_id] = -1
                # count[database_id] += 1
                # interaction_id = count[database_id]
                interaction_id = data['interaction_id'] # We maintain the interaction_id in multilingual examples

                # Read Schema
                schema_list = []
                prev_table = None
                for column_name in column_names_surface_form[database_id]:
                    current_table = column_name.split('.')[0]
                    if current_table != prev_table:
                        schema_list.append(tokenizer.sep_token)
                        prev_table = current_table
                    schema_list.append(column_name)

                # Read utterance
                utterance = data['question'][language].lower()
                raw_utterance = utterance

                # Form the utterance_schema input
                utterance_schema = tokenizer.cls_token + ' ' + utterance + ' ' + ' '.join(
                    schema_list) + ' ' + tokenizer.sep_token
                utterance_tokenized = utterance_schema.split()
                len_list.append(len(utterance_tokenized))

                mr = removefrom_sql[database_id][interaction_id].split()
                query_pointers = []
                for token in mr:
                    if token in sql_keywords:
                        query_pointers.append(token)
                        sql_keywords_appear.add(token)
                    else:
                        assert (token in schema_list)
                        span_idx = schema_list.index(token) + len((tokenizer.cls_token + ' ' + utterance).split())
                        assert (token == utterance_schema.split()[span_idx])
                        query_pointers.append('@ptr{}'.format(span_idx))
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance_schema, query_pointers, database_id]) + '\n')

        # train_f = 'data/spider_data_removefrom/train.json'
        # with open(train_f) as f:
        #     train_data = json.load(f)
        # with open(os.path.join(output_dir_language, 'train.tsv'), 'w') as f:
        #     for data in train_data:
        #         utterance, utterance_schema, query_pointers, query, database_id = process_data(data)
        #         if query_pointers:
        #             f.write('\t'.join([utterance, utterance_schema, query_pointers, database_id]) + '\n')

        # dev_f = 'data/spider_data_removefrom/dev.json'
        # with open(dev_f) as f:
        #     dev_data = json.load(f)
        # with open(os.path.join(output_dir, 'dev.tsv'), 'w') as f:
        #     for data in dev_data:
        #         utterance, utterance_schema, query_pointers, query, database_id = process_data(data)
        #         if query_pointers:
        #             f.write('\t'.join([utterance, utterance_schema, query_pointers, database_id]) + '\n')
        # print(sql_keywords_appear - set(sql_keywords))
        # print(set(sql_keywords) - sql_keywords_appear)
        assert (sql_keywords_appear <= set(sql_keywords))
        with open(os.path.join(output_dir_language, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(sql_keywords):
                f.write(token + '\n')


def read_matis_old(model, output_dir):
    # languages = ['en','de','fr','th','es','hi']
    languages = ['en']

    if model in ['bert-base-cased', 'bert-base-multilingual-cased']:
        tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
    elif model in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(model, add_prefix_space=True)
    elif model in ['xlm-roberta-base', 'xlm-roberta-large']:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model)
    elif model in ['facebook/bart-base', 'facebook/bart-large']:
        tokenizer = BartTokenizer.from_pretrained(model, add_prefix_space=True)

    airports = airportsdata.load("IATA")

    for language in languages:
        output_dir = os.path.join(output_dir, 'matis', model.replace('/', '_'), language)

        output_dir_sql = os.path.join(output_dir, "sql")

        if not os.path.exists(output_dir_sql):
            os.makedirs(output_dir_sql)

        output_vocab_sql = set()
        len_list_sql = []

        PREDS = ["AS", "ON", "AND", "OR", "FROM"]
        DIGITS = ['0', '1', '1000', '1030', '1130', '1159', '1200', '1230', '1300', '1330', '14', '1400', '1430',
                  '1500',
                  '1530', '1600', '1630', '1645', '1700', '1730', '1745', '1759', '1800', '1830', '1845', '1900',
                  '1930',
                  '1991', '2', '2000', '2010', '2030', '2100', '2130', '2159', '2200', '2330', '2359', '2400', '3',
                  '300',
                  '301', '4', '400', '430', '497', '5', '500', '530', '540', '6', '600', '601', '630', '7', '730',
                  '766',
                  '8', '800', '830', '9', '900', '930']

        with open(os.path.join(output_dir_sql, "dev_digit.tsv"), "w") as f_digit:
            with open(os.path.join(output_dir_sql, "dev.tsv"), "w") as f:
                utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}matis/dev.json", "en", "sql")
                for utterance, mr in zip(utterances, mrs):
                    utterances_tokenized = tokenizer.tokenize(utterance)
                    len_list_sql.append(len(utterances_tokenized))

                    mr = re.sub("\bAA\b", "american", mr)
                    mr = re.sub("\bCP\b", "canadian", mr)
                    mr = re.sub("\bEA\b", "eastern", mr)
                    mr = re.sub("\bHP\b", "america west", mr)
                    mr = re.sub("\bLP\b", "lufthansa", mr)
                    mr = re.sub("\bNW\b", "northwest", mr)
                    mr = re.sub("\bUA\b", "united", mr)
                    mr = re.sub("\bYX\b", "express", mr)
                    mr = re.sub("\bTW\b", "twa", mr)
                    mr = re.sub("\bLH\b", "lufthansa", mr)
                    mr = re.sub("\bDL\b", "delta", mr)
                    mr = re.sub("\bML\b", "midway", mr)
                    mr = re.sub("\bNX\b", "nationair", mr)
                    mr = re.sub("\bWN\b", "southwest", mr)

                    pattern = re.compile('"(.*?)"')
                    res_ = pattern.findall(mr)
                    res = []

                    for token in res_:
                        res.extend(token.split())

                    mr_padding = mr.replace('"', ' " ').replace(';', ' ;').replace('(', ' ( ').replace(')',
                                                                                                       ' ) ').replace(
                        ",", " , ").replace('  ', ' ')
                    if re.search(r"\b{}\b".format("los angeles"), utterance):
                        mr_padding = re.sub(r"\bLOS ANGELES\b", "los angeles", mr_padding)

                    # print(mr_padding)

                    for mr_token in mr_padding.split():
                        if mr_token in airports and mr_token.lower() not in utterance.split():
                            temp = []
                            airport_name = airports[mr_token]['name']
                            for token in airport_name.lower().split():
                                if token in utterance.split():
                                    temp.append(token)
                            temp = " ".join(temp)
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token.isdigit() and str(int(int(mr_token) / 100) - 12) in utterance.split():
                            index = utterance.split().index(str(int(int(mr_token) / 100) - 12))
                            temp = str(int(int(mr_token) / 100) - 12)
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), str(int(int(mr_token) / 100) - 12),
                                                    mr_padding)
                        elif mr_token.isdigit() and int(mr_token) % 100 == 0 and str(
                                int(int(mr_token) / 100)) in utterance.split():
                            index = utterance.split().index(str(int(int(mr_token) / 100)))
                            temp = str(int(int(mr_token) / 100))
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token.isdigit() and str(
                                (int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100) in utterance.split():
                            index = utterance.split().index(
                                str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100))
                            temp = str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100)
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                    str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100),
                                                    mr_padding)
                        elif mr_token not in res and mr_token.lower() + 's' in utterance.split() and mr_token not in PREDS:
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower() + 's', mr_padding)
                        elif mr_token not in res and mr_token.lower() in utterance.split() and mr_token not in PREDS: \
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower(), mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                                num2words.num2words(int(mr_token), ordinal=True).split("-"))), utterance):
                            mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                " ".join(num2words.num2words(int(mr_token), ordinal=True).split("-")),
                                                mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                                num2words.num2words(int(mr_token)).split("-"))), utterance):
                            mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                " ".join(num2words.num2words(int(mr_token)).split("-")),
                                                mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 12 and calendar.month_name[
                            int(mr_token)].lower() in utterance.split():
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), calendar.month_name[int(mr_token)].lower(),
                                                mr_padding)
                        elif us.states.lookup(mr_token) and mr_token != "AS":
                            temp = str(us.states.lookup(mr_token)).lower()
                            if mr_token == "DC":
                                temp = "washington dc"
                            elif mr_token == "BB":
                                temp = "breakfast"
                            temp_ = []
                            for token in temp.split():
                                if token in utterance.split():
                                    temp_.append(token)
                            temp = " ".join(temp_)
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token == "ANGELES" and "la" in utterance.split():
                            mr_padding = re.sub(r"\bANGELES\b", "la", mr_padding)

                    mr_padding = mr_padding.split()
                    # print(mr_padding)

                    for mr_token in mr_padding:
                        if mr_token.lower() in utterance.split() and mr_token not in PREDS:
                            tokens = tokenizer.tokenize(mr_token.lower())
                            pointers = []
                            for token in tokens:
                                pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                            mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        elif mr_token not in res:
                            output_vocab_sql.add(mr_token)

                    f.write('\t'.join([utterance, " ".join(utterances_tokenized), " ".join(mr_padding)]) + '\n')
                    for token in mr_padding:
                        if token in DIGITS:
                            f_digit.write('\t'.join([utterance, mr]) + '\n')

        with open(os.path.join(output_dir_sql, "test_digit.tsv"), "w") as f_digit:
            with open(os.path.join(output_dir_sql, "test.tsv"), "w") as f:
                utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}matis/test.json", "en", "sql")
                for utterance, mr in zip(utterances, mrs):
                    utterances_tokenized = tokenizer.tokenize(utterance)
                    len_list_sql.append(len(utterances_tokenized))

                    mr = re.sub("\bAA\b", "american", mr)
                    mr = re.sub("\bCP\b", "canadian", mr)
                    mr = re.sub("\bEA\b", "eastern", mr)
                    mr = re.sub("\bHP\b", "america west", mr)
                    mr = re.sub("\bLP\b", "lufthansa", mr)
                    mr = re.sub("\bNW\b", "northwest", mr)
                    mr = re.sub("\bUA\b", "united", mr)
                    mr = re.sub("\bYX\b", "express", mr)
                    mr = re.sub("\bTW\b", "twa", mr)
                    mr = re.sub("\bLH\b", "lufthansa", mr)
                    mr = re.sub("\bDL\b", "delta", mr)
                    mr = re.sub("\bML\b", "midway", mr)
                    mr = re.sub("\bNX\b", "nationair", mr)
                    mr = re.sub("\bWN\b", "southwest", mr)

                    pattern = re.compile('"(.*?)"')
                    res_ = pattern.findall(mr)
                    res = []

                    for token in res_:
                        res.extend(token.split())

                    mr_padding = mr.replace('"', ' " ').replace(';', ' ;').replace('(', ' ( ').replace(')',
                                                                                                       ' ) ').replace(
                        ",", " , ").replace('  ', ' ')
                    if re.search(r"\b{}\b".format("los angeles"), utterance):
                        mr_padding = re.sub(r"\bLOS ANGELES\b", "los angeles", mr_padding)

                    # print(mr_padding)

                    for mr_token in mr_padding.split():
                        if mr_token in airports and mr_token.lower() not in utterance.split():
                            temp = []
                            airport_name = airports[mr_token]['name']
                            for token in airport_name.lower().split():
                                if token in utterance.split():
                                    temp.append(token)
                            temp = " ".join(temp)
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token.isdigit() and str(int(int(mr_token) / 100) - 12) in utterance.split():
                            index = utterance.split().index(str(int(int(mr_token) / 100) - 12))
                            temp = str(int(int(mr_token) / 100) - 12)
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), str(int(int(mr_token) / 100) - 12),
                                                    mr_padding)
                        elif mr_token.isdigit() and int(mr_token) % 100 == 0 and str(
                                int(int(mr_token) / 100)) in utterance.split():
                            index = utterance.split().index(str(int(int(mr_token) / 100)))
                            temp = str(int(int(mr_token) / 100))
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token.isdigit() and str(
                                (int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100) in utterance.split():
                            index = utterance.split().index(
                                str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100))
                            temp = str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100)
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                    str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100),
                                                    mr_padding)
                        elif mr_token not in res and mr_token.lower() + 's' in utterance.split() and mr_token not in PREDS:
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower() + 's', mr_padding)
                        elif mr_token not in res and mr_token.lower() in utterance.split() and mr_token not in PREDS: \
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower(), mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                                num2words.num2words(int(mr_token), ordinal=True).split("-"))), utterance):
                            mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                " ".join(num2words.num2words(int(mr_token), ordinal=True).split("-")),
                                                mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                                num2words.num2words(int(mr_token)).split("-"))), utterance):
                            mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                " ".join(num2words.num2words(int(mr_token)).split("-")),
                                                mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 12 and calendar.month_name[
                            int(mr_token)].lower() in utterance.split():
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), calendar.month_name[int(mr_token)].lower(),
                                                mr_padding)
                        elif us.states.lookup(mr_token) and mr_token != "AS":
                            temp = str(us.states.lookup(mr_token)).lower()
                            if mr_token == "DC":
                                temp = "washington dc"
                            elif mr_token == "BB":
                                temp = "breakfast"
                            temp_ = []
                            for token in temp.split():
                                if token in utterance.split():
                                    temp_.append(token)
                            temp = " ".join(temp_)
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token == "ANGELES" and "la" in utterance.split():
                            mr_padding = re.sub(r"\bANGELES\b", "la", mr_padding)

                    mr_padding = mr_padding.split()
                    # print(mr_padding)

                    for mr_token in mr_padding:
                        if mr_token.lower() in utterance.split() and mr_token not in PREDS:
                            tokens = tokenizer.tokenize(mr_token.lower())
                            pointers = []
                            for token in tokens:
                                pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                            mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        elif mr_token not in res:
                            output_vocab_sql.add(mr_token)

                    f.write('\t'.join([utterance, " ".join(utterances_tokenized), " ".join(mr_padding)]) + '\n')
                    for token in mr_padding:
                        if token in DIGITS:
                            f_digit.write('\t'.join([utterance, mr]) + '\n')

        with open(os.path.join(output_dir_sql, "train_digit.tsv"), "w") as f_digit:
            with open(os.path.join(output_dir_sql, "train.tsv"), "w") as f:
                utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}matis/train.json", "en", "sql")
                for utterance, mr in zip(utterances, mrs):
                    utterances_tokenized = tokenizer.tokenize(utterance)
                    len_list_sql.append(len(utterances_tokenized))

                    mr = re.sub("\bAA\b", "american", mr)
                    mr = re.sub("\bCP\b", "canadian", mr)
                    mr = re.sub("\bEA\b", "eastern", mr)
                    mr = re.sub("\bHP\b", "america west", mr)
                    mr = re.sub("\bLP\b", "lufthansa", mr)
                    mr = re.sub("\bNW\b", "northwest", mr)
                    mr = re.sub("\bUA\b", "united", mr)
                    mr = re.sub("\bYX\b", "express", mr)
                    mr = re.sub("\bTW\b", "twa", mr)
                    mr = re.sub("\bLH\b", "lufthansa", mr)
                    mr = re.sub("\bDL\b", "delta", mr)
                    mr = re.sub("\bML\b", "midway", mr)
                    mr = re.sub("\bNX\b", "nationair", mr)
                    mr = re.sub("\bWN\b", "southwest", mr)

                    pattern = re.compile('"(.*?)"')
                    res_ = pattern.findall(mr)
                    res = []

                    for token in res_:
                        res.extend(token.split())

                    mr_padding = mr.replace('"', ' " ').replace(';', ' ;').replace('(', ' ( ').replace(')',
                                                                                                       ' ) ').replace(
                        ",", " , ").replace('  ', ' ')
                    if re.search(r"\b{}\b".format("los angeles"), utterance):
                        mr_padding = re.sub(r"\bLOS ANGELES\b", "los angeles", mr_padding)

                    for mr_token in mr_padding.split():
                        if mr_token in airports and mr_token.lower() not in utterance.split():
                            temp = []
                            airport_name = airports[mr_token]['name']
                            for token in airport_name.lower().split():
                                if token in utterance.split():
                                    temp.append(token)
                            temp = " ".join(temp)
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token.isdigit() and str(int(int(mr_token) / 100) - 12) in utterance.split():
                            index = utterance.split().index(str(int(int(mr_token) / 100) - 12))
                            temp = str(int(int(mr_token) / 100) - 12)
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), str(int(int(mr_token) / 100) - 12),
                                                    mr_padding)
                        elif mr_token.isdigit() and int(mr_token) % 100 == 0 and str(
                                int(int(mr_token) / 100)) in utterance.split():
                            index = utterance.split().index(str(int(int(mr_token) / 100)))
                            temp = str(int(int(mr_token) / 100))
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token.isdigit() and str(
                                (int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100) in utterance.split():
                            index = utterance.split().index(
                                str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100))
                            temp = str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100)
                            if "o'clock" in utterance.split()[index:]:
                                temp += " o'clock"
                            if "am" in utterance.split()[index:]:
                                temp += " am"
                            if "pm" in utterance.split()[index:]:
                                temp += " pm"
                            if "at" in utterance.split()[:index] or "before" in utterance.split()[
                                                                                :index] or "after" in utterance.split()[
                                                                                                      :index] or "from" in utterance.split()[
                                                                                                                           :index] or "to" in utterance.split()[
                                                                                                                                              :index] or "between" in utterance.split()[
                                                                                                                                                                      :index] or "and" in utterance.split()[
                                                                                                                                                                                          :index]:
                                mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                    str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100),
                                                    mr_padding)
                        elif mr_token not in res and mr_token.lower() + 's' in utterance.split() and mr_token not in PREDS:
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower() + 's', mr_padding)
                        elif mr_token not in res and mr_token.lower() in utterance.split() and mr_token not in PREDS: \
                                mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower(), mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                                num2words.num2words(int(mr_token), ordinal=True).split("-"))), utterance):
                            mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                " ".join(num2words.num2words(int(mr_token), ordinal=True).split("-")),
                                                mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                                num2words.num2words(int(mr_token)).split("-"))), utterance):
                            mr_padding = re.sub(r"\b{}\b".format(mr_token),
                                                " ".join(num2words.num2words(int(mr_token)).split("-")),
                                                mr_padding)
                        elif mr_token.isdigit() and 1 <= int(mr_token) <= 12 and calendar.month_name[
                            int(mr_token)].lower() in utterance.split():
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), calendar.month_name[int(mr_token)].lower(),
                                                mr_padding)
                        elif us.states.lookup(mr_token) and mr_token != "AS":
                            temp = str(us.states.lookup(mr_token)).lower()
                            if mr_token == "DC":
                                temp = "washington dc"
                            elif mr_token == "BB":
                                temp = "breakfast"
                            temp_ = []
                            for token in temp.split():
                                if token in utterance.split():
                                    temp_.append(token)
                            temp = " ".join(temp_)
                            mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                        elif mr_token == "ANGELES" and "la" in utterance.split():
                            mr_padding = re.sub(r"\bANGELES\b", "la", mr_padding)

                    mr_padding = mr_padding.split()
                    # print(mr_padding)

                    for mr_token in mr_padding:
                        if mr_token.lower() in utterance.split() and mr_token not in PREDS:
                            tokens = tokenizer.tokenize(mr_token.lower())
                            pointers = []
                            for token in tokens:
                                pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                            mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        elif mr_token not in res:
                            output_vocab_sql.add(mr_token)

                    f.write('\t'.join([utterance, " ".join(utterances_tokenized), " ".join(mr_padding)]) + '\n')
                    for token in mr_padding:
                        if token in DIGITS:
                            f_digit.write('\t'.join([utterance, mr]) + '\n')

        with open(os.path.join(output_dir_sql, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_sql)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_sql)):
                f.write(token + '\n')


def read_matis(model, output_dir):
    languages = ['en', 'es', 'de', 'fr', 'pt', 'zh', 'ja']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['en']

    tokenizer = get_tokenizer(model)

    airports = airportsdata.load("IATA")

    for language in languages:
        output_dir_sql = os.path.join(output_dir, 'matis', model.replace('/', '_'), language)
        output_dir_sql = os.path.join(output_dir_sql, "sql")
        if not os.path.exists(output_dir_sql):
            os.makedirs(output_dir_sql)

        output_vocab_sql = set()
        len_list_sql = []

        PREDS = ["AS", "ON", "AND", "OR", "FROM"]
        PREDS_lower = ["as", "on", "and", "or", "from"]
        DIGITS = ['0', '1', '1000', '1030', '1130', '1159', '1200', '1230', '1300', '1330', '14', '1400', '1430',
                  '1500',
                  '1530', '1600', '1630', '1645', '1700', '1730', '1745', '1759', '1800', '1830', '1845', '1900',
                  '1930',
                  '1991', '2', '2000', '2010', '2030', '2100', '2130', '2159', '2200', '2330', '2359', '2400', '3',
                  '300',
                  '301', '4', '400', '430', '497', '5', '500', '530', '540', '6', '600', '601', '630', '7', '730',
                  '766',
                  '8', '800', '830', '9', '900', '930']

        # with open(os.path.join(output_dir_sql, "dev_digit.tsv"), "w") as f_digit:
        with open(os.path.join(output_dir_sql, "dev.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}matis/dev.json", language, "sql")
            print('dev', len(utterances))
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_sql.append(len(utterances_tokenized))

                mr_original = mr
                # we cannot chang mr
                # mr = re.sub('"AA"', '"american"', mr)
                # mr = re.sub('"CP"', '"canadian"', mr)
                # mr = re.sub('"EA"', '"eastern"', mr)
                # mr = re.sub('"HP"', '"america west"', mr)
                # mr = re.sub('"NW"', '"northwest"', mr)
                # mr = re.sub('"UA"', '"united"', mr)
                # mr = re.sub('"YX"', '"express"', mr)
                # mr = re.sub('"TW"', '"twa"', mr)
                # mr = re.sub('"LH"', '"lufthansa"', mr)
                # mr = re.sub('"DL"', '"delta"', mr)
                # mr = re.sub('"ML"', '"midway"', mr)
                # mr = re.sub('"NX"', '"nationair"', mr)
                # mr = re.sub('"WN"', '"southwest"', mr)

                pattern = re.compile('"(.*?)"')
                res_ = pattern.findall(mr)
                res = []

                for token in res_:
                    res.extend(token.split())

                mr_padding = mr.replace('"', ' " ').replace(';', ' ;').replace('(', ' ( ').replace(')', ' ) ').replace(
                    ",", " , ").replace('.', ' . ').replace('  ', ' ').replace('   ', ' ').strip(';')
                # if re.search(r"\b{}\b".format("los angeles"), utterance):
                #     print(mr_padding)
                #     mr_padding = re.sub(r"\bLOS ANGELES\b", "los angeles", mr_padding)
                #     print(mr_padding)
                #     exit()

                # print('1', mr_padding.lower())

                # for mr_token in mr_padding.split():
                #     # replace the 3-digit airport code in mr with its name
                #     if mr_token in airports and mr_token in res_:
                #         temp = []
                #         airport_name = airports[mr_token]['name']
                #         for token in airport_name.lower().split():
                #             if token in utterance.split():
                #                 temp.append(token)
                #         temp = " ".join(temp)
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     # replace time in mr such as 1500 with its real mapping in utterance such as
                #     # at/before/after 3, from '3' to 4, from 2 to '3', between '3' and 4, and between 2 and '3'
                #     # 3 could also be in the form of 3 pm, 3 o'clock and 3 o'clock pm
                #     elif mr_token.isdigit() and str(int(int(mr_token) / 100) - 12) in utterance.split():
                #         index = utterance.split().index(str(int(int(mr_token) / 100) - 12))
                #         temp = str(int(int(mr_token) / 100) - 12)
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "pm" in utterance.split()[index:]:
                #             temp += " pm"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token), str(int(int(mr_token) / 100) - 12),
                #                                 mr_padding)
                #     # replace time in mr such as 500 with its real mapping in utterance such as
                #     # at/before/after 5, from '5' to 6, from 4 to '5', between '5' and 6, and between 4 and '5'
                #     # 5 could also be in the form of 5 am, 5 o'clock and 5 o'clock am
                #     elif mr_token.isdigit() and int(mr_token) % 100 == 0 and str(
                #             int(int(mr_token) / 100)) in utterance.split():
                #         index = utterance.split().index(str(int(int(mr_token) / 100)))
                #         temp = str(int(int(mr_token) / 100))
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "am" in utterance.split()[index:]:
                #             temp += " am"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     # replace time in mr such as 1505 with its real mapping in utterance such as
                #     # at/before/after 1505, from '1505' to 1530, from 1030 to '1505', between '1505' and 1530,
                #     # and between 1030 and '1505'
                #     # 5 could also be in the form of 1505 pm, 1505 o'clock and 1505 o'clock pm
                #     elif mr_token.isdigit() and str(
                #             (int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100) in utterance.split():
                #         index = utterance.split().index(
                #             str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100))
                #         temp = str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100)
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "am" in utterance.split()[index:]:
                #             temp += " am"
                #         if "pm" in utterance.split()[index:]:
                #             temp += " pm"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                                 str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100),
                #                                 mr_padding)
                #     # check the word shown in lower case in the utterance while in upper case in mr
                #     elif mr_token not in res and mr_token.lower() in utterance.split() and mr_token not in PREDS:
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower(), mr_padding)
                #     # replace token: atlanta in mr with atlanta's in utterance
                #     elif mr_token in res and mr_token not in utterances_tokenized and mr_token.lower() + "'s" in \
                #             utterance.split() and mr_token not in PREDS:
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower() + "'s", mr_padding)
                #     # replace 22 in mr with twenty second
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                #             num2words.num2words(int(mr_token), ordinal=True).split("-"))), utterance):
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                             " ".join(num2words.num2words(int(mr_token), ordinal=True).split("-")),
                #                             mr_padding)
                #     # replace 22 in mr with twenty two
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                #             num2words.num2words(int(mr_token)).split("-"))), utterance):
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                             " ".join(num2words.num2words(int(mr_token)).split("-")),
                #                             mr_padding)
                #     # replace 1 in mr with january
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 12 and calendar.month_name[
                #         int(mr_token)].lower() in utterance.split():
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), calendar.month_name[int(mr_token)].lower(),
                #                             mr_padding)
                #     # replace state codes with their names
                #     elif us.states.lookup(mr_token) and mr_token != "AS":
                #         temp = str(us.states.lookup(mr_token)).lower()
                #         if mr_token == "DC":
                #             temp = "washington dc"
                #         elif mr_token == "BB":
                #             temp = "breakfast"
                #         temp_ = []
                #         for token in temp.split():
                #             if token in utterance.split():
                #                 temp_.append(token)
                #         temp = " ".join(temp_)
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     elif mr_token == "ANGELES" and "la" in utterance.split():
                #         mr_padding = re.sub(r"\bANGELES\b", "la", mr_padding)

                mr_padding = mr_padding.lower().split()
                # print('2', " ".join(mr_padding))
                # TODO: why missing AND in where conditions
                # airport code AND refers to Anderson Regional Airport which is not involved here

                query_pointers = []
                # print(res)
                for mr_token in mr_padding:
                    # print(mr_token)
                    if mr_token in utterances_tokenized and mr_token not in PREDS_lower:
                        # TODO: only values need to change to pointers in matis
                        # tokens = tokenizer.tokenize(mr_token.lower())
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                    else:  # TODO: why check mr_token not in res here?
                        # condition removed
                        output_vocab_sql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')
                # f.write('\t'.join([utterance, " ".join(utterances_tokenized), " ".join(mr_padding)]) + '\n')
                # TODO: how to deal with digits?
                # haven't figured out how to deal with the rest digits which are not filtered by the above if clauses ...
                # for token in mr_padding:
                #     if token in DIGITS:
                #         f_digit.write('\t'.join([utterance, mr]) + '\n')
                #         break # TODO: should we break here?
                #         # what if there are more than 1 digit in 1 mr?

        # with open(os.path.join(output_dir_sql, "test_digit.tsv"), "w") as f_digit:
        with open(os.path.join(output_dir_sql, "test.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}matis/test.json", language, "sql")
            print('test', len(utterances))
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_sql.append(len(utterances_tokenized))

                mr_original = mr
                # mr = re.sub('"AA"', '"american"', mr)
                # mr = re.sub('"CP"', '"canadian"', mr)
                # mr = re.sub('"EA"', '"eastern"', mr)
                # mr = re.sub('"HP"', '"america west"', mr)
                # mr = re.sub('"NW"', '"northwest"', mr)
                # mr = re.sub('"UA"', '"united"', mr)
                # mr = re.sub('"YX"', '"express"', mr)
                # mr = re.sub('"TW"', '"twa"', mr)
                # mr = re.sub('"LH"', '"lufthansa"', mr)
                # mr = re.sub('"DL"', '"delta"', mr)
                # mr = re.sub('"ML"', '"midway"', mr)
                # mr = re.sub('"NX"', '"nationair"', mr)
                # mr = re.sub('"WN"', '"southwest"', mr)

                pattern = re.compile('"(.*?)"')
                res_ = pattern.findall(mr)
                res = []

                for token in res_:
                    res.extend(token.split())

                mr_padding = mr.replace('"', ' " ').replace(';', ' ;').replace('(', ' ( ').replace(')', ' ) ').replace(
                    ",", " , ").replace('  ', ' ').replace('.', ' . ').strip(';')
                # if re.search(r"\b{}\b".format("los angeles"), utterance):
                #     mr_padding = re.sub(r"\bLOS ANGELES\b", "los angeles", mr_padding)

                # print(mr_padding)

                # for mr_token in mr_padding.split():
                #     if mr_token in airports and mr_token in res_:
                #         temp = []
                #         airport_name = airports[mr_token]['name']
                #         for token in airport_name.lower().split():
                #             if token in utterance.split():
                #                 temp.append(token)
                #         temp = " ".join(temp)
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     elif mr_token.isdigit() and str(int(int(mr_token) / 100) - 12) in utterance.split():
                #         index = utterance.split().index(str(int(int(mr_token) / 100) - 12))
                #         temp = str(int(int(mr_token) / 100) - 12)
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "am" in utterance.split()[index:]:
                #             temp += " am"
                #         if "pm" in utterance.split()[index:]:
                #             temp += " pm"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token), str(int(int(mr_token) / 100) - 12),
                #                                 mr_padding)
                #     elif mr_token.isdigit() and int(mr_token) % 100 == 0 and str(
                #             int(int(mr_token) / 100)) in utterance.split():
                #         index = utterance.split().index(str(int(int(mr_token) / 100)))
                #         temp = str(int(int(mr_token) / 100))
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "am" in utterance.split()[index:]:
                #             temp += " am"
                #         if "pm" in utterance.split()[index:]:
                #             temp += " pm"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     elif mr_token.isdigit() and str(
                #             (int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100) in utterance.split():
                #         index = utterance.split().index(
                #             str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100))
                #         temp = str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100)
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "am" in utterance.split()[index:]:
                #             temp += " am"
                #         if "pm" in utterance.split()[index:]:
                #             temp += " pm"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                                 str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100),
                #                                 mr_padding)
                #     elif mr_token not in res and mr_token.lower() in utterance.split() and mr_token not in PREDS: \
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower(), mr_padding)
                #     elif mr_token in res and mr_token not in utterances_tokenized and mr_token.lower() + "'s" in \
                #             utterance.split() and mr_token not in PREDS:
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower() + "'s", mr_padding)
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                #             num2words.num2words(int(mr_token), ordinal=True).split("-"))), utterance):
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                             " ".join(num2words.num2words(int(mr_token), ordinal=True).split("-")),
                #                             mr_padding)
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                #             num2words.num2words(int(mr_token)).split("-"))), utterance):
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                             " ".join(num2words.num2words(int(mr_token)).split("-")),
                #                             mr_padding)
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 12 and calendar.month_name[
                #         int(mr_token)].lower() in utterance.split():
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), calendar.month_name[int(mr_token)].lower(),
                #                             mr_padding)
                #     elif us.states.lookup(mr_token) and mr_token != "AS":
                #         temp = str(us.states.lookup(mr_token)).lower()
                #         if mr_token == "DC":
                #             temp = "washington dc"
                #         elif mr_token == "BB":
                #             temp = "breakfast"
                #         temp_ = []
                #         for token in temp.split():
                #             if token in utterance.split():
                #                 temp_.append(token)
                #         temp = " ".join(temp_)
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     elif mr_token == "ANGELES" and "la" in utterance.split():
                #         mr_padding = re.sub(r"\bANGELES\b", "la", mr_padding)

                mr_padding = mr_padding.lower().split()
                # print(mr_padding)

                query_pointers = []
                for mr_token in mr_padding:
                    # print(mr_token)
                    if mr_token in utterances_tokenized and mr_token not in PREDS_lower:
                        # tokens = tokenizer.tokenize(mr_token.lower())
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                    elif mr_token not in res:
                        output_vocab_sql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')
                # f.write('\t'.join([utterance, " ".join(utterances_tokenized), " ".join(mr_padding)]) + '\n')
                # for token in mr_padding:
                #     if token in DIGITS:
                #         f_digit.write('\t'.join([utterance, mr]) + '\n')
                #         # break

        # with open(os.path.join(output_dir_sql, "train_digit.tsv"), "w") as f_digit:
        with open(os.path.join(output_dir_sql, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}matis/train.json", language, "sql")
            print('train', len(utterances))
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + raw_utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_sql.append(len(utterances_tokenized))

                mr_original = mr
                # mr = re.sub('"AA"', '"american"', mr)
                # mr = re.sub('"CP"', '"canadian"', mr)
                # mr = re.sub('"EA"', '"eastern"', mr)
                # mr = re.sub('"HP"', '"america west"', mr)
                # mr = re.sub('"NW"', '"northwest"', mr)
                # mr = re.sub('"UA"', '"united"', mr)
                # mr = re.sub('"YX"', '"express"', mr)
                # mr = re.sub('"TW"', '"twa"', mr)
                # mr = re.sub('"LH"', '"lufthansa"', mr)
                # mr = re.sub('"DL"', '"delta"', mr)
                # mr = re.sub('"ML"', '"midway"', mr)
                # mr = re.sub('"NX"', '"nationair"', mr)
                # mr = re.sub('"WN"', '"southwest"', mr)

                pattern = re.compile('"(.*?)"')
                res_ = pattern.findall(mr)
                res = []

                for token in res_:
                    res.extend(token.split())

                mr_padding = mr.replace('"', ' " ').replace(';', ' ;').replace('(', ' ( ').replace(')', ' ) ').replace(
                    ",", " , ").replace('  ', ' ').replace('.', ' . ').strip(';')
                # if re.search(r"\b{}\b".format("los angeles"), utterance):
                #     mr_padding = re.sub(r"\bLOS ANGELES\b", "los angeles", mr_padding)

                # for mr_token in mr_padding.split():
                #     if mr_token in airports and mr_token in res_:
                #         temp = []
                #         airport_name = airports[mr_token]['name']
                #         for token in airport_name.lower().split():
                #             if token in utterance.split():
                #                 temp.append(token)
                #         temp = " ".join(temp)
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     elif mr_token.isdigit() and str(int(int(mr_token) / 100) - 12) in utterance.split():
                #         index = utterance.split().index(str(int(int(mr_token) / 100) - 12))
                #         temp = str(int(int(mr_token) / 100) - 12)
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "am" in utterance.split()[index:]:
                #             temp += " am"
                #         if "pm" in utterance.split()[index:]:
                #             temp += " pm"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token), str(int(int(mr_token) / 100) - 12),
                #                                 mr_padding)
                #     elif mr_token.isdigit() and int(mr_token) % 100 == 0 and str(
                #             int(int(mr_token) / 100)) in utterance.split():
                #         index = utterance.split().index(str(int(int(mr_token) / 100)))
                #         temp = str(int(int(mr_token) / 100))
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "am" in utterance.split()[index:]:
                #             temp += " am"
                #         if "pm" in utterance.split()[index:]:
                #             temp += " pm"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     elif mr_token.isdigit() and str(
                #             (int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100) in utterance.split():
                #         index = utterance.split().index(
                #             str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100))
                #         temp = str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100)
                #         if "o'clock" in utterance.split()[index:]:
                #             temp += " o'clock"
                #         if "am" in utterance.split()[index:]:
                #             temp += " am"
                #         if "pm" in utterance.split()[index:]:
                #             temp += " pm"
                #         if "at" in utterance.split()[:index] or "before" in utterance.split()[
                #                                                             :index] or "after" in utterance.split()[
                #                                                                                   :index] or "from" in utterance.split()[
                #                                                                                                        :index] or "to" in utterance.split()[
                #                                                                                                                           :index] or "between" in utterance.split()[
                #                                                                                                                                                   :index] or "and" in utterance.split()[
                #                                                                                                                                                                       :index]:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                                 str((int(int(mr_token) / 100) - 12) * 100 + int(mr_token) % 100),
                #                                 mr_padding)
                #     elif mr_token not in res and mr_token.lower() in utterance.split() and mr_token not in PREDS:
                #             mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower(), mr_padding)
                #     elif mr_token in res and mr_token not in utterances_tokenized and mr_token.lower() + "'s" in \
                #             utterance.split() and mr_token not in PREDS:
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), mr_token.lower() + "'s", mr_padding)
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                #             num2words.num2words(int(mr_token), ordinal=True).split("-"))), utterance):
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                             " ".join(num2words.num2words(int(mr_token), ordinal=True).split("-")),
                #                             mr_padding)
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 31 and re.search(r"\b{}\b".format(" ".join(
                #             num2words.num2words(int(mr_token)).split("-"))), utterance):
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token),
                #                             " ".join(num2words.num2words(int(mr_token)).split("-")),
                #                             mr_padding)
                #     elif mr_token.isdigit() and 1 <= int(mr_token) <= 12 and calendar.month_name[
                #         int(mr_token)].lower() in utterance.split():
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), calendar.month_name[int(mr_token)].lower(),
                #                             mr_padding)
                #     elif us.states.lookup(mr_token) and mr_token != "AS":
                #         temp = str(us.states.lookup(mr_token)).lower()
                #         if mr_token == "DC":
                #             temp = "washington dc"
                #         elif mr_token == "BB":
                #             temp = "breakfast"
                #         temp_ = []
                #         for token in temp.split():
                #             if token in utterance.split():
                #                 temp_.append(token)
                #         temp = " ".join(temp_)
                #         mr_padding = re.sub(r"\b{}\b".format(mr_token), temp, mr_padding)
                #     elif mr_token == "ANGELES" and "la" in utterance.split():
                #         mr_padding = re.sub(r"\bANGELES\b", "la", mr_padding)

                mr_padding = mr_padding.lower().split()
                # print(mr_padding)

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in utterances_tokenized and mr_token not in PREDS_lower:
                        # tokens = tokenizer.tokenize(mr_token.lower())
                        # pointers = []
                        # for token in tokens:
                        #     pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        # mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                        query_pointers.append("@ptr{}".format(utterances_tokenized.index(mr_token)))
                    elif mr_token not in res:
                        output_vocab_sql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')
                # f.write('\t'.join([utterance, " ".join(utterances_tokenized), " ".join(mr_padding)]) + '\n')
                # for token in mr_padding:
                #     if token in DIGITS:
                #         f_digit.write('\t'.join([utterance, mr]) + '\n')
                #         # break

        with open(os.path.join(output_dir_sql, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_sql)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_sql)):
                f.write(token + '\n')


def read_mfree917_old(model, output_dir):
    # languages = ['en','de','fr','th','es','hi']
    languages = ['en']

    if model in ['bert-base-cased', 'bert-base-multilingual-cased']:
        tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
    elif model in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(model, add_prefix_space=True)
    elif model in ['xlm-roberta-base', 'xlm-roberta-large']:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model)
    elif model in ['facebook/bart-base', 'facebook/bart-large']:
        tokenizer = BartTokenizer.from_pretrained(model, add_prefix_space=True)

    for language in languages:
        output_dir = os.path.join(output_dir, 'mfree917', model.replace('/', '_'), language)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        output_dir_lambda = os.path.join(output_dir, "lambda")

        if not os.path.exists(output_dir_lambda):
            os.makedirs(output_dir_lambda)

        output_vocab_lambda = set()
        len_list_lambda = []

        with open(os.path.join(output_dir_lambda, "test.tsv"), "w") as f:
            utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mfree917/test.json", "en", "lambda")
            for utterance, mr in zip(utterances, mrs):
                utterances_tokenized = tokenizer.tokenize(utterance)
                len_list_lambda.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(":", " :").replace("!", " ! ").replace(
                    "de_sterrennacht", "starry night").replace("alluminum_alloy_v6", "alluminum-alloy v6").replace(
                    "   ", " ").replace("  ", " ")
                pattern = re.compile(r'en\.([a-zA-Z0-9_-]+)')
                res = pattern.findall(mr_padding)
                pattern = re.compile(r'\(\bdate ([0-9][0-9][0-9][0-9])')
                res.extend(pattern.findall(mr))

                if utterance == "what 's the horsepower of an alluminum-alloy v6 engine":
                    res.append("v6")
                elif utterance == "how heavy is a panasonic lumix dmc-tz3":
                    mr_padding = mr_padding.replace("dmc_tz3", "dmc-tz3")
                elif utterance == "how long is wired _ s gadget lab podcast":
                    mr_padding = mr_padding.replace("wireds_gadget_lab_podcast_podcast_feed",
                                                    "wireds_gadget_lab_podcast")
                elif utterance == "who is the newscaster on abc 6 news":
                    mr_padding = mr_padding.replace("news_presenter", "newscaster").replace("abc_news_washington_dc",
                                                                                            "abc_6_news")

                if utterance == "what titles were at stake in the the rumble in the jungle":
                    utterances_tokenized[utterances_tokenized.index("the")] = "###"

                mr_padding = mr_padding.replace("en.", "en. ").split()

                for mr_token in mr_padding:
                    if mr_token in res:
                        if "_" in mr_token:
                            mr_tokens = mr_token.split("_")
                            mr_padding = " ".join(mr_padding).replace(mr_token, " ".join(mr_token.split("_"))).split()
                        elif "-" in mr_token and "".join(mr_token.split("-")) in utterance.split():
                            mr_tokens = ["".join(mr_token.split("-"))]
                            mr_padding = " ".join(mr_padding).replace(mr_token, "".join(mr_token.split("-"))).split()
                        else:
                            mr_tokens = [mr_token]

                        for target_token in mr_tokens:
                            if target_token not in utterance.split() and target_token + "s" in utterance.split():
                                mr_padding[mr_padding.index(target_token)] = target_token + "s"
                                target_token += "s"
                                assert target_token in utterance.split()
                            elif target_token.endswith('s') and target_token[
                                                                :len(target_token) - 1] + " _ s" in utterance:
                                mr_padding[mr_padding.index(target_token)] = target_token[
                                                                             :len(target_token) - 1] + " _ s"
                                target_token = target_token[:len(target_token) - 1] + " _ s"
                            tokens = tokenizer.tokenize(target_token)
                            pointers = []
                            for token in tokens:
                                if token in utterances_tokenized and target_token in utterance.split():
                                    if "@ptr{}".format(utterances_tokenized.index(token)) in mr_padding:
                                        pointers.append("@ptr{}".format(
                                            utterances_tokenized.index(token, utterances_tokenized.index(token) + 1)))
                                    else:
                                        if utterance == "what titles were at stake in the the rumble in the jungle" and token == "in":
                                            pointers.append("@ptr{}".format(utterances_tokenized.index(token,
                                                                                                       utterances_tokenized.index(
                                                                                                           token) + 1)))
                                        else:
                                            pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                            if pointers:
                                mr_padding[mr_padding.index(target_token)] = " ".join(pointers)
                    else:
                        if not (mr_token == 'v6' or
                                mr_token == "panasonic_lumix_dmc-tz3" or
                                mr_token == "wireds_gadget_lab_podcast" or
                                mr_token == "newscaster" or
                                mr_token == "abc_6_news" or
                                mr_token == "night"):
                            output_vocab_lambda.add(mr_token)
                if "###" in utterances_tokenized:
                    utterances_tokenized[utterances_tokenized.index("###")] = "the"
                f.write('\t'.join([utterance, " ".join(utterances_tokenized), " ".join(mr_padding)]) + '\n')

        with open(os.path.join(output_dir_lambda, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mfree917/train.json", "en", "lambda")
            for utterance, mr in zip(utterances, mrs):
                utterances_tokenized = tokenizer.tokenize(utterance)
                len_list_lambda.append(len(utterances_tokenized))

                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(":", " :").replace("!", " ! ").replace(
                    "de_sterrennacht", "starry night").replace("alluminum_alloy_v6", "alluminum-alloy v6").replace(
                    "   ", " ").replace("  ", " ")
                pattern = re.compile(r'en\.([a-zA-Z0-9_-]+)')
                res = pattern.findall(mr_padding)
                pattern = re.compile(r'\(\bdate ([0-9][0-9][0-9][0-9])')
                res.extend(pattern.findall(mr))

                if utterance == "who was the 22nd president":
                    mr_padding = mr_padding.replace("22.0", "22nd")
                    res.append("22nd")
                elif utterance == "who won muhammad ali vs. joe frazier ii":
                    mr_padding = mr_padding.replace("ali-frazier_ii", "muhammad_ali_vs._joe_frazier_ii")
                elif utterance == "what movies won the golden globe award for best drama film":
                    mr_padding = mr_padding.replace("golden_globe_award_for_best_motion_picture_-_drama",
                                                    "golden_globe_award_for_best_drama_film")
                elif utterance == "what was the cover price of the x-men issue 1":
                    mr_padding = mr_padding.replace("the_x_men_1", "the_x-men_issue_1")
                elif utterance == "what stadium do the phillies play in":
                    mr_padding = mr_padding.replace("philadelphia_phillies", "phillies")
                elif utterance == "who got the gold medal in men _ s singles tennis at the 1896 summer olympics":
                    mr_padding = mr_padding.replace("tennis_at_the_1896_summer_olympics_mens_singles",
                                                    "men_%_s_singles_tennis_at_the_1896_summer_olympics")
                    utterance = utterance.replace("_", "%")
                    utterances_tokenized = tokenizer.tokenize(utterance)

                mr_padding = mr_padding.replace("en.", "en. ").split()

                for mr_token in mr_padding:
                    if mr_token in res:
                        if "_" in mr_token:
                            mr_tokens = mr_token.split("_")
                            mr_padding = " ".join(mr_padding).replace(mr_token, " ".join(mr_token.split("_"))).split()
                        elif "-" in mr_token and "".join(mr_token.split("-")) in utterance.split():
                            mr_tokens = ["".join(mr_token.split("-"))]
                            mr_padding = " ".join(mr_padding).replace(mr_token, "".join(mr_token.split("-"))).split()
                        else:
                            mr_tokens = [mr_token]

                        for target_token in mr_tokens:
                            if target_token not in utterance.split() and target_token + "s" in utterance.split():
                                mr_padding[mr_padding.index(target_token)] = target_token + "s"
                                target_token += "s"
                                assert target_token in utterance.split()
                            elif target_token.endswith('s') and target_token[
                                                                :len(target_token) - 1] + " _ s" in utterance:
                                mr_padding[mr_padding.index(target_token)] = target_token[
                                                                             :len(target_token) - 1] + " _ s"
                                target_token = target_token[:len(target_token) - 1] + " _ s"

                            tokens = tokenizer.tokenize(target_token)
                            pointers = []
                            for token in tokens:
                                if token in utterances_tokenized and target_token in utterance.split():
                                    if "@ptr{}".format(utterances_tokenized.index(token)) in mr_padding:
                                        if utterance == "what is the voltage of a aa alkaline battery" and token == "battery":
                                            pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                                        else:
                                            pointers.append("@ptr{}".format(utterances_tokenized.index(token,
                                                                                                       utterances_tokenized.index(
                                                                                                           token) + 1)))
                                    else:
                                        pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                            if pointers:
                                mr_padding[mr_padding.index(target_token)] = " ".join(pointers)
                    else:
                        if not (mr_token == '22nd' or
                                mr_token == "muhammad_ali_vs._joe_frazier_ii" or
                                mr_token == "golden_globe_award_for_best_drama_film" or
                                mr_token == "the_x-men_issue_1" or
                                mr_token == "phillies" or
                                mr_token == "men_%_s_singles_tennis_at_the_1896_summer_olympics"):
                            output_vocab_lambda.add(mr_token)
                if utterance == "who got the gold medal in men % s singles tennis at the 1896 summer olympics":
                    utterance = utterance.replace("%", "_")
                    utterances_tokenized = tokenizer.tokenize(utterance)
                f.write('\t'.join([utterance, " ".join(utterances_tokenized), " ".join(mr_padding)]) + '\n')

        with open(os.path.join(output_dir_lambda, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_lambda)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_lambda)):
                f.write(token + '\n')


def read_mfree917(model, output_dir):
    languages = ['en', 'de']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['en']

    tokenizer = get_tokenizer(model)

    for language in languages:
        output_dir_lambda = os.path.join(output_dir, 'mfree917', model.replace('/', '_'), language)
        output_dir_lambda = os.path.join(output_dir_lambda, "lambda")
        if not os.path.exists(output_dir_lambda):
            os.makedirs(output_dir_lambda)
        # print(language, output_dir_lambda)
        output_vocab_lambda = set()
        len_list_lambda = []

        with open(os.path.join(output_dir_lambda, "test.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mfree917/test.json", language, "lambda")
            for idx, (utterance, mr) in enumerate(zip(utterances, mrs)):
                raw_utterance = utterance

                # NO: correct utterance to ensure it mentions values in mr, so that we can create pointers
                # Yes: correct utterance only for annotation errors

                # utterance = utterance.replace('alluminum-alloy', "alluminum alloy")
                utterance = utterance.replace('philosopher _ s', 'philosophers')
                utterance = utterance.replace('fry _ s', 'frys')
                utterance = utterance.replace('charlie _ s', 'charlies')
                utterance = utterance.replace('paul _ s', 'pauls')
                utterance = utterance.replace('wired _ s', 'wireds')
                # utterance = utterance.replace('postcards', 'postcard _ s')
                # utterance = utterance.replace('engineers', 'engineer _ s')
                # utterance = utterance.replace('peseta', 'spanish peseta')
                # utterance = utterance.replace('poisoning', 'poison')
                # utterance = utterance.replace('ceo of savealot', 'chief executive officer of save-a-lot')
                utterance = utterance.replace('julie andrews', 'julie edwards')
                # utterance = utterance.replace('vancouver', 'vancouver british columbia')
                # utterance = utterance.replace('ny yankees', 'new york yankees')
                # utterance = utterance.replace("sgt. pepper 's", 'sgt peppers')
                # utterance = utterance.replace('300', '300 2007')
                # utterance = utterance.replace('wagner', 'richardwagner')
                # utterance = utterance.replace('home depot', 'the home depot')
                # utterance = utterance.replace('taylor made piano', 'taylor made piano a jazz history')
                # utterance = utterance.replace('titanic', 'titanic special edition dvd')
                # utterance = utterance.replace('primetieme emmy award for comedy series', 'primetime emmy award award for outstanding comedy series')
                # utterance = utterance.replace('in the world', 'in the world on earth')
                # utterance = utterance.replace('christian', 'christianity')
                # utterance = utterance.replace('new york city', 'new york ny')
                # utterance = utterance.replace('omarion', 'omarion grandberry')
                # utterance = utterance.replace('henry viii', 'king henry viii of england')
                # utterance = utterance.replace('cow', 'cattle')
                # utterance = utterance.replace('science museum', 'science museum great britain')
                # utterance = utterance.replace('john j. raskob', 'john j raskob')
                # utterance = utterance.replace('headache', 'severe headache')
                # utterance = utterance.replace('lithium batteries', 'lithium battery')
                # utterance = utterance.replace('new york ny subway', 'new york city subway')
                # utterance = utterance.replace('jcpenney', 'j c penney')
                # utterance = utterance.replace('construction of', 'initial design and construction of')
                # utterance = utterance.replace('nutty professor', 'the nutty professor 1996')
                # utterance = utterance.replace('tutor', 'tudor dynasty')
                # utterance = utterance.replace('german', 'germany')
                # utterance = utterance.replace('dmc-tz3', 'dmc tz3')
                # utterance = utterance.replace('come a can', 'come a beverage can')
                # utterance = utterance.replace('the red cross', 'the american red cross')
                # utterance = utterance.replace('wireds gadget lab podcast', 'wireds gadget lab podcast podcast feed')
                # utterance = utterance.replace('us navy', 'united states navy')
                # utterance = utterance.replace('syphilis', 'syphillis')
                # utterance = utterance.replace('raleigh', 'raleigh bicycle company')
                # utterance = utterance.replace('who is the newscaster on abc 6 news', 'who is the news presenter on abc news washington dc')
                # utterance = utterance.replace('what titles were at stake in the the rumble in the jungle', 'what titles were at stake in the rumble in the jungle')

                # add special tokens to utterance and split
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                # tokenize mr
                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(":", " :").replace("!", " ! ").replace(
                    "de_sterrennacht", "starry night").replace("   ", " ").replace("  ", " ")

                # find values in mr
                pattern = re.compile(r'en\.([a-zA-Z0-9_-]+)')
                res = pattern.findall(mr_padding)
                pattern = re.compile(r'\(\bdate ([0-9][0-9][0-9][0-9])')
                res.extend(pattern.findall(mr))

                mr_padding = mr_padding.replace("en.", "en. ").split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res:
                        if "_" in mr_token:
                            mr_tokens = mr_token.split("_")
                        # elif "-" in mr_token and "".join(mr_token.split("-")) in utterances_tokenized:
                        #     mr_tokens = ["".join(mr_token.split("-"))]
                        else:
                            mr_tokens = [mr_token]

                        pointer = sum(1 for target_token in mr_tokens if target_token in utterances_tokenized) == len(
                            mr_tokens)
                        if pointer:
                            for target_token in mr_tokens:
                                query_pointers.append("@ptr{}".format(utterances_tokenized.index(target_token)))
                        else:
                            output_vocab_lambda.add(mr_token)
                            query_pointers.append(mr_token)

                        # for target_token in mr_tokens:
                        #     if not target_token in utterances_tokenized:
                        #         # print(' 1 ', target_token, utterances_tokenized)
                        #         output_vocab_lambda.update(target_token)
                        #         query_pointers.append(target_token)
                        #     else:
                        #         query_pointers.append("@ptr{}".format(utterances_tokenized.index(target_token)))
                    else:
                        # output_vocab_lambda.add(mr_token)
                        # query_pointers.append(mr_token)
                        # Change fb :computer.computer.introduced to fb : computer . computer . introduced
                        output_vocab_lambda.update(mr_token.replace(":", " : ").replace(".", " . ").split())
                        query_pointers.extend(mr_token.replace(":", " : ").replace(".", " . ").split())
                # add special tokens to query_pointers
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_lambda, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mfree917/train.json", language, "lambda")
            for idx, (utterance, mr) in enumerate(zip(utterances, mrs)):
                raw_utterance = utterance

                # NO: correct utterance to ensure it mentions values in mr, so that we can create pointers
                # Yes: correct utterance only for annotation errors
                
                # utterance = utterance.replace('22nd', "22.0")
                # utterance = utterance.replace('muhammad ali vs. joe frazier ii', 'ali-frazier ii')
                # utterance = utterance.replace('golden globe award for best drama film', 'golden globe award for best motion picture- drama')
                # utterance = utterance.replace('the x-men issue 1', 'the x men 1')
                # utterance = utterance.replace('phillies', 'philadelphia_phillies')
                # utterance = utterance.replace('men _ s singles tennis at the 1896 summer olympics', 'tennis at the 1896 summer olympics mens singles')
                # utterance = utterance.replace('what is the usa money currency code', 'what is the us money currency code')
                utterance = utterance.replace('men _ s', "men's")
                utterance = utterance.replace('usa _ s', "usa's")
                utterance = utterance.replace('gilligan _ s', 'gilligans')
                utterance = utterance.replace('charlie _ s', 'charlies')
                utterance= utterance.replace('magician _ s', 'magicians')
                # utterance = utterance.replace('nyse', 'new york stock exchange inc')
                # utterance = utterance.replace('inc.', 'inc')
                # utterance = utterance.replace('monopoly', 'monopoly boardgame')
                # utterance = utterance.replace('first amendment', 'first amendment to the united states constitution')
                # utterance = utterance.replace('lse', 'london stock exchange')
                # utterance = utterance.replace('borders', 'borders group')
                utterance = utterance.replace('forune', 'fortune')
                utterance = utterance.replace('28 days later', '28 weeks later')
                # utterance = utterance.replace('f. c .', 'fc')
                # utterance = utterance.replace('what blogs are in german', 'what blogs are in german language')
                # utterance = utterance.replace('f. c .', 'fc')
                # utterance = utterance.replace('nathan smith', 'nathan smith 1770')
                # utterance = utterance.replace('what is the symbol for yen', 'what is the symbol for japanese yen')
                # utterance = utterance.replace('roe v. wade', 'roe v wade')

                # add special tokens to utterance and split
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                # tokenize mr
                mr_padding = mr.replace(")", " ) ").replace("(", " ( ").replace(":", " :").replace("!", " ! ").replace(
                    "   ", " ").replace("  ", " ")

                # find values in mr
                pattern = re.compile(r'en\.([a-zA-Z0-9_-]+)')
                res = pattern.findall(mr_padding)
                pattern = re.compile(r'\(\bdate ([0-9][0-9][0-9][0-9])')
                res.extend(pattern.findall(mr))

                mr_padding = mr_padding.replace("en.", "en. ").split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res:
                        if "_" in mr_token:
                            mr_tokens = mr_token.split("_")
                        # elif "-" in mr_token and "".join(mr_token.split("-")) in utterances_tokenized:
                        #     mr_tokens = ["".join(mr_token.split("-"))]
                        else:
                            mr_tokens = [mr_token]

                        pointer = sum(1 for target_token in mr_tokens if target_token in utterances_tokenized) == len(
                            mr_tokens)
                        if pointer:
                            for target_token in mr_tokens:
                                query_pointers.append("@ptr{}".format(utterances_tokenized.index(target_token)))
                        else:
                            output_vocab_lambda.add(mr_token)
                            query_pointers.append(mr_token)
                        # for target_token in mr_tokens:
                        #     # if target_token not in utterances_tokenized and target_token + 's' in utterances_tokenized:
                        #     #     target_token = target_token + 's'
                        #     # if target_token not in utterances_tokenized and target_token[:len(target_token) - 1] + 'ies' \
                        #     #         in utterances_tokenized:
                        #     #     target_token = target_token + 's'
                        #     # there 162 cases here, checked that part of the tokens are able to refer to the mapping in utterance?
                        #     if not target_token in utterances_tokenized:
                        #         # print(raw_utterance)
                        #         # print(mr_padding)
                        #         # print(res)
                        #         # print(target_token)
                        #         # print()
                        #         # print(' 1 ', target_token + '\t' + mr_token + '\t' + raw_utterance)
                        #         # output_vocab_lambda.update(target_token)
                        #         # query_pointers.append(target_token)
                        #         pointer = False
                        #         break
                        #     else:
                        #         # target_token in utterances_tokenized:
                        #         query_pointers.append("@ptr{}".format(utterances_tokenized.index(target_token)))
                    else:
                        # Change fb :computer.computer.introduced to fb : computer . computer . introduced
                        output_vocab_lambda.update(mr_token.replace(":", " : ").replace(".", " . ").split())
                        query_pointers.extend(mr_token.replace(":", " : ").replace(".", " . ").split())
                # add special tokens to query_pointers
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_lambda, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_lambda)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_lambda)):
                f.write(token + '\n')


def read_mlsp(model, output_dir):
    # languages = ['en','zh']
    if MULTILINGUAL:
        languages = ['multilingual']
    languages = ['en']

    tokenizer = get_tokenizer(model)

    CONJ = ["(", ")", ":", ",", "-"]
    PREDS = ["lambda", "exist", "and", "or", "?x", "?y", "count", "argmin", "argmax", "argmore", "argless", "max",
             "min", "isa", "1", "equal", "american_state"]

    for language in languages:
        output_dir_lambda = os.path.join(output_dir, 'mlsp', model.replace('/', '_'), language)
        output_dir_lambda = os.path.join(output_dir_lambda, "lambda")
        if not os.path.exists(output_dir_lambda):
            os.makedirs(output_dir_lambda)

        output_vocab_lambda = set()
        len_list_lambda = []

        with open(os.path.join(output_dir_lambda, "dev.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mlsp/dev.json", language, "lambda")
            for utterance, mr in zip(utterances, mrs):
                utterance_raw = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                mr_original = mr
                mr_padding = mr.split()
                for mr_token in mr_padding:
                    if mr_token in CONJ or mr_token in PREDS or (":" in mr_token and not mr_token.endswith(":")) or (
                            "." in mr_token and not mr_token.endswith(".") and mr_token not in utterance.split()):
                        output_vocab_lambda.update(
                            mr_token.replace(":", " : ").replace(".", " . ").replace("?", " ? ").split())
                    else:
                        if "_" in mr_token and mr_token not in utterance.split() and " ".join(
                                mr_token.split("_")) in utterance:
                            mr_padding[mr_padding.index(mr_token)] = " ".join(mr_token.split("_"))
                            mr_token = " ".join(mr_token.split("_"))

                        tokens = mr_token.split()
                        pointers = []
                        for token in tokens:
                            if token not in utterances_tokenized:
                                print(' 1 ', token, mr_token, utterances_tokenized)
                                pointers.append(token)
                            else:
                                pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                f.write(
                    '\t'.join([utterance_raw, utterance, " ".join(mr_padding).replace(":", " : ").replace("?", " ? ")
                              .replace(".", " . ").replace("  ", " ").replace("   ", " ")]) + '\n')

        with open(os.path.join(output_dir_lambda, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mlsp/train.json", language, "lambda")
            for utterance, mr in zip(utterances, mrs):
                utterance_raw = utterance
                # utterance = utterance.replace("which is second chemical element in periodic table", "which is 2 chemical element in periodic table")
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                mr_original = mr
                mr_padding = mr.split()
                for mr_token in mr_padding:
                    if mr_token in CONJ or mr_token in PREDS or (":" in mr_token and not mr_token.endswith(":")) or (
                            "." in mr_token and not mr_token.endswith(".") and mr_token not in utterance.split()):
                        output_vocab_lambda.update(
                            mr_token.replace(":", " : ").replace("?", " ? ").replace(".", " . ").split())
                    else:
                        if "_" in mr_token and mr_token not in utterance.split() and " ".join(
                                mr_token.split("_")) in utterance:
                            mr_padding[mr_padding.index(mr_token)] = " ".join(mr_token.split("_"))
                            mr_token = " ".join(mr_token.split("_"))
                        tokens = mr_token.split()
                        pointers = []
                        for token in tokens:
                            if token not in utterances_tokenized:
                                print(' 1 ', token, mr_token, utterances_tokenized)
                                pointers.append(token)
                            else:
                                pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
                f.write(
                    '\t'.join([utterance_raw, utterance, " ".join(mr_padding).replace(":", " : ").replace("?", " ? ")
                              .replace(".", " . ").replace("  ", " ").replace("   ", " ")]) + '\n')

        with open(os.path.join(output_dir_lambda, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_lambda)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_lambda)):
                f.write(token + '\n')


def read_mnlmaps(model, output_dir):
    languages = ['en', 'de']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['multilingual']
    # languages = ['en']

    tokenizer = get_tokenizer(model)

    CONJ = ["(", ")", ":", ",", "-"]
    PREDS = ["city", "of", "*", "only", "yes", "no", "christian", "parking", "08221000", "motorway", "restaurant"]

    for language in languages:
        output_dir_funql = os.path.join(output_dir, 'mnlmaps', model.replace('/', '_'), language)
        output_dir_funql = os.path.join(output_dir_funql, "funql")
        if not os.path.exists(output_dir_funql):
            os.makedirs(output_dir_funql)

        output_vocab_funql = set()
        len_list_funql = []

        with open(os.path.join(output_dir_funql, "test.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mnlmaps/test.json", language, "funql")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = utterance.lower().replace("?", " ?").replace(
                    "!", " ! ").replace(",", " , ").replace("'", " ' ").replace(".", " . ").replace("  ", " ").strip()
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_funql.append(len(utterances_tokenized))

                # utterance = utterance.replace("Nîmes", "Osterode")
                # utterances_tokenized = tokenizer.tokenize(utterance)

                mr_original = mr

                # if "bathrooms" in utterance:
                #     mr = mr.replace("toilets", "bathrooms")
                # elif "bathroom" in utterance:
                #     mr = mr.replace("toilets", "bathroom")
                # if "town halls" in utterance:
                #     mr = mr.replace("townhall", "town halls")
                # elif "town hall" in utterance:
                #     mr = mr.replace("townhall", "town hall")
                # if "airports" in utterance:
                #     mr = mr.replace("aerodrome", "airports")
                # elif "airport" in utterance:
                #     mr = mr.replace("aerodrome", "airport")
                # if "mountains" in utterance:
                #     mr = mr.replace("peak", "mountains")
                # if "Vauxhall" in utterance:
                #     mr = mr.replace("car", "Vauxhall")
                # if "skateboarding" in utterance:
                #     mr = mr.replace("skateboard", "skateboarding")
                # if "go shopping" in utterance:
                #     mr = mr.replace("supermarket", "go shopping")
                # if "mountain" in utterance:
                #     mr = mr.replace("peak", "mountain")

                # temp = re.findall(r"(A \d+)", mr)
                # if temp:
                #     temp = temp[0]
                #     print(mr)
                #     mr = mr.replace(temp, "".join(temp.split()))
                #     print(mr)
                #     exit()

                # if utterance == "Which museum is closest to Edinburgh Waverley?":
                #     mr = mr.replace("station", "Edinburgh Waverley")
                # elif utterance == "What are the primary schools of Bielefeld called?":
                #     mr = mr.replace("Grundschule", "primary schools")
                # elif utterance == "How many Volkshochschule are there in Germany?":
                #     mr = mr.replace("school", "Volkshochschule")
                # elif utterance == "What are the names of butchers in Edinburgh?":
                #     mr = mr.replace("bakery", "butcher")
                # elif utterance == "Where is the Fernsehturm Heidelberg?":
                #     mr = mr.replace("communication", "Fernsehturm Heidelberg")
                # elif utterance == "Where are hiking maps close to a car park in Edinburgh?":
                #     mr = mr.replace("peak", "car park")
                # elif utterance == "Give me cafes in Edinburgh that have a car park close by.":
                #     mr = mr.replace("cinema", "car park")
                # elif utterance == "Can you tell me the location of a work of art in Edinburgh?":
                #     mr = mr.replace("artwork", "work of art")
                # elif utterance == "Are there more than 5 arts centres in Paris?":
                #     mr = mr.replace("20", "5")
                # elif utterance == "How many schools are close to Neuenheim?":
                #     mr = mr.replace("Heidelberg", "Neuenheim")

                # if "Gare du Nord" in mr:
                #     mr = mr.replace("station", "Gare du Nord")
                # if "Heidelberg Hbf" in mr:
                #     mr = mr.replace("station", "Heidelberg Hbf")
                # if "2" in mr and not "2" in utterance:
                #     if "several" in utterance:
                #         mr = mr.replace("2", "several")
                #     elif "more than one" in utterance:
                #         mr = mr.replace("2", "more than one")
                # if "Tour Eiffel" in mr and "Eiffel Tower" in utterance:
                #     mr = mr.replace("Tour Eiffel", "Eiffel Tower")
                # if "Eiffel Tower" in mr and "Tour Eiffel" in utterance:
                #     mr = mr.replace("Eiffel Tower", "Tour Eiffel")

                # utterance = utterance.lower()
                mr = mr.lower()

                pattern_post = re.compile(r"keyval\(\'[0-9a-zA-Z\_\-:]+\',\'([0-9 a-zA-Z\_\-:]+)\'\)")
                res_post = re.findall(pattern_post, mr)
                pattern_pre = re.compile(r"keyval\(\'([0-9 a-zA-Z\_\-:]+)\',")
                res_pre = re.findall(pattern_pre, mr)
                res = []
                for token in res_post:
                    if re.search(r"[a-zA-Z]\d+", token):
                        res.append(token)
                    else:
                        res.extend(token.split())
                        # print(mr)
                # print(res_pre)
                # print(res)
                # print()
                for token in res_pre:
                    output_vocab_funql.add(token)

                # mr_padding = mr.replace("france", "paris").replace("hbf", "hauptbahnhof").replace("bbq", "barbecue").replace(
                #     "deutschland", "germany").replace("bayern", "bavaria").replace("stationery", "stationary").replace(
                #     "copyshop", "copy shops").replace("fuel", "petrol").replace("soccer", "football").replace(
                #     "heidelberger schloss", "heidelberg castle").replace("united kingdom", "the uk").replace("ludwigshafen am rhein", "kreuztal").replace(
                #     "stolperstein", "stolpersteine").replace("30000", "30km").replace("50000", "50km").replace(
                #     "5000", "5km").replace("10000", "10km").replace("muslim", "mosques").replace("2000", "2km").replace(
                #     "(", " ( ").replace(")", " ) ").replace(",", " , ").replace("?", " ? ").replace("'", " ' ").replace(
                #     "   ", " ").replace("  ", " ").split()
                mr_padding = mr.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace("?", " ? ").replace(
                    "'", " ' ").replace("   ", " ").replace("  ", " ").split()

                query_pointers = []
                for mr_token in mr_padding:
                    if (
                            mr_token in res and not mr_token in PREDS and not mr_token in res_pre and not "_" in mr_token) or (
                            mr_token.isdigit() and mr_token in utterance.split()) or (re.search(r"\d", mr_token) and (
                            mr_token.endswith("e") or mr_token.endswith("er"))):
                        # check the issue of single/plural form
                        # if not mr_token in utterance.split() and mr_token + "s" in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token + "s"
                        #     mr_token = mr_token + "s"
                        # elif not mr_token in utterance.split() and mr_token[:-1] + "ies" in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-1] + "ies"
                        #     mr_token = mr_token[:-1] + "ies"
                        # elif not mr_token in utterance.split() and mr_token + "es" in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token + "es"
                        #     mr_token = mr_token + "es"
                        # elif not mr_token in utterance and mr_token.endswith("s") and mr_token[:-1] in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-1]
                        #     mr_token = mr_token[:-1]
                        # elif not mr_token in utterance and mr_token.endswith("es") and mr_token[:-2] in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-2]
                        #     mr_token = mr_token[:-2]

                        # check the issue of ordinal numbers
                        # if mr_token.endswith("e") and mr_token[:-1] + "th" in utterance.split():
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-1] + "th"
                        #     mr_token = mr_token[:-1] + "th"
                        # elif mr_token.endswith("er") and mr_token[:-2] + "st" in utterance.split():
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-2] + "st"
                        #     mr_token = mr_token[:-2] + "st"
                        # elif mr_token.endswith("er") and "first" in utterance.split():
                        #     mr_padding[mr_padding.index(mr_token)] = "first"
                        #     mr_token = "first"

                        # if mr_token in utterance:
                        #     tokens = tokenizer.tokenize(mr_token)
                        #     pointers = []
                        #     for token in tokens:
                        #         pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        #     mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)

                        if mr_token in utterances_tokenized:
                            pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                            query_pointers.append(pointer)
                        else:
                            output_vocab_funql.add(mr_token)
                            query_pointers.append(mr_token)
                    # elif not mr_token in utterance:
                    else:
                        output_vocab_funql.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')
                # print(utterance)
                # print(mr_padding)
                # print(query_pointers)
                # print()

        with open(os.path.join(output_dir_funql, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mnlmaps/train.json", language, "funql")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = utterance.lower().replace("?", " ?").replace(
                    "!", " ! ").replace(",", " , ").replace("'", " ' ").replace(".", " . ").replace("  ", " ")
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_funql.append(len(utterances_tokenized))

                # utterance = utterance.replace("Nîmes", "Osterode")
                # utterances_tokenized = tokenizer.tokenize(utterance)

                mr_original = mr

                # if "bathrooms" in utterance:
                #     mr = mr.replace("toilets", "bathrooms")
                # elif "bathroom" in utterance:
                #     mr = mr.replace("toilets", "bathroom")
                # if "town halls" in utterance:
                #     mr = mr.replace("townhall", "town halls")
                # elif "town hall" in utterance:
                #     mr = mr.replace("townhall", "town hall")
                # if "airports" in utterance:
                #     mr = mr.replace("aerodrome", "airports")
                # elif "airport" in utterance:
                #     mr = mr.replace("aerodrome", "airport")
                # if "mountains" in utterance:
                #     mr = mr.replace("peak", "mountains")
                # if "Vauxhall" in utterance:
                #     mr = mr.replace("car", "Vauxhall")
                # if "skateboarding" in utterance:
                #     mr = mr.replace("skateboard", "skateboarding")
                # if "go shopping" in utterance:
                #     mr = mr.replace("supermarket", "go shopping")
                # if "mountain" in utterance:
                #     mr = mr.replace("peak", "mountain")
                # if "all the time" in utterance:
                #     mr = mr.replace("24/7", "all the time")
                # if "Edinburgh Waverley" in utterance:
                #     mr = mr.replace("station", "Edinburgh Waverley")

                # temp = re.findall(r"(A \d+)", mr)
                # if temp:
                #     temp = temp[0]
                #     mr = mr.replace(temp, "".join(temp.split()))

                # if utterance == "How many Stolpersteine can be found in the vicinity of Heidelberg?":
                #     mr = mr.replace("Heidelberger Schloss", "Heidelberg")
                # elif utterance == "Where are hiking maps in Heidelberg close to a car park?":
                #     mr = mr.replace("peak", "car park")
                # elif utterance == "Where are amenities close to Heidelberg-Altstadt train station?":
                #     mr = mr.replace("'Heidelberg'", "'Heidelberg-Altstadt'")
                #     # print("#$%", mr)
                # elif utterance == "How many Stolpersteine can be found east of Heidelberg?":
                #     mr = mr.replace("Heidelberger Schloss", "Heidelberg")
                # elif utterance == "What is a location in Paris where I can row?" or utterance == "What is a location in Heidelberg where I can row?":
                #     mr = mr.replace("rowing", "row")
                # elif utterance == "Can I walk from the Heidelberg castle to the main station?":
                #     mr = mr.replace("Heidelberg Hbf", "main station")
                # elif utterance == "Where is the closest Renault dealer ship from Paris?":
                #     mr = mr.replace("car", "Renault")
                # elif utterance == "Can you give me the street name in Edinburgh?":
                #     mr = mr.replace("bakery", "name")
                # elif utterance == "How far apart are the Heidelberg Castle and the Heidelberg main station?" or utterance == "What is the closest Italian or Indian restaurant from the main station in Heidelberg?":
                #     mr = mr.replace("Heidelberg Hbf", "Heidelberg main station")
                # elif utterance == "Where is the closest Mercedes dealer ship from Heidelberg?":
                #     mr = mr.replace("car", "Mercedes")
                # elif utterance == "How many schools are close to the 14th Arrondissement?":
                #     mr = mr.replace("Paris", "paris").replace("paris", "the 14th Arrondissement").replace("France", "the 14th Arrondissement")
                # elif utterance == "How many schools are close to Restalrig?":
                #     mr = mr.replace("City of Edinburgh", "Restalrig")
                # elif utterance == "Who operates the Fernsehturm Heidelberg?":
                #     mr = mr.replace("communication", "Fernsehturm Heidelberg")
                # elif utterance == "Give me cafes in Paris that have a car park close by." or utterance == "Give me cafes in Heidelberg that have a car park close by.":
                #     mr = mr.replace("cinema", "car park")
                # elif utterance == "What kind of amenities are close to Heidelberg-Altstadt train station?":
                #     mr = mr.replace("'Heidelberg'", "'Heidelberg-Altstadt'")
                # elif utterance == "What are the opening times of the Tesco that is closest to the Deaconess Garden in Edinburgh?":
                #     mr = mr.replace("Carpet Lane", "Deaconess Garden")
                # elif utterance == "How many emergency sirens are there in Witten?" or utterance == "Where in Baden-Baden are emergency sirens?":
                #     mr = mr.replace("phone", "sirens")
                # elif utterance == "What is the closest fast food restaurant from the main station in Heidelberg?":
                #     mr = mr.replace("Heidelberg Hbf", "the main station in Heidelberg")

                # if "Gare du Nord" in mr:
                #     mr = mr.replace("station", "Gare du Nord")
                # if "Heidelberg Hbf" in mr:
                #     mr = mr.replace("station", "Heidelberg Hbf")
                # if "2" in mr and not "2" in utterance:
                #     if "several" in utterance:
                #         mr = mr.replace("2", "several")
                #     elif "more than one" in utterance:
                #         mr = mr.replace("2", "more than one")
                # if "'artwork'" in mr and "work of art" in utterance:
                #     mr = mr.replace("'artwork'", "''work of art'")
                # if "'artwork'" in mr and "works of art" in utterance:
                #     mr = mr.replace("'artwork'", "''works of art'")
                # if "Tour Eiffel" in mr and "Eiffel Tower" in utterance:
                #     mr = mr.replace("Tour Eiffel", "Eiffel Tower")
                # if "Eiffel Tower" in mr and "Tour Eiffel" in utterance:
                #     mr = mr.replace("Eiffel Tower", "Tour Eiffel")

                # utterance = utterance.lower().replace("stolpersteine", "stolperstein").replace("?", " ?").replace(
                #     "!", " ! ").replace(",", " , ").replace("'", " ' ").replace(".", " . ").replace("  ", " ")
                # mr = mr.lower().replace("france", "paris").replace("hbf", "hauptbahnhof").replace(
                #     "bbq", "barbecue").replace("deutschland", "germany").replace("bayern", "bavaria").replace(
                #     "stationery", "stationary").replace("copyshop", "copy shops").replace("fuel", "petrol").replace(
                #     "soccer", "football").replace("sachsen", "saxony").replace(
                #     "pizza", "pizzeria").replace("30000", "30km").replace("50000", "50km").replace(
                #     "5000", "5km").replace("10000", "10km").replace("muslim", "mosques").replace("frankfurt am main", "frankfurt").replace(
                #     "heidelberger schloss", "heidelberg castle").replace("minefield", "mine fields").replace("meslay", "lauriston").replace("2000", "2km")
                mr = mr.lower()

                pattern_post = re.compile(r"keyval\(\'[0-9a-zA-Z\_\-:]+\',\'([0-9 a-zA-Z\_\-:]+)\'\)")
                res_post = re.findall(pattern_post, mr)
                pattern_pre = re.compile(r"keyval\(\'([0-9 a-zA-Z\_\-:]+)\',")
                res_pre = re.findall(pattern_pre, mr)
                res = []
                for token in res_post:
                    if re.search(r"[a-zA-Z]\d+", token):
                        res.append(token)
                    else:
                        res.extend(token.split())

                for token in res_pre:
                    output_vocab_funql.add(token)

                mr_padding = mr.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace("?", " ? ").replace(
                    "'", " ' ").replace("   ", " ").replace("  ", " ").split()

                query_pointers = []
                for mr_token in mr_padding:
                    if (
                            mr_token in res and not mr_token in PREDS and not mr_token in res_pre and not "_" in mr_token) or (
                            mr_token.isdigit() and mr_token in utterance.split()) or (re.search(r"\d", mr_token) and (
                            mr_token.endswith("e") or mr_token.endswith("er"))):
                        # if not mr_token in utterance.split() and mr_token + "s" in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token + "s"
                        #     mr_token = mr_token + "s"
                        # elif not mr_token in utterance.split() and mr_token[:-1] + "ies" in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-1] + "ies"
                        #     mr_token = mr_token[:-1] + "ies"
                        # elif not mr_token in utterance.split() and mr_token + "es" in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token + "es"
                        #     mr_token = mr_token + "es"
                        # elif not mr_token in utterance and mr_token.endswith("s") and mr_token[:-1] in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-1]
                        #     mr_token = mr_token[:-1]
                        # elif not mr_token in utterance and mr_token.endswith("es") and mr_token[:-2] in utterance:
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-2]
                        #     mr_token = mr_token[:-2]

                        # if mr_token.endswith("e") and mr_token[:-1] + "th" in utterance.split():
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-1] + "th"
                        #     mr_token = mr_token[:-1] + "th"
                        # elif mr_token.endswith("e") and mr_token[:-1] + "nd" in utterance.split():
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-1] + "nd"
                        #     mr_token = mr_token[:-1] + "nd"
                        # elif mr_token.endswith("er") and mr_token[:-2] + "st" in utterance.split():
                        #     mr_padding[mr_padding.index(mr_token)] = mr_token[:-2] + "st"
                        #     mr_token = mr_token[:-2] + "st"
                        # elif mr_token.endswith("er") and "first" in utterance.split():
                        #     mr_padding[mr_padding.index(mr_token)] = "first"
                        #     mr_token = "first"

                        # if mr_token in utterance.split():
                        #     tokens = tokenizer.tokenize(mr_token)
                        #     pointers = []
                        #     for token in tokens:
                        #         pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        #     mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)

                        if mr_token in utterances_tokenized:
                            pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                            query_pointers.append(pointer)
                        else:
                            output_vocab_funql.add(mr_token)
                            query_pointers.append(mr_token)
                    # elif not mr_token in utterance:
                    else:
                        output_vocab_funql.add(mr_token)
                        query_pointers.append(mr_token)
                # f.write('\t'.join([utterance_raw, utterance, " ".join(mr_padding)]) + '\n')
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_funql, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_funql)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_funql)):
                f.write(token + '\n')


def read_movernight(model, output_dir):
    languages = ['en', 'de', 'zh']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['multilingual']
    # languages = ['en']

    tokenizer = get_tokenizer(model)

    CONJ = ["(", ")", ":", ",", "-"]
    PREDS = ["cuisine", "restaurant", "neighborhood"]

    for language in languages:
        output_dir_lambda = os.path.join(output_dir, 'movernight', model.replace('/', '_'), language)
        output_dir_lambda = os.path.join(output_dir_lambda, "lambda")
        if not os.path.exists(output_dir_lambda):
            os.makedirs(output_dir_lambda)

        output_vocab_lambda = set()
        len_list_lambda = []

        with open(os.path.join(output_dir_lambda, "dev.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}movernight/dev.json", language, "lambda")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = utterance.lower()
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                mr_original = mr

                pattern = re.compile(r'en\.([a-zA-Z0-9_.-]+)\b')
                res_list = re.findall(pattern, mr)
                res = []
                for token in res_list:
                    if len(token.split(".")) > 1:
                        # res.extend(token.split(".")[-1].split("_"))
                        # mr = mr.replace(token.split(".")[-1], " ".join(token.split(".")[-1].split("_")))
                        res.append(token.split(".")[-1])
                # if res:
                #     print(mr_original)
                #     print(res)
                #     print(mr)
                #     exit()
                # continue

                mr_padding = mr.replace(".", " . ").replace("!", " ! ").split()

                # this_utterance = utterance.lower().replace("flat", "apartment").replace("condominium", "condo").replace(
                #     "condomonium", "condo").replace("bricks", "block").replace("brick", "block").replace(
                #     "mickinsey", "mckinsey").replace("beijing", "bejing").replace("men", "male").replace(
                #     "181cm", "180").replace("180cm", "180")

                # if utterance == "what2 dollar sign restaurant has outdoor seating":
                #     this_utterance = "what 2 dollar sign restaurant has outdoor seating"

                # if "alice" in mr and not "alice" in this_utterance and "female" in this_utterance:
                #     this_utterance = this_utterance.replace("females", "alice").replace("female", "alice")
                # temp = re.findall(r'\d[a-z][a-z]\b', this_utterance)
                # if temp:
                #     this_utterance = this_utterance.replace("".join(temp), "".join(temp)[:-2] + " " + "".join(temp)[-2:])
                # temp = re.findall(r'(\d[ ]*pm)', this_utterance)
                # if temp:
                #     this_utterance = this_utterance.replace("".join(temp), str(int("".join(temp)[:-2].strip()) + 12))
                # this_utterance_tokenized = tokenizer.tokenize(this_utterance)

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res or (mr_token.isdigit() and mr_token != -1):
                        # mr_token_raw = mr_token
                        # if mr_token == "lakers" and not "laker" in this_utterance and "kobe bryant" in this_utterance:
                        #     mr_token = "kobe bryant"
                        # if mr_token == "180" and "6 feet" in this_utterance:
                        #     mr_token = "6 feet"
                        # if mr_token[:-1].isalpha() and mr_token[-1:].isdigit():
                        #     if mr_token[:-1] in this_utterance.split():
                        #         mr_token = mr_token[:-1] + " " + mr_token[-1:]
                        #     elif (not mr_token[:-1] in this_utterance.split()) and mr_token[:-1] + "s" in this_utterance.split():
                        #         mr_token = mr_token[:-1] + "s " + mr_token[-1:]
                        #     if not mr_token[-1:] in this_utterance.split():
                        #         mr_token = mr_token[:-1]
                        # if not mr_token in this_utterance.split() and mr_token + "s" in this_utterance.split():
                        #     mr_token += "s"
                        # elif not mr_token in this_utterance.split() and mr_token[:-1] in this_utterance.split():
                        #     mr_token = mr_token[:-1]

                        # if mr_token in this_utterance.split():
                        #     tokens = tokenizer.tokenize(mr_token)
                        #     pointers = []
                        #     for token in tokens:
                        #         pointers.append("@ptr{}".format(this_utterance_tokenized.index(token)))
                        #     mr_padding[mr_padding.index(mr_token_raw)] = " ".join(pointers)

                        if mr_token in utterances_tokenized:
                            pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                            query_pointers.append(pointer)
                        else:
                            output_vocab_lambda.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_lambda.add(mr_token)
                        query_pointers.append(mr_token)
                # f.write('\t'.join([utterance_raw, utterance, " ".join(mr_padding)]) + '\n')
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_lambda, "test.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}movernight/test.json", language, "lambda")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = utterance.lower()
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                mr_original = mr

                pattern = re.compile(r'en\.([a-zA-Z0-9_.-]+)\b')
                res_list = re.findall(pattern, mr)
                res = []
                for token in res_list:
                    if len(token.split(".")) > 1:
                        # res.extend(token.split(".")[-1].split("_"))
                        # mr = mr.replace(token.split(".")[-1], " ".join(token.split(".")[-1].split("_")))
                        res.append(token.split(".")[-1])

                mr_padding = mr.replace(".", " . ").replace("!", " ! ").split()

                # this_utterance = utterance.lower().replace("flat", "apartment").replace("condominium", "condo").replace(
                #     "condomonium", "condo").replace("bricks", "block").replace("brick", "block").replace(
                #     "mickinsey", "mckinsey").replace("beijing", "bejing").replace("men", "male").replace(
                #     "181cm", "180").replace("180cm", "180")

                # if utterance == "what2 dollar sign restaurant has outdoor seating":
                #     this_utterance = "what 2 dollar sign restaurant has outdoor seating"

                # if "alice" in mr and not "alice" in this_utterance and "female" in this_utterance:
                #     this_utterance = this_utterance.replace("females", "alice").replace("female", "alice")
                # temp = re.findall(r'\d[a-z][a-z]\b', this_utterance)
                # if temp:
                #     this_utterance = this_utterance.replace("".join(temp), "".join(temp)[:-2] + " " + "".join(temp)[-2:])
                # temp = re.findall(r'(\d[ ]*pm)', this_utterance)
                # if temp:
                #     this_utterance = this_utterance.replace("".join(temp), str(int("".join(temp)[:-2].strip()) + 12))
                # this_utterance_tokenized = tokenizer.tokenize(this_utterance)

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res or (mr_token.isdigit() and mr_token != -1):
                        # mr_token_raw = mr_token
                        # if mr_token == "lakers" and not "laker" in this_utterance and "kobe bryant" in this_utterance:
                        #     mr_token = "kobe bryant"
                        # if mr_token == "180" and "6 feet" in this_utterance:
                        #     mr_token = "6 feet"
                        # if mr_token[:-1].isalpha() and mr_token[-1:].isdigit():
                        #     if mr_token[:-1] in this_utterance.split():
                        #         mr_token = mr_token[:-1] + " " + mr_token[-1:]
                        #     elif (not mr_token[:-1] in this_utterance.split()) and mr_token[:-1] + "s" in this_utterance.split():
                        #         mr_token = mr_token[:-1] + "s " + mr_token[-1:]
                        #     if not mr_token[-1:] in this_utterance.split():
                        #         mr_token = mr_token[:-1]
                        # if not mr_token in this_utterance.split() and mr_token + "s" in this_utterance.split():
                        #     mr_token += "s"
                        # elif not mr_token in this_utterance.split() and mr_token[:-1] in this_utterance.split():
                        #     mr_token = mr_token[:-1]

                        # if mr_token in this_utterance.split():
                        #     tokens = tokenizer.tokenize(mr_token)
                        #     pointers = []
                        #     for token in tokens:
                        #         pointers.append("@ptr{}".format(this_utterance_tokenized.index(token)))
                        #     mr_padding[mr_padding.index(mr_token_raw)] = " ".join(pointers)

                        if mr_token in utterances_tokenized:
                            pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                            query_pointers.append(pointer)
                        else:
                            output_vocab_lambda.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_lambda.add(mr_token)
                        query_pointers.append(mr_token)
                # f.write('\t'.join([utterance_raw, utterance, " ".join(mr_padding)]) + '\n')
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_lambda, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}movernight/train.json", language, "lambda")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = utterance.lower()
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_lambda.append(len(utterances_tokenized))

                mr_original = mr

                pattern = re.compile(r'en\.([a-zA-Z0-9_.-]+)\b')
                res_list = re.findall(pattern, mr)
                res = []
                for token in res_list:
                    if len(token.split(".")) > 1:
                        # res.extend(token.split(".")[-1].split("_"))
                        # mr = mr.replace(token.split(".")[-1], " ".join(token.split(".")[-1].split("_")))
                        res.append(token.split(".")[-1])

                mr_padding = mr.replace(".", " . ").replace("!", " ! ").split()

                # this_utterance = utterance.lower().replace("flat", "apartment").replace("condominium", "condo").replace(
                #     "condomonium", "condo").replace("bricks", "block").replace("brick", "block").replace(
                #     "mickinsey", "mckinsey").replace("beijing", "bejing").replace("men", "male").replace(
                #     "181cm", "180").replace("180cm", "180")

                # if utterance == "what2 dollar sign restaurant has outdoor seating":
                #     this_utterance = "what 2 dollar sign restaurant has outdoor seating"

                # if "alice" in mr and not "alice" in this_utterance and "female" in this_utterance:
                #     this_utterance = this_utterance.replace("females", "alice").replace("female", "alice")
                # temp = re.findall(r'\d[a-z][a-z]\b', this_utterance)
                # if temp:
                #     this_utterance = this_utterance.replace("".join(temp), "".join(temp)[:-2] + " " + "".join(temp)[-2:])
                # temp = re.findall(r'(\d[ ]*pm)', this_utterance)
                # if temp:
                #     this_utterance = this_utterance.replace("".join(temp), str(int("".join(temp)[:-2].strip()) + 12))
                # this_utterance_tokenized = tokenizer.tokenize(this_utterance)

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res or (mr_token.isdigit() and mr_token != -1):
                        # mr_token_raw = mr_token
                        # if mr_token == "lakers" and not "laker" in this_utterance and "kobe bryant" in this_utterance:
                        #     mr_token = "kobe bryant"
                        # if mr_token == "180" and "6 feet" in this_utterance:
                        #     mr_token = "6 feet"
                        # if mr_token[:-1].isalpha() and mr_token[-1:].isdigit():
                        #     if mr_token[:-1] in this_utterance.split():
                        #         mr_token = mr_token[:-1] + " " + mr_token[-1:]
                        #     elif (not mr_token[:-1] in this_utterance.split()) and mr_token[:-1] + "s" in this_utterance.split():
                        #         mr_token = mr_token[:-1] + "s " + mr_token[-1:]
                        #     if not mr_token[-1:] in this_utterance.split():
                        #         mr_token = mr_token[:-1]
                        # if not mr_token in this_utterance.split() and mr_token + "s" in this_utterance.split():
                        #     mr_token += "s"
                        # elif not mr_token in this_utterance.split() and mr_token[:-1] in this_utterance.split():
                        #     mr_token = mr_token[:-1]

                        # if mr_token in this_utterance.split():
                        #     tokens = tokenizer.tokenize(mr_token)
                        #     pointers = []
                        #     for token in tokens:
                        #         pointers.append("@ptr{}".format(this_utterance_tokenized.index(token)))
                        #     mr_padding[mr_padding.index(mr_token_raw)] = " ".join(pointers)

                        if mr_token in utterances_tokenized:
                            pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                            query_pointers.append(pointer)
                        else:
                            output_vocab_lambda.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_lambda.add(mr_token)
                        query_pointers.append(mr_token)
                # f.write('\t'.join([utterance_raw, utterance, " ".join(mr_padding)]) + '\n')
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_lambda, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_lambda)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_lambda)):
                f.write(token + '\n')


def read_mschema2qa(model, output_dir):
    languages = ['en', 'ar', 'de', 'es', 'fa', 'fi', 'it', 'ja', 'pl', 'tr', 'zh']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['multilingual']
    # languages = ['en']

    tokenizer = get_tokenizer(model)

    for language in languages:
        output_dir_thingtalk = os.path.join(output_dir, 'mschema2qa', model.replace('/', '_'), language)
        output_dir_thingtalk = os.path.join(output_dir_thingtalk, "thingtalk")
        if not os.path.exists(output_dir_thingtalk):
            os.makedirs(output_dir_thingtalk)
        print(output_dir_thingtalk)
        # output_dir_thingtalk_no_value = os.path.join(output_dir, "thingtalk_no_value")
        # if not os.path.exists(output_dir_thingtalk_no_value):
        #     os.makedirs(output_dir_thingtalk_no_value)

        output_vocab_thingtalk = set()
        len_list_thingtalk = []
        # output_vocab_thingtalk_no_value = set()
        # len_list_thingtalk_no_value = []

        with open(os.path.join(output_dir_thingtalk, "test.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mschema2qa/test.json", language, "thingtalk")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_thingtalk.append(len(utterances_tokenized))

                pattern = re.compile(r'.+?"(.+?)"')
                res_temp = re.findall(pattern, mr)
                res_final = []
                for i in range(len(res_temp)):
                    res_final.extend(res_temp[i].split())

                mr_original = mr
                mr_padding = mr.replace("@", " @ ").replace(":", " : ").replace("^^", " ^^ ").replace("(",
                                                                                                      " ( ").replace(
                    ")", " ) ").split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res_final:
                        # index = mr_padding.index(mr_token)
                        # if '"' in mr_padding[:index] and '"' in mr_padding[index:]:
                        #     tokens = mr_token.split()
                        #     pointers = []
                        #     for token in tokens:
                        #         pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        #     mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)

                        if mr_token in utterances_tokenized:
                            pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                            query_pointers.append(pointer)
                        else:
                            output_vocab_thingtalk.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_thingtalk.add(mr_token)
                        query_pointers.append(mr_token)
                # f.write('\t'.join([utterance_raw, utterance, " ".join(mr_padding)]) + '\n')
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')
        # continue
        # print('train')
        # exit()
        with open(os.path.join(output_dir_thingtalk, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mschema2qa/train.json", language, "thingtalk")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list_thingtalk.append(len(utterances_tokenized))

                pattern = re.compile(r'.+?"(.+?)"')
                res_temp = re.findall(pattern, mr)
                res_final = []
                for i in range(len(res_temp)):
                    res_final.extend(res_temp[i].split())

                mr_original = mr
                mr_padding = mr.replace("@", " @ ").replace(":", " : ").replace("^^", " ^^ ").replace("(",
                                                                                                      " ( ").replace(
                    ")", " ) ").split()

                query_pointers = []
                for mr_token in mr_padding:
                    if mr_token in res_final:
                        # index = mr_padding.index(mr_token)
                        # if '"' in mr_padding[:index] and '"' in mr_padding[index:]:
                        #     tokens = mr_token.split()
                        #     pointers = []
                        #     for token in tokens:
                        #         pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
                        #     mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)

                        if mr_token in utterances_tokenized:
                            pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                            query_pointers.append(pointer)
                        else:
                            output_vocab_thingtalk.add(mr_token)
                            query_pointers.append(mr_token)
                    else:
                        output_vocab_thingtalk.add(mr_token)
                        query_pointers.append(mr_token)
                # f.write('\t'.join([utterance_raw, utterance, " ".join(mr_padding)]) + '\n')
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        # with open(os.path.join(output_dir_thingtalk_no_value, "test.tsv"), "w") as f:
        #     utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mschema2qa/test.json", "en", "thingtalk_no_value")
        #     for utterance, mr in zip(utterances, mrs):
        #         utterance_raw = utterance
        #         mr_original = mr
        #         utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
        #         utterances_tokenized = utterance.split()
        #         len_list_thingtalk_no_value.append(len(utterances_tokenized))

        #         pattern = re.compile(r'.+?"(.+?)"')
        #         res_temp = re.findall(pattern, mr)
        #         res_final = []

        #         for i in range(len(res_temp)):
        #             res_final.extend(res_temp[i].split())

        #         to_remove = [token for token in res_final if 'value' in token]
        #         res_final = [token for token in res_final if token not in to_remove]

        #         mr_padding = mr.replace("@", " @ ").replace(":", " : ").replace("^^", " ^^ ").replace("(", " ( ").replace(")", " ) ").split()

        #         for mr_token in mr_padding:
        #             if mr_token in res_final:
        #                 index = mr_padding.index(mr_token)
        #                 if '"' in mr_padding[:index] and '"' in mr_padding[index:]:
        #                     tokens = mr_token.split()
        #                     pointers = []
        #                     for token in tokens:
        #                         pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
        #                     mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
        #             else:
        #                 output_vocab_thingtalk_no_value.add(mr_token)

        #         f.write('\t'.join([utterance_raw, utterance, " ".join(mr_padding)]) + '\n')

        # with open(os.path.join(output_dir_thingtalk_no_value, "train.tsv"), "w") as f:
        #     utterances, mrs = get_utterances_mrs(f"{DATASET_PATH}mschema2qa/train.json", "en", "thingtalk_no_value")
        #     for utterance, mr in zip(utterances, mrs):
        #         utterance_raw = utterance
        #         mr_original = mr
        #         utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
        #         utterances_tokenized = utterance.split()
        #         len_list_thingtalk_no_value.append(len(utterances_tokenized))

        #         pattern = re.compile(r'.+?"(.+?)"')
        #         res_temp = re.findall(pattern, mr)
        #         res_final = []

        #         for i in range(len(res_temp)):
        #             res_final.extend(res_temp[i].split())

        #         to_remove = [token for token in res_final if 'value' in token]
        #         res_final = [token for token in res_final if token not in to_remove]

        #         mr_padding = mr.replace("@", " @ ").replace(":", " : ").replace("^^", " ^^ ").replace("(", " ( ").replace(")", " ) ").split()

        #         for mr_token in mr_padding:
        #             if mr_token in res_final:
        #                 index = mr_padding.index(mr_token)
        #                 if '"' in mr_padding[:index] and '"' in mr_padding[index:]:
        #                     tokens = mr_token.split()
        #                     pointers = []
        #                     for token in tokens:
        #                         pointers.append("@ptr{}".format(utterances_tokenized.index(token)))
        #                     mr_padding[mr_padding.index(mr_token)] = " ".join(pointers)
        #             else:
        #                 output_vocab_thingtalk_no_value.add(mr_token)

        #         f.write('\t'.join([utterance_raw, utterance, " ".join(mr_padding)]) + '\n')

        with open(os.path.join(output_dir_thingtalk, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list_thingtalk)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab_thingtalk)):
                f.write(token + '\n')

        # with open(os.path.join(output_dir_thingtalk_no_value, 'output_vocab.txt'), 'w') as f:
        #     for idx in range(max(len_list_thingtalk_no_value)):
        #         f.write('@ptr{}'.format(idx) + '\n')
        #     for token in sorted(list(output_vocab_thingtalk_no_value)):
        #         f.write(token + '\n')


def read_mcwq(model, output_dir):
    languages = ['en', 'kn', 'he', 'zh']
    if MULTILINGUAL:
        languages = ['multilingual']
    # languages = ['multilingual']
    # languages = ['en']
    tokenizer = get_tokenizer(model)

    for language in languages:
        output_dir_language = os.path.join(output_dir, 'mcwq', model.replace('/', '_'), language)
        output_dir_language = os.path.join(output_dir_language, "sparql")
        if not os.path.exists(output_dir_language):
            os.makedirs(output_dir_language)
        # print(output_dir_language)

        output_vocab = set()
        len_list = []

        with open(os.path.join(output_dir_language, "test.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mcwq/mcd3/test.json", language, "sparql")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list.append(len(utterances_tokenized))

                mr_original = mr
                mr_padding = mr.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if re.search("^M\d$", mr_token):
                        # print(utterances_tokenized)
                        # print(mr)
                        # print(mr_token)
                        # print(mr_token in utterances_tokenized)
                        # print()
                        assert (mr_token in utterances_tokenized)
                        pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                        query_pointers.append(pointer)
                    else:
                        output_vocab.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_language, "dev.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mcwq/mcd3/dev.json", language, "sparql")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list.append(len(utterances_tokenized))

                mr_original = mr
                mr_padding = mr.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if re.search("^M\d$", mr_token):
                        # print(utterances_tokenized)
                        # print(mr)
                        # print(mr_token)
                        # print()
                        assert (mr_token in utterances_tokenized)
                        pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                        query_pointers.append(pointer)
                    else:
                        output_vocab.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_language, "train.tsv"), "w") as f:
            utterances, mrs = get_utterances_mr(f"{DATASET_PATH}mcwq/mcd3/train.json", language, "sparql")
            for utterance, mr in zip(utterances, mrs):
                raw_utterance = utterance
                utterance = tokenizer.cls_token + ' ' + utterance + ' ' + tokenizer.sep_token
                utterances_tokenized = utterance.split()
                len_list.append(len(utterances_tokenized))

                mr_original = mr
                mr_padding = mr.split()

                query_pointers = []
                for mr_token in mr_padding:
                    if re.search("^M\d$", mr_token):
                        # print(utterances_tokenized)
                        # print(mr)
                        # print(mr_token)
                        # print()
                        assert (mr_token in utterances_tokenized)
                        pointer = "@ptr{}".format(utterances_tokenized.index(mr_token))
                        query_pointers.append(pointer)
                    else:
                        output_vocab.add(mr_token)
                        query_pointers.append(mr_token)
                query_pointers = tokenizer.cls_token + ' ' + ' '.join(query_pointers) + ' ' + tokenizer.sep_token
                f.write('\t'.join([raw_utterance, utterance, query_pointers]) + '\n')

        with open(os.path.join(output_dir_language, 'output_vocab.txt'), 'w') as f:
            for idx in range(max(len_list)):
                f.write('@ptr{}'.format(idx) + '\n')
            for token in sorted(list(output_vocab)):
                f.write(token + '\n')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # This argument only indicates dataset name, not directory.
    # Default SpiderX dataset directory is "../../dataset/arg.dataset".
    parser.add_argument('--dataset', type=str, default='mspider')

    # The huggingface model used to do tokenization.
    # Now, we support 11 models: 'bert-base-cased', 'bert-base-multilingual-cased',
    # 'roberta-base', 'roberta-large',  'xlm-roberta-base', 'xlm-roberta-large',
    # 'facebook/bart-base', 'facebook/bart-large',
    # 'facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt',
    # 't5-large', 't5-base', 't5-small', 'google/mt5-large'
    parser.add_argument('--model', type=str, default='xlm-roberta-large')

    # The directory of output data folder.
    parser.add_argument('--output_dir', type=str, default='data')

    # Load the dataset for different settings
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)

    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    MULTILINGUAL = "multilingual" in DATASET_PATH
    # If multilingual, the only language we have in the data file is multilingual

    if args.dataset == 'mtop':
        read_mtop(args.model, args.output_dir)

    if args.dataset == 'mgeoquery':
        read_mgeoquery(args.model, args.output_dir)

    if args.dataset == 'mspider':
        read_mspider(args.model, args.output_dir)

    if args.dataset == 'matis':
        read_matis(args.model, args.output_dir)

    if args.dataset == 'mfree917':
        read_mfree917(args.model, args.output_dir)

    if args.dataset == 'mlsp':
        read_mlsp(args.model, args.output_dir)

    if args.dataset == 'mnlmaps':
        read_mnlmaps(args.model, args.output_dir)

    if args.dataset == 'movernight':
        read_movernight(args.model, args.output_dir)

    if args.dataset == 'mschema2qa':
        read_mschema2qa(args.model, args.output_dir)

    if args.dataset == 'mcwq':
        read_mcwq(args.model, args.output_dir)
