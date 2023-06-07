import os, re, random, sys, json
import sqlparse, calendar, num2words
from collections import defaultdict

random.seed(0)

def read_spider():
    ### Source files:
    ###     Original Spider:   dataset/spider/spider
    ###     CSpider:           dataset/spider/CSpider
    ###     VSpider:         dataset/spider/ViText2SQL
    ### Output files:
    ###     dataset/mspider/

    # Copy the original files from spider
    os.system('cp dataset/spider/spider/tables.json dataset/mspider')
    os.system('cp dataset/spider/spider/dev_gold.sql dataset/mspider')
    os.system('cp dataset/spider/spider/train_gold.sql dataset/mspider')

    # Delete geoquery examples in spider_train
    with open('dataset/mspider/train_gold.sql') as f:
        lines = [line for line in f.readlines() if not line.endswith('geo\n')]
    with open('dataset/mspider/train_gold.sql', 'w') as f:
        f.write(''.join(lines))

    # Read original spider dataset
    print('Spider')
    dev_json = "dataset/spider/spider/dev.json"
    with open(dev_json) as f:
        spider_dev = json.load(f)
        print('spider_dev', len(spider_dev))

    train_json = "dataset/spider/spider/train_spider.json"
    with open(train_json) as f:
        spider_train = json.load(f)
        print('spider_train', len(spider_train))

    train_json = "dataset/spider/spider/train_others.json"
    with open(train_json) as f:
        spider_train_others = json.load(f)
        print('spider_train_others', len(spider_train_others))
    
    spider_train = spider_train + spider_train_others

    # Read CSpider
    print('CSpider')
    dev_json = "dataset/spider/CSpider/dev.json"
    with open(dev_json) as f:
        cspider_dev = json.load(f)
        print('cspider_dev', len(cspider_dev))

    train_json = "dataset/spider/CSpider/train.json"
    with open(train_json) as f:
        cspider_train = json.load(f)
        print('cspider_train', len(cspider_train))

    # original spider and CSpider have the same split
    assert(len(spider_dev) == len(cspider_dev))
    assert(len(spider_train) == len(cspider_train))

    # Read VSpider (syllable-level)
    print('ViText2SQL')
    dev_json = "dataset/spider/ViText2SQL/data/syllable-level/dev.json"
    with open(dev_json) as f:
        vspider_dev = json.load(f)
        print('vspider_dev', len(vspider_dev))    

    train_json = "dataset/spider/ViText2SQL/data/syllable-level/train.json"
    with open(train_json) as f:
        vspider_train = json.load(f)
        print('vspider_train', len(vspider_train))

    test_json = "dataset/spider/ViText2SQL/data/syllable-level/test.json"
    with open(test_json) as f:
        vspider_test = json.load(f)
        print('vspider_test', len(vspider_test))

    # Note that VSpider use original train+dev and split them into train+dev+test
    # So we need to reorganize
    vspider = defaultdict(list)
    for example in vspider_dev+vspider_train+vspider_test:
        db_id = example['db_id']
        vspider[db_id].append(example)

    db_id_cnt = defaultdict(int)
    for dev1, dev2 in zip(spider_dev, cspider_dev):
        assert(dev1['db_id'] == dev2['db_id'])
        # Find corresponding vspider example
        db_id = dev1['db_id']
        vspider_cnt = db_id_cnt[db_id]
        vi_example = vspider[db_id][vspider_cnt]
        db_id_cnt[db_id] = db_id_cnt[db_id] + 1

        # Don't store question_toks or sql
        dev1.pop('question_toks')
        dev1.pop('sql')
        dev1['mr'] = {}
        dev1['mr']['sql'] = dev1['query']
        dev1['mr']['sql_toks'] = dev1['query_toks']
        dev1['mr']['sql_toks_no_value'] = dev1['query_toks_no_value']
        dev1.pop('query')
        dev1.pop('query_toks')
        dev1.pop('query_toks_no_value')
        dev1['question'] = {"en": dev1['question'], 
                            "zh": dev2['question'],
                            "vi": vi_example['question']}

    for train1, train2 in zip(spider_train, cspider_train):
        assert(train1['db_id'] == train2['db_id'])
        assert(train1['query_toks_no_value'] == train2['query_toks_no_value'])

        db_id = train1['db_id']
        vspider_cnt = db_id_cnt[db_id]
        vi_example = vspider[db_id][vspider_cnt]
        db_id_cnt[db_id] = db_id_cnt[db_id] + 1

        train1.pop('question_toks')
        train1.pop('sql')
        train1['mr'] = {}
        train1['mr']['sql'] = train1['query']
        train1['mr']['sql_toks'] = train1['query_toks']
        train1['mr']['sql_toks_no_value'] = train1['query_toks_no_value']
        train1.pop('query')
        train1.pop('query_toks')
        train1.pop('query_toks_no_value')
        train1['question'] = {"en": train1['question'], 
                              "zh": train2['question'],
                              "vi": vi_example['question']}    
    
    # Delete geoquery examples in spider_train
    spider_train = [train for train in spider_train if not(train['db_id'] == 'geo')]

    print('train', len(spider_train), 'dev', len(spider_dev))

    with open('dataset/mspider/dev.json', 'w', encoding='utf-8') as f:
        json.dump(spider_dev, f, ensure_ascii=False, indent=4)

    with open('dataset/mspider/train.json', 'w', encoding='utf-8') as f:
        json.dump(spider_train, f, ensure_ascii=False, indent=4)

    return


def read_geoquery():
    ### Source files:
    ###     english with sql:            dataset/text2sql-data/data
    ###     translation with logic form: dataset/statnlp-sp-neural-master/data/geoquery
    ### Output files:
    ###     dataset/mgeoquery/

    train = []
    dev = []
    test = []

    # Read translation with FunQL
    translations = defaultdict(dict)
    geo_logic = defaultdict(str)
    geo_en_to_id = defaultdict(list)
    languages = ['en', 'de', 'el', 'fa', 'id', 'sv', 'th', 'zh']
    for language in languages:
        language_file = "dataset/statnlp-sp-neural-master/data/geoquery/geoFunql-{}.corpus".format(language)
        cnt = 0
        with open(language_file) as f:
            for line in f:
                if line.startswith('id:'):
                    geo_id = int(line[3:])
                    cnt += 1
                elif line.startswith('nl:'):
                    translations[geo_id][language] = line[3:].strip()
                    if language == 'en':
                        text = line[3:].strip().strip('.').strip(',').strip('?').strip()
                        if text == "what rivers flow though colorado":
                            text = "what rivers flow through colorado"
                        translations[geo_id][language] = text
                        geo_en_to_id[text].append(geo_id)
                elif line.startswith('mrl:'):
                    geo_logic[geo_id] = line[4:].strip()
        assert(cnt == 880)
    
    # Read english with lambda and Prolog
    train_prolog_f = 'dataset/Unimer/data/geo/geo_prolog_train_fixed.tsv'
    test_prolog_f = 'dataset/Unimer/data/geo/geo_prolog_test_fixed.tsv'
    train_lambda_f = 'dataset/Unimer/data/geo/geo_lambda_calculus_train.tsv'
    test_lambda_f = 'dataset/Unimer/data/geo/geo_lambda_calculus_test.tsv'

    geo_prolog = {}
    geo_en_to_id_prolog = defaultdict(list)
    cnt = 0
    with open(train_prolog_f) as f:
        for line in f:
            text, prolog = line.strip().split('\t')
            text = text.strip().strip('.').strip(',').strip('?').strip()
            geo_prolog[cnt] = (text, prolog)
            geo_en_to_id_prolog[text].append(cnt)
            cnt += 1
    with open(test_prolog_f) as f:
        for line in f:
            text, prolog = line.strip().split('\t')
            text = text.strip().strip('.').strip(',').strip('?').strip()
            geo_prolog[cnt] = (text, prolog)
            geo_en_to_id_prolog[text].append(cnt)
            cnt += 1
    assert(cnt == 880)

    geo_lambda = {}
    geo_en_to_id_lambda = defaultdict(list)
    cnt = 0
    with open(train_lambda_f) as f:
        for line in f:
            text, lambda_c = line.strip().split('\t')
            text = text.strip().strip('.').strip(',').strip('?').strip()
            if text == "which states capital city is the largest":
                text = "which state 's capital city is the largest"
            geo_lambda[cnt] = (text, lambda_c)
            geo_en_to_id_lambda[text].append(cnt)
            cnt += 1
    with open(test_lambda_f) as f:
        for line in f:
            text, lambda_c = line.strip().split('\t')
            text = text.strip().strip('.').strip(',').strip('?').strip()
            if text == "which states capital city is the largest":
                text = "which state 's capital city is the largest"
            geo_lambda[cnt] = (text, lambda_c)
            geo_en_to_id_lambda[text].append(cnt)
            cnt += 1
    assert(cnt == 880)

    with open('geo.prolog', 'w') as f:
        for text in sorted(geo_prolog[cnt][0] for cnt in geo_prolog):
            f.write(text+'\n')
    with open('geo.lambda', 'w') as f:
        for text in sorted(geo_lambda[cnt][0] for cnt in geo_lambda):
            f.write(text+'\n')

    # Read manual cleaned sql (This is not needed as we already used normalized SQL)
    # manual = {}
    # with open('geo.clean') as f:
    #     examples = f.read().split('\n\n')
    #     for example in examples:
    #         eid, half_cleaned, cleaned = example.split('\n')
    #         manual[int(eid)] = (half_cleaned, cleaned)
    # print(manual)

    # Read english with sql
    # TODO: we need to use normalized sql files from https://github.com/talk2data/db-domain-adaptation/tree/equiv_patterns/revised
    #       the schema is also available https://github.com/talk2data/db-domain-adaptation/tree/equiv_patterns/tables

    # geo_sql_json = 'dataset/text2sql-data/data/geography.json'
    geo_sql_json = 'dataset/db-domain-adaptation/revised/geography.json'
    with open(geo_sql_json) as f:
        geo_sql_data = json.load(f)
    cnt = 0
    cnt_missing = 0
    cnt_missing_prolog_lambda = 0
    # special_sql_cnt = 0
    text_to_sql = {}
    missing = []
    for example in geo_sql_data:        
        for sentence in example['sentences']:
            cnt += 1
            text = sentence['text'].strip()
            # assert(len(example['sql'])==1)
            # if not(len(example['sql'])==1):
            #     print('here')
            #     print(len(example['sql']), example['sql'])
            #     print('end')
            #     exit()
            sql = example['sql'][0].strip(';').strip().replace('MAX(', 'MAX (').replace('MIN(', 'MIN (')
            for name, value in sentence['variables'].items():
                text = text.replace(name, value)
                sql = sql.replace(name, value)


            # TODO: preprocess sql to remove alias
            # print(text)
            # sql_tokens = sql.split()
            # alias_to_table = {}
            # for idx, token in enumerate(sql_tokens):
            #     if token == 'AS':
            #         table = sql_tokens[idx-1]
            #         alias = sql_tokens[idx+1]
            #         alias_to_table[alias] = table

            # clean_sql_tokens = []
            # if len(alias_to_table) == 1:
            #     alias = list(alias_to_table.keys())[0]
            #     table = alias_to_table[alias]
            #     for idx, token in enumerate(sql_tokens):
            #         if token == 'AS' or token == alias:
            #             continue
            #         if alias in token:
            #             token = token.split('.')[1]
            #         clean_sql_tokens.append(token)
            #     clean_sql = ' '.join(clean_sql_tokens)
            # elif ',' not in sql:
            #     for idx, token in enumerate(sql_tokens):
            #         if token == 'AS' or token in alias_to_table:
            #             continue
            #         if token.split('.')[0] in alias_to_table:
            #             token = token.split('.')[1]
            #         clean_sql_tokens.append(token)
            #     clean_sql = ' '.join(clean_sql_tokens)                   
            # else:
            #     # Check if it is special SQL (not handled in Spider format)
            #     special_sql = False
            #     for idx, token in enumerate(sql_tokens):
            #         if '.' in token and 'alias' in token.split('.')[0] and 'alias' in token.split('.')[1]:
            #             special_sql = True

            #     if special_sql:
            #         special_sql_cnt += 1
            #         clean_sql = sql
            #     else:
            #         # get tables in join statement
            #         join_tables_alias = []
            #         sql_format = sqlparse.format(sql, reindent=True).split('\n')
            #         join = False
            #         for line in sql_format:
            #             line = line.strip()
            #             if line.startswith('FROM') and line.endswith(','):
            #                 join = True
            #             elif line.startswith('WHERE'):
            #                 join = False
            #             # print(line, join)
            #             if join:
            #                 tokens = line.strip(',').split()
            #                 for idx, token in enumerate(tokens):
            #                     if token == 'AS':
            #                         table = tokens[idx-1]
            #                         alias = tokens[idx+1]
            #                         join_tables_alias.append(alias)                        
                    
            #         # replace those table alias with T1,....
            #         replace_alias = {}
            #         for i in range(len(join_tables_alias)):
            #             replace_alias[join_tables_alias[i]] = 'T{}'.format(i+1)

            #         # remove other alias
            #         other_alias = set(alias_to_table.keys()) - set(join_tables_alias)
            #         for idx, token in enumerate(sql_tokens):
            #             # remove other alias
            #             if (token == 'AS' and sql_tokens[idx+1] in other_alias) or token in other_alias:
            #                 continue
                        
            #             if token.split('.')[0] in other_alias:
            #                 # remove other alias
            #                 token = token.split('.')[1]
            #             elif token == ',':
            #                 # print(sql_tokens[idx-2], sql_tokens[idx-1], sql_tokens[idx])
            #                 if (sql_tokens[idx-2] == 'AS'):
            #                     token = 'JOIN'
            #             else:
            #                 # replace those table alias with T1,....
            #                 for k,v in replace_alias.items():
            #                     token = token.replace(k,v)
            #             clean_sql_tokens.append(token)
            #         clean_sql = ' '.join(clean_sql_tokens)

            #         # Manual: move the join condition from where to join using ON....
            #         if cnt in manual:
            #             assert (clean_sql == manual[cnt][0])
            #             clean_sql = manual[cnt][1]
                    # print(cnt)
                    # print(sql)
                    # print()
                    # print(sqlparse.format(sql, reindent=True))
                    # print()
                    # print(join_tables_alias)
                    # print()
                    # print(sqlparse.format(clean_sql, reindent=True))
                    # print()
                    # print(clean_sql)
                    # print(clean_sql)
                    # print()
                
            # print()
            text_to_sql[text] = sql

            split = sentence['question-split']
            ex = {}
            # ex['query'] = clean_sql
            ex['mr'] = {}
            ex['mr'] ['sql'] = sql

            original_text = text
            if text not in geo_en_to_id:
                if text == "what are the capital city in texas":
                    text = "what is the capital city in texas"
                elif text == "what are the population of mississippi":
                    text = "what is the population of mississippi"
                elif text == "what are the populations of states through which the mississippi river run":
                    text = "what are the populations of the states through which the mississippi river runs"
                elif text == "what are the populations of states through which the mississippi run":
                    text = "what are the populations of the states through which the mississippi river runs"
                elif text == "what are the populations of the states through which the mississippi river run":
                    text = "what are the populations of the states through which the mississippi river runs"
                elif text == "what are the populations of the states through which the mississippi run":
                    text = "what are the populations of the states through which the mississippi river runs"
                elif text == "what are the states that the potomac run through":
                    text = "what are the states that the potomac runs through"
                elif text == "what rivers run through texas":
                    text = "what rivers are in texas"
                elif text == "what states capital is dover":
                    text = "what state 's capital is dover"
                elif text == "what is the largest of the state that the rio grande runs through":
                    text = "what is the largest of the states that the rio grande runs through"
                elif text == "what is the population of atlanta georgia":
                    text = "what is the population of atlanta ga"
                elif text == "what states high point are higher than that of colorado":
                    text = "what state 's high point is higher than that of colorado"
                elif text == "whats the largest city":
                    text = "what 's the largest city"
                # elif text == "which us city has the highest population":
                #     text = ""
                # elif text == "what city in the united states has the highest population":
                #     text = ""
                # elif text == "what is the area of washington":
                #     text = ""
                else:
                    cnt_missing += 1
                    missing.append(text)
                    continue
            
            # if (text not in geo_en_to_id_prolog and original_text not in geo_en_to_id_prolog) or (text not in geo_en_to_id_lambda and original_text not in geo_en_to_id_lambda):
            #     print(text)
            #     print(original_text)
            #     cnt_missing_prolog_lambda += 1 
            if text in geo_en_to_id_prolog:
                uid = geo_en_to_id_prolog[text][0]
            else:
                uid = geo_en_to_id_prolog[original_text][0]
            ex['mr']['prolog'] = geo_prolog[uid][1]

            if text in geo_en_to_id_lambda:
                uid = geo_en_to_id_lambda[text][0]
            else:
                uid = geo_en_to_id_lambda[original_text][0]
            ex['mr']['lambda'] = geo_lambda[uid][1]
         
            uid = geo_en_to_id[text][0]
            ex['question'] = translations[uid]
            ex['mr']['funql'] = geo_logic[uid]
            if split == 'train':
                train.append(ex)
            elif split == 'dev':
                dev.append(ex)
            elif split == 'test':
                test.append(ex)

    print('translations', len(translations))
    print('cnt', cnt, 'cnt_missing', cnt_missing)
    print('train', len(train), 'dev', len(dev), 'test', len(test))

    with open('geo.missing', 'w') as f:
        for text in sorted(missing):
            f.write(text+'\n')
    with open('geo.source1', 'w') as f:
        for text in sorted(text_to_sql.keys()):
            f.write(text+'\n')
    with open('geo.source2', 'w') as f:
        for text in sorted(geo_en_to_id.keys()):
            f.write(text+'\n')
    # return

    with open('dataset/mgeoquery/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
    with open('dataset/mgeoquery/dev.json', 'w', encoding='utf-8') as f:
        json.dump(dev, f, ensure_ascii=False, indent=4)
    with open('dataset/mgeoquery/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    
    return


def read_atis():
    ### Source files:
    ###     english with sql:                 dataset/text2sql-data/data
    ###     translation with slot-intent BIO: dataset/atis_seven_languages/docs
    ### Output files:
    ###     dataset/matis/

    train = []
    dev = []
    test = []

    # Read translation with slot-intent BIO : 
    translations = defaultdict(dict)
    # languages = ['en', 'es', 'de', 'fr', 'ja', 'pt', 'zh']
    languages = ['en', 'es', 'de', 'fr', 'pt', 'zh']
    for language in languages:
        train_file = "dataset/atis/atis_seven_languages/data/train_{}.tsv".format(language.upper())
        with open(train_file) as f:
            for line in f.readlines():
                if line.startswith('id'):
                    continue
                uid, utterance, slot_labels, intent = line.strip().split('\t')
                translations['train'+uid][language] = utterance

        test_file = "dataset/atis/atis_seven_languages/data/test_{}.tsv".format(language.upper())
        with open(test_file) as f:
            for line in f.readlines():
                if line.startswith('id'):
                    continue
                uid, utterance, slot_labels, intent = line.strip().split('\t')
                translations['test'+uid][language] = utterance
    
    atis_en_to_id = defaultdict(list)
    for uid in translations:
        atis_en = translations[uid]['en']
        atis_en_to_id[atis_en].append(uid)

    # Read english with sql
    # TODO: we need to use normalized sql files from https://github.com/talk2data/db-domain-adaptation/tree/equiv_patterns/revised
    #       the schema is also available https://github.com/talk2data/db-domain-adaptation/tree/equiv_patterns/tables
    # atis_sql_json = 'dataset/text2sql-data/data/atis.json'
    atis_sql_json = 'dataset/db-domain-adaptation/revised/atis.json'
    with open(atis_sql_json) as f:
        atis_sql_data = json.load(f)
    
    atis_missing_done = {}
    atis_missing_done_f = 'atis_missing/atis.missing.done'
    with open(atis_missing_done_f) as f:
        lines = f.readlines()
        num_missing_done = int(len(lines) / 3)
    print('num_missing_done', num_missing_done)
    for i in range(num_missing_done):
        missing_id = int(lines[i*3].strip())
        original = lines[i*3+1].strip()
        match = lines[i*3+2].strip()
        atis_missing_done[missing_id] = (original, match)

    cnt = 0
    cnt_missing = 0
    text_to_sql = {}
    missing = []
    for example in atis_sql_data:        
        for sentence in example['sentences']:
            cnt += 1
            text = sentence['text'].strip()
            sql = example['sql'][0]
            for name, value in sentence['variables'].items():
                sql = sql.replace(name, value)
                if 'day_number' in name:
                    value = num2words.num2words(int(value), ordinal=True).replace('-', ' ')
                if 'month_number' in name:
                    value = calendar.month_name[int(value)]
                text = text.replace(name, value)
            
            text = text.lower()
            # text = text.replace("what 's", "what's")
            text = text.replace(" 's", "'s")
            text = text.replace("mke", "general mitchell international")
            text = text.replace("aa", "american airlines")
            # text = text.replace("us", "us air")
            text = text.replace('1200', 'noon')
            text = text.replace('1600', '4 pm')
            text = text.replace('1700', '5 pm')
            text = text.replace('1800', '6 pm')
            text = text.replace('2000', '8 pm')
            text = text.replace("400 o'clock", "4 o'clock")
            text = text.replace("900 o'clock", "9 o'clock")
            text = text.replace("800 o'clock", "8 o'clock")
            # text = text.replace("2000 o'clock", "8 o'clock")
            # text = text.replace("2100 o'clock", "9 o'clock")
            text = text.replace("900", "9 am")
            text = text.replace("800", "8 am")

            if text not in atis_en_to_id:
                if text.replace('washington', 'washington dc') in atis_en_to_id:
                    text = text.replace('washington', 'washington dc')
                if text.replace('new york', 'new york city') in atis_en_to_id:
                    text = text.replace('new york', 'new york city')
                if text.replace(' dl ', ' delta ') in atis_en_to_id:
                    text = text.replace(' dl ', ' delta ')
                if text.replace('dl ', 'delta ') in atis_en_to_id:
                    text = text.replace('dl ', 'delta ')
                if text.replace(' dl', ' delta') in atis_en_to_id:
                    text = text.replace(' dl', ' delta')
                if text.replace(' co ', ' continental ') in atis_en_to_id:
                    text = text.replace(' co ', ' continental ')
                if text.replace(' co', ' continental') in atis_en_to_id:
                    text = text.replace(' co', ' continental')
                if text.replace(' ua flights ', ' united flights ') in atis_en_to_id:
                    text = text.replace(' ua flights ', ' united flights ')
                if text.replace('us', 'us air') in atis_en_to_id:
                    text = text.replace('us', 'us air')
                if text.replace(' as flights ', ' alaska airline flights ') in atis_en_to_id:
                    text = text.replace(' as flights ', ' alaska airline flights ')
                if text.replace(' tw ', ' twa ') in atis_en_to_id:
                    text = text.replace(' tw ', ' twa ')
                if text.replace(' lga ', ' la guardia ') in atis_en_to_id:
                    text = text.replace(' lga ', ' la guardia ')
                if text.replace('dal', 'love field') in atis_en_to_id:
                    text = text.replace('dal', 'love field')
                if text.replace(' atl ', ' atlanta ') in atis_en_to_id:
                    text = text.replace(' atl ', ' atlanta ')
                # if text.replace(' lh ', ' lufthansa ') in atis_en_to_id:
                #     text = text.replace(' lh ', ' lufthansa ')
                # if text.replace('monday', 'mondays') in atis_en_to_id:
                #     text = text.replace('monday', 'mondays')
                # if text.replace('friday', 'fridays') in atis_en_to_id:
                #     text = text.replace('friday', 'fridays')
                # if text.replace('sunday', 'sundays') in atis_en_to_id:
                #     text = text.replace('sunday', 'sundays')
                # if text.replace('wednesday', 'wednesdays') in atis_en_to_id:
                #     text = text.replace('wednesday', 'wednesdays')
                
            
            text_to_sql[text] = sql
            # TODO: preprocess sql to remove alias
            # print(text)
            # print(sql)
            # print()
            split = sentence['question-split']
            ex = {}
            # ex['query'] = sql
            # ex['text'] = text
            ex['mr'] = {}
            ex['mr'] ['sql'] = sql

            if text not in atis_en_to_id:
                if cnt in atis_missing_done:
                    original, match = atis_missing_done[cnt]
                    assert(original == text)
                    text =  match
                    assert(text in atis_en_to_id)
                else:
                    cnt_missing += 1
                    # closest_match = difflib.get_close_matches(text, atis_en_to_id.keys(), n=1)
                    closest_match = [text]
                    missing.append((str(cnt), text, closest_match[0]))
                    continue

            uid = atis_en_to_id[text][0]
            ex['question'] = translations[uid]
            if split == 'train':
                train.append(ex)
            elif split == 'dev':
                dev.append(ex)
            elif split == 'test':
                test.append(ex)

    print('translations', len(translations), 'atis_en_to_id', len(atis_en_to_id))
    print('cnt', cnt, 'cnt_missing', cnt_missing)
    print('train', len(train), 'dev', len(dev), 'test', len(test))


    # TODO: We need to manually check translation
    #
    #       We will use three files:
    #       missing: each example has id, english in text2sql data, best difflib match of english in atis_seven_languages
    #       source1: all sorted english utterances in text2sql data
    #       source2: all sorted english utterances in atis_seven_languages
    #
    #       The cleaning step is:
    #       for each example in missing, check if two sentences really match
    #            if match, do nothing
    #            if not match, then keep the first sentence, search in source2 the closest one to it, and copy the closest one as the second sentence 
    #        
    #       Note that some entity names does not match (e.g., love field vs DAL), but it is hard to change it because all the translations are based on love field while the SQL is based on DAL. But this is probably acceptable, especially if we don't predict value.
    #
    # return

    # with open('atis.missing', 'w') as f:
    #     for text in missing:
    #         f.write(text[0]+'\n'+text[1]+'\n'+text[2]+'\n')
    # with open('atis.source1', 'w') as f:
    #     for text in sorted(text_to_sql.keys()):
    #         f.write(text+'\n')
    # with open('atis.source2', 'w') as f:
    #     for text in sorted(atis_en_to_id.keys()):
    #         # if len(atis_en_to_id[text]) > 1:
    #         #     print(text, atis_en_to_id[text])
    #         f.write(text+'\n')

    with open('dataset/matis/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
    with open('dataset/matis/dev.json', 'w', encoding='utf-8') as f:
        json.dump(dev, f, ensure_ascii=False, indent=4)
    with open('dataset/matis/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    
    return


def read_overnight():
    train = []
    dev = []
    test = []

    domain_cnt = defaultdict(int)

    overnight_file = "dataset/bootstrap/onight_master_27_03_2021.json"
    with open(overnight_file) as f:
        overnight_data = json.load(f)
    for ex in overnight_data['examples']:
        # print(ex['id'])
        split = ex['split']
        example = {}
        example['domain'] = ex['domain']
        domain_cnt[ex['domain']] += 1
        # example['query'] = ex['lf']
        example['mr'] = {}
        example['mr']['lambda'] = ex['lf']

        example['question'] = {}
        example['question']['en'] = ex['nl']
        if split == 'train':
            example['question']['zh'] = ex['mt']['zh']['baidu']
            example['question']['de'] = ex['mt']['de']['gtranslate']
        else:
            example['question']['zh'] = ex['zh']
            example['question']['de'] = ex['de']

        if split == 'train':
            train.append(example)
        elif split == 'dev':
            dev.append(example)
        elif split == 'test':
            test.append(example)
    
    print(len(overnight_data['examples']))
    print('train', len(train), 'dev', len(dev), 'test', len(test))
    print(domain_cnt)
    
    with open('dataset/movernight/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
    with open('dataset/movernight/dev.json', 'w', encoding='utf-8') as f:
        json.dump(dev, f, ensure_ascii=False, indent=4)
    with open('dataset/movernight/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    
    return


def read_nlmaps():
    train = []
    test = []

    train_en_f = "dataset/nlmaps/nlmaps.train.en"
    train_de_f = "dataset/nlmaps/nlmaps.train.de"
    train_query_f = "dataset/nlmaps/nlmaps.train.mrl"

    test_en_f = "dataset/nlmaps/nlmaps.test.en"
    test_de_f = "dataset/nlmaps/nlmaps.test.de"
    test_query_f = "dataset/nlmaps/nlmaps.test.mrl"

    train_en = []
    train_de= []
    train_query = []
    with open(train_en_f) as f:
        for line in f.readlines():
            train_en.append(line.strip())
    with open(train_de_f) as f:
        for line in f.readlines():
            train_de.append(line.strip())
    with open(train_query_f) as f:
        for line in f.readlines():
            train_query.append(line.strip())
    for train_en_ex, train_de_ex, train_query_ex in zip(train_en, train_de, train_query):
        example = {}
        # example['query'] = train_query_ex
        example['mr'] = {}
        example['mr']['funql'] = train_query_ex
        example['question'] = {}
        example['question']['en'] = train_en_ex
        example['question']['de'] = train_de_ex
        train.append(example)

    test_en = []
    test_de= []
    test_query = []
    with open(test_en_f) as f:
        for line in f.readlines():
            test_en.append(line.strip())
    with open(test_de_f) as f:
        for line in f.readlines():
            test_de.append(line.strip())
    with open(test_query_f) as f:
        for line in f.readlines():
            test_query.append(line.strip())
    for test_en_ex, test_de_ex, test_query_ex in zip(test_en, test_de, test_query):
        example = {}
        # example['query'] = test_query_ex
        example['mr'] = {}
        example['mr']['funql'] = test_query_ex
        example['question'] = {}
        example['question']['en'] = test_en_ex
        example['question']['de'] = test_de_ex
        test.append(example)

    print('train', len(train), 'test', len(test))

    with open('dataset/mnlmaps/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
    with open('dataset/mnlmaps/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    
    return


def read_mtop():
    languages = ['en','de','fr','th','es','hi']
    train_dict = {}
    dev_dict = {}
    test_dict = {}

    bad = 0
    for language in languages:
        train_language = 'dataset/mtop/{}/train.txt'.format(language)
        with open(train_language) as f:
            for line in f.readlines():
                uid, intent, slot, utterance, domain, locale, query, token = line.strip().split('\t')
                token = token.replace('"""', '"\\""')
                token = json.loads(token)
                utterance = ' '.join(token['tokens'])

                query_correct = []
                for query_token in query.split():
                    if '[' in query_token or ']' in query_token:
                        query_correct.append(query_token)
                    elif query_token in utterance.split():
                        query_correct.append(query_token)
                    elif query_token+'s' in utterance.split():
                        query_correct.append(query_token+'s')
                    elif query_token[:-1] in utterance.split():
                        query_correct.append(query_token[:-1])
                    elif query_token.lower() in utterance.split():
                        query_correct.append(query_token.lower())
                    elif query_token.capitalize() in utterance.split():
                        query_correct.append(query_token.capitalize())
                    elif query_token == 'Whatsapp' and 'WhatsApp' in utterance.split():
                        query_correct.append('WhatsApp')
                    else:
                        # print(language, utterance)
                        # print(query)
                        # print(query_token)
                        # print()
                        # exit()
                        bad += 1
                        break
                if len(query_correct) != len(query.split()):
                    continue
                query = ' '.join(query_correct)
                if uid not in train_dict:
                    train_dict[uid] = {}
                    train_dict[uid]['question'] = {}
                    train_dict[uid]['query'] = {}
                train_dict[uid]['question'][language] = utterance
                train_dict[uid]['query'][language] = query
        
        dev_language = 'dataset/mtop/{}/eval.txt'.format(language)
        with open(dev_language) as f:
            for line in f.readlines():
                uid, intent, slot, utterance, domain, locale, query, token = line.strip().split('\t')
                token = json.loads(token)
                utterance = ' '.join(token['tokens'])
                query_correct = []
                for query_token in query.split():
                    if '[' in query_token or ']' in query_token:
                        query_correct.append(query_token)
                    elif query_token in utterance.split():
                        query_correct.append(query_token)
                    elif query_token+'s' in utterance.split():
                        query_correct.append(query_token+'s')
                    elif query_token[:-1] in utterance.split():
                        query_correct.append(query_token[:-1])
                    elif query_token.lower() in utterance.split():
                        query_correct.append(query_token.lower())
                    elif query_token.capitalize() in utterance.split():
                        query_correct.append(query_token.capitalize())
                    elif query_token == 'Whatsapp' and 'WhatsApp' in utterance.split():
                        query_correct.append('WhatsApp')
                    else:
                        # print(language, utterance)
                        # print(query)
                        # print(query_token)
                        # print()
                        # exit()
                        bad += 1
                        break
                if len(query_correct) != len(query.split()):
                    continue
                query = ' '.join(query_correct)
                if uid not in dev_dict:
                    dev_dict[uid] = {}
                    dev_dict[uid]['question'] = {}
                    dev_dict[uid]['query'] = {}
                dev_dict[uid]['question'][language] = utterance
                dev_dict[uid]['query'][language] = query
        
        test_language = 'dataset/mtop/{}/test.txt'.format(language)
        with open(test_language) as f:
            for line in f.readlines():
                uid, intent, slot, utterance, domain, locale, query, token = line.strip().split('\t')
                token = json.loads(token)
                utterance = ' '.join(token['tokens'])
                query_correct = []
                for query_token in query.split():
                    if '[' in query_token or ']' in query_token:
                        query_correct.append(query_token)
                    elif query_token in utterance.split():
                        query_correct.append(query_token)
                    elif query_token+'s' in utterance.split():
                        query_correct.append(query_token+'s')
                    elif query_token[:-1] in utterance.split():
                        query_correct.append(query_token[:-1])
                    elif query_token.lower() in utterance.split():
                        query_correct.append(query_token.lower())
                    elif query_token.capitalize() in utterance.split():
                        query_correct.append(query_token.capitalize())
                    elif query_token == 'Whatsapp' and 'WhatsApp' in utterance.split():
                        query_correct.append('WhatsApp')
                    else:
                        # print(language, utterance)
                        # print(query)
                        # print(query_token)
                        # print()
                        # exit()
                        bad += 1
                        break
                if len(query_correct) != len(query.split()):
                    continue
                query = ' '.join(query_correct)
                if uid not in test_dict:
                    test_dict[uid] = {}
                    test_dict[uid]['question'] = {}
                    test_dict[uid]['query'] = {}
                test_dict[uid]['question'][language] = utterance
                test_dict[uid]['query'][language] = query            
    print(bad)
    train = []
    for uid in train_dict.keys():
        example = {}
        example['id'] = uid
        example['mr'] = {}
        example['mr']['slot_intent'] = train_dict[uid]['query']
        example['question'] = train_dict[uid]['question']
        if len(example['question']) == len(example['mr']['slot_intent']) == len(languages):
            train.append(example)

    dev = []
    for uid in dev_dict.keys():
        example = {}
        example['id'] = uid
        example['mr'] = {}
        example['mr']['slot_intent'] = dev_dict[uid]['query']
        example['question'] = dev_dict[uid]['question']
        if len(example['question']) == len(example['mr']['slot_intent']) == len(languages):
            dev.append(example)

    test = []
    for uid in test_dict.keys():
        example = {}
        example['id'] = uid
        example['mr'] = {}
        example['mr']['slot_intent'] = test_dict[uid]['query']
        example['question'] = test_dict[uid]['question']
        if len(example['question']) == len(example['mr']['slot_intent']) == len(languages):
            test.append(example)

    print('train', len(train), 'dev', len(dev), 'test', len(test))

    with open('dataset/mtop/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
    with open('dataset/mtop/dev.json', 'w', encoding='utf-8') as f:
        json.dump(dev, f, ensure_ascii=False, indent=4)
    with open('dataset/mtop/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)

    return


def read_schema2qa():
    domains = ['hotels', 'restaurants']
    languages = ['en', 'ar', 'de', 'es', 'fa', 'fi', 'it', 'ja', 'pl', 'tr', 'zh']

    train_dict = {}
    test_dict = {}
    for language in languages:
        train_dict[language] = {}
        test_dict[language] = {}
        for domain in domains:
            train_dict[language][domain] = {}
            train_file = "dataset/SPL/spl-release/dataset/{}/{}/train.tsv".format(domain, language)
            with open(train_file) as f:
                for line in f.readlines():
                    uid, utterance, query = line.strip().split('\t')

                    if language == 'en':
                        assert(len(uid.split('-')) == 3)
                        uid_original = uid
                    else:
                        # normalize id for non-English datasets
                        uid_original = uid
                        uid = '-'.join(uid.split('-')[:3])
                        uid = re.sub('^RSR', '', uid)
                        if re.search('^[0-9]', uid) or re.search('^S', uid):
                            uid = 'R' + uid
                        assert(uid in train_dict['en'][domain])
                        # if not uid in train_dict['en'][domain]:
                        #     print('not correct', language, domain, uid_original, uid)

                    # assert(uid not in train_dict[language][domain])
                    # TODO: some uid in train set may appear multiple times
                    if uid not in train_dict[language][domain]:
                        train_dict[language][domain][uid] = {}
                        train_dict[language][domain][uid]['utterance'] = utterance
                        train_dict[language][domain][uid]['query'] = query
                    # else:
                    #     print('multiple', language, domain, uid_original, uid)

            test_dict[language][domain] = {}
            test_file = "dataset/SPL/spl-release/dataset/{}/{}/eval.tsv".format(domain, language)
            with open(test_file) as f:
                for line in f.readlines():
                    uid, utterance, query = line.strip().split('\t')
                    # RRSR331-0-0-0
                    # RRS331-0-0
                    if re.search("^RRSR\d{3}-0-0-0$", uid):
                        uid = uid[3:-4]
                    elif re.search("^RRS\d{3}-0-0$", uid):
                        uid = 'R'+uid[3:-2]
                    assert(uid not in test_dict[language][domain])
                    test_dict[language][domain][uid] = {}
                    test_dict[language][domain][uid]['utterance'] = utterance
                    test_dict[language][domain][uid]['query'] = query
    
    for language in languages:
        print('train', language, sum([len(train_dict[language][domain]) for domain in domains]))
    for language in languages:
        print('test', language, sum([len(test_dict[language][domain]) for domain in domains]))
    
    def normalize_thingtalk(query):
        # replace values with the value token 
        query = re.sub("[0-9]+", "value", query)
        query = query.replace("[ value ]", "[ 1 ]")
        query = re.sub("\"[\w\s]+\"", "\" value \"", query)
        return query

    train = []
    for domain in domains:
        for uid in sorted(train_dict['en'][domain].keys()): 
            example = {}
            example['id'] = uid
            example['domain'] = domain

            example['mr'] = {}
            example['mr']['thingtalk'] = {}
            for language in languages:
                if uid in train_dict[language][domain]:
                    example['mr']['thingtalk'][language] = train_dict[language][domain][uid]['query']
            example['mr']['thingtalk_no_value'] = normalize_thingtalk(train_dict['en'][domain][uid]['query'])

            example['question'] = {}            
            for language in languages:
                if uid in train_dict[language][domain]:
                    example['question'][language] = train_dict[language][domain][uid]['utterance']
            
            if len(example['question']) == len(example['mr']['thingtalk']) == len(languages):
                train.append(example)
            # else:
            #     print(len(example['question']), len(languages))

    test = []
    for domain in domains:
        for uid in sorted(test_dict['en'][domain].keys()): 
            example = {}
            example['id'] = uid
            example['domain'] = domain

            example['mr'] = {}
            example['mr']['thingtalk'] = {}
            for language in languages:
                example['mr']['thingtalk'][language] = test_dict[language][domain][uid]['query']
            example['mr']['thingtalk_no_value'] = normalize_thingtalk(test_dict['en'][domain][uid]['query'])
            
            example['question'] = {}
            for language in languages:
                example['question'][language] = test_dict[language][domain][uid]['utterance']

            test.append(example)
    
    random.shuffle(train)
    train = train[:int(len(train)/20)]
    print('train', len(train), 'test', len(test))
    
    with open('dataset/mschema2qa/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
    with open('dataset/mschema2qa/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    
    return


def read_mcwq():
    dataset_json = 'dataset/seq2sparql/cwq/dataset.json'
    random_split_json = 'dataset/seq2sparql/cwq/split/random_split.json'
    mcd3_split_json = 'dataset/seq2sparql/cwq/split/mcd3.json'
    with open(dataset_json) as f:
        data = json.load(f)
    with open(random_split_json) as f:
        random_split = json.load(f)
    with open(mcd3_split_json) as f:
        mcd3_split = json.load(f)

    print(len(data))
    print('Random Split:', len(random_split['trainIdxs']), len(random_split['devIdxs']), len(random_split['testIdxs']))
    print('MCD3 Split:', len(mcd3_split['trainIdxs']), len(mcd3_split['devIdxs']), len(mcd3_split['testIdxs']))
    random_train = []
    random_dev = []
    random_test = []
    mcd3_train = []
    mcd3_dev = []
    mcd3_test = []
    for idx, ex in enumerate(data):
        # print(ex.keys())
        # exit()
        # idx = ex['CFQquestionIdx']
        example = {}
        example['mr'] = {}
        # example['mr']['sparql'] = ex['sparql'].replace('\n', '')
        example['mr']['sparql'] = ex['sparqlPatternModEntities'].replace('\n', ' ')
        # .replace('.', '. ').replace('{', '{ ').replace('}', ' }')
        example['question'] = {}
        # example['question']['en'] = ex['questionWithBrackets']
        # example['question']['kn'] = ex['questionWithBrackets_kn']
        # example['question']['he'] = ex['questionWithBrackets_he']
        # example['question']['zh'] = ex['questionWithBrackets_zh']
        example['question']['en'] = ex['questionPatternModEntities']
        example['question']['kn'] = ex['questionPatternModEntities_kn']
        example['question']['he'] = ex['questionPatternModEntities_he']
        example['question']['zh'] = ' '.join(list(ex['questionPatternModEntities_zh'])).replace('M 0', 'M0').replace('M 1', 'M1').replace('M 2', 'M2').replace('M 3', 'M3').replace('M 4', 'M4').replace('M 5', 'M5').replace('M 6', 'M6').replace('M 7', 'M7').replace('M 8', 'M8')

        skip = False
        for mr_token in example['mr']['sparql'].split():
            if re.search("^M\d$", mr_token):
                if not mr_token in example['question']['en'].split():
                    skip = True
                    break
                if not mr_token in example['question']['kn'].split():
                    skip = True
                    break
                if not mr_token in example['question']['he'].split():
                    skip = True
                    break
                if not mr_token in example['question']['zh'].split():
                    skip = True
                    break
        if skip:
            continue

        # print(idx)
        if idx in random_split['trainIdxs']:
            random_train.append(example)
        elif idx in random_split['devIdxs']:
            random_dev.append(example)
        elif idx in random_split['testIdxs']:
            random_test.append(example)
        # else:
            # print('not found', idx)
            # exit()

        if idx in mcd3_split['trainIdxs']:
            mcd3_train.append(example)
        elif idx in mcd3_split['devIdxs']:
            mcd3_dev.append(example)
        elif idx in mcd3_split['testIdxs']:
            mcd3_test.append(example)

    print('Random', 'train', len(random_train), 'dev', len(random_dev), 'test', len(random_test))
    print('MCD3', 'train', len(mcd3_train), 'dev', len(mcd3_dev), 'test', len(mcd3_test))

    with open('dataset/mcwq/random/train.json', 'w', encoding='utf-8') as f:
        json.dump(random_train, f, ensure_ascii=False, indent=4)
    with open('dataset/mcwq/random/dev.json', 'w', encoding='utf-8') as f:
        json.dump(random_dev, f, ensure_ascii=False, indent=4)
    with open('dataset/mcwq/random/test.json', 'w', encoding='utf-8') as f:
        json.dump(random_test, f, ensure_ascii=False, indent=4)

    with open('dataset/mcwq/mcd3/train.json', 'w', encoding='utf-8') as f:
        json.dump(mcd3_train, f, ensure_ascii=False, indent=4)
    with open('dataset/mcwq/mcd3/dev.json', 'w', encoding='utf-8') as f:
        json.dump(mcd3_dev, f, ensure_ascii=False, indent=4)
    with open('dataset/mcwq/mcd3/test.json', 'w', encoding='utf-8') as f:
        json.dump(mcd3_test, f, ensure_ascii=False, indent=4)
    return


if __name__ == "__main__":
    #### Spider
    read_spider()
    #### ATIS
    read_atis()
    #### GeoQuery
    read_geoquery()
    #### NLmaps
    read_nlmaps()
    #### Overnight
    read_overnight()
    #### Schema2QA
    read_schema2qa()
    #### MTOP
    read_mtop()
    #### mCWQ
    read_mcwq()