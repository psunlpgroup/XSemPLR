import argparse
import random, os, csv, logging, json, copy, math, operator, copy, subprocess
from queue import PriorityQueue
from typing import Optional
from dataclasses import dataclass

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm, trange

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Sampler)
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer,
                          BartTokenizer, MBart50Tokenizer, T5Tokenizer, MT5Tokenizer, AutoTokenizer,
                          BertConfig, RobertaConfig, XLMRobertaConfig,
                          BartConfig, MBartConfig, T5Config, MT5Config, AutoConfig,
                          AdamW, Adafactor, get_linear_schedule_with_warmup)
from src.data import create_sampler_dataloader
# from src.data import create_sampler_dataloader, DataProcessor, semParse_convert_examples_to_features, spider_convert_examples_to_features, BucketSampler
from src.ptrbert import PtrRoberta, PtrBART, PtrT5
from evaluate import get_exact_match

import wandb

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

SUPPORT_MODELS = ["xlm-roberta-large", "bert-base-multilingual-cased"]

def read_output_vocab(tokenizer, output_vocab_f, dataset):
    # TODO: This function is very important. Change this will probably cause model not to work.
    # if not tokenizer.cls_token:
    #     tokenizer.cls_token = '[cls]'
    # if not tokenizer.sep_token:
    #     tokenizer.sep_token = '[sep]'
    if not tokenizer.cls_token:
        tokenizer.cls_token = tokenizer.pad_token
    if not tokenizer.sep_token:
        tokenizer.sep_token = tokenizer.eos_token

    if dataset == 'MTOP':
        vocab = [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token, ']']
    elif dataset == 'TOP':
        vocab = [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
    elif dataset in ['MGEOQUERY', 'MSPIDER', 'MFREE917', 'MCWQ', 'MATIS', 'MSCHEMA2QA', 'MNLMAPS', 'MOVERNIGHT']:
        vocab = [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
    else:
        print('unknown dataset', dataset)
        exit()

    slot_vocab = []
    intent_vocab = []
    ptr_vocab = []
    all_output = [] 
    num_ptr = 0
    num_slot = 0 
    num_intent = 0

    with open(output_vocab_f) as f:
        for line in f:
            token = line.strip()
            if dataset in ['MTOP', 'TOP']:
                if 'SL:' in token:
                    num_slot += 1
                    slot_vocab.append(token)
                elif 'IN:' in token:
                    num_intent += 1
                    intent_vocab.append(token)
                elif '@ptr' in token:
                    num_ptr += 1
                    ptr_vocab.append(token)
                else:
                    assert(dataset == 'MTOP')
                    num_slot += 1
                    # vocab.append(token)
            elif dataset in ['MGEOQUERY', 'MSPIDER', 'MFREE917', 'MCWQ', 'MATIS', 'MSCHEMA2QA', 'MNLMAPS', 'MOVERNIGHT']:
                if '@ptr' in token:
                    num_ptr += 1
                    ptr_vocab.append(token)
                else:
                    vocab.append(token)

    vocab = vocab + slot_vocab + intent_vocab
    all_output = vocab + ptr_vocab

    return vocab, all_output, num_ptr, num_slot, num_intent

def save_model(model, tokenizer, output_dir):
    # Save a trained model and the associated configuration
    # output_dir = os.path.join(output_dir, 'epoch{}'.format(epoch))
    # os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # json.dump(model_config, open(os.path.join(output_dir, "model_config.json"), "w"))

    return

def evaluate(model, examples, eval_sampler, eval_dataloader, id2token, device, n_gpu, output_f, decode=False, output_json=None, use_decode_emb=False):
    eval_loss, nb_eval_steps, nb_eval_examples = 0, 0, 0
    # eval_slot_accuracy, eval_intent_accuracy = 0, 0
    # y_slot_true = []
    # y_slot_pred = []
    # y_intent_true = []
    # y_intent_pred = []
    exact_matches = []
    # y_intent_pred_first = []
    # y_intent_true_first = []

    f = open(output_f, 'w')

    cnt_ex = 0
    if output_json:
        f_out = open(output_json, 'w')
    for idx, i in enumerate(eval_sampler):
        logger.info("{}/{}: {}".format(idx, len(eval_sampler), len(i)))
        batch = eval_dataloader.dataset[i]
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            if len(batch) == 11:
                input_ids, attention_mask, source_mask, token_type_ids, output_ids, \
                                output_ids_y, target_mask, output_mask, ntokens, input_length, output_length = batch
                outputs = model(input_ids=batch[0], attention_mask=batch[1], source_mask=batch[2],
                                token_type_ids=batch[3], output_ids=batch[4],
                                output_ids_y=batch[5], target_mask=batch[6], output_mask=batch[7], ntokens=batch[8],
                                input_length=batch[9], output_length=batch[10], decode=decode)
            else:
                input_ids, attention_mask, source_mask, token_type_ids, output_ids, \
                                output_ids_y, target_mask, output_mask, ntokens, input_length, output_length, input_token_length, span_indices, span_indices_mask, pointer_mask, schema_token_mask = batch
                outputs = model(input_ids=batch[0], attention_mask=batch[1], source_mask=batch[2],
                                token_type_ids=batch[3], output_ids=batch[4],
                                output_ids_y=batch[5], target_mask=batch[6], output_mask=batch[7], ntokens=batch[8],
                                input_length=batch[9], output_length=batch[10], decode=decode,
                                input_token_length=batch[11],
                                span_indices=batch[12], span_indices_mask=batch[13],
                                pointer_mask=batch[14], schema_token_mask=batch[15])
            if decode:
                tmp_eval_loss, pred_ids = outputs
            else:
                tmp_eval_loss = outputs

            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        if decode:
            if use_decode_emb:
                output_ids = output_ids
            else:
                start_symbol = 1
                output_ids = torch.cat([
                    torch.zeros(output_ids_y.size(0), 1, dtype=output_ids_y.dtype).fill_(start_symbol).to(output_ids_y.device), 
                    output_ids_y], axis=1)

            # y_slot_p, y_slot_t = get_slots_info(output_ids, pred_ids, num_slots)
            # y_intent_p, y_intent_t, em, y_intent_p_first, y_intent_t_first = get_intent_n_exact_match_info(output_ids, pred_ids, num_slots, num_intents)
            # y_slot_pred += y_slot_p
            # y_slot_true += y_slot_t
            # y_intent_pred += y_intent_p
            # y_intent_true += y_intent_t
            exact_matches += get_exact_match(output_ids, pred_ids)
            # y_intent_pred_first += y_intent_p_first
            # y_intent_true_first += y_intent_t_first

            for ii in range(output_ids.size(0)):
                example = examples[cnt_ex]
                tokens = example.text_inp.split(" ")

                output_token = []
                output_token_surface = []
                for output_id in output_ids[ii]:        
                    if output_id.item() in [0,1,2]:
                        continue
                    token = id2token[output_id.item()]
                    output_token.append(token)
                    if '@ptr' in token:
                        ptr_id = min(int(token[4:]), len(tokens) - 2)
                        # sometimes the generated number is larger than seq length.
                        # Thus, we constrain the input not exceeding the limit
                        output_token_surface.append(tokens[ptr_id])
                    else:
                        output_token_surface.append(token)

                pred_token = []
                pred_token_surface = []
                for pred_id in pred_ids[ii][0]:
                    if pred_id in [0,1,2]:
                        continue
                    token = id2token[pred_id]
                    pred_token.append(token)
                    if '@ptr' in token:
                        ptr_id = min(int(token[4:]), len(tokens) - 2)
                        # sometimes the generated number is larger than seq length.
                        # Thus, we constrain the input not exceeding the limit
                        pred_token_surface.append(tokens[ptr_id])
                    else:
                        pred_token_surface.append(token)
                if example.db_id:
                    f.write(example.db_id+'\n'+example.text_inp+'\n'+example.text_out+'\n')
                else:
                    f.write(example.text_inp+'\n'+example.text_out+'\n')

                f.write(' '.join(output_token)+'\n')
                f.write(' '.join(output_token_surface)+'\n')
                f.write(' '.join(pred_token)+'\n')
                f.write(' '.join(pred_token_surface)+'\n')                               
                f.write('\n')
                data = {'database_id': example.db_id,
                        'interaction_id': 0,
                        'index_in_interaction': 0,
                        'flat_prediction': pred_token_surface, 
                        'flat_gold_queries': [example.text_out.split()]}
                if output_json:
                    json.dump(data, f_out)
                    f_out.write('\n')
                cnt_ex += 1
    f.close()
    if output_json:
        f_out.close()
    eval_loss = eval_loss / nb_eval_steps

    results = {
        "loss": eval_loss,
        # "accuracy_slots": np.mean(np.array(y_slot_true) == np.array(y_slot_pred)),
        # "accuracy_intents": np.mean(np.array(y_intent_true) == np.array(y_intent_pred)),
        # "accuracy_intent_first": np.mean(np.array(y_intent_true_first) == np.array(y_intent_pred_first)),
        "exact_match": np.mean(np.array(exact_matches)),
    }

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MTOP')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--eval_on', type=str, default='test.py')

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--overwrite_output_vocab', default=False, action="store_true")
    parser.add_argument('--output_vocab', type=str)

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--wandb_project', type=str)

    parser.add_argument('--bert_load_path', type=str, default='')
    parser.add_argument('--bert_model', type=str, default='bert-base-cased')
    parser.add_argument('--do_lower_case', type=bool, default=True)

    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--use_decode_emb', type=int, default=1)
    parser.add_argument('--use_avg_span_extractor', type=int, default=1)
    parser.add_argument('--use_schema_token_mask', type=int, default=1)

    parser.add_argument('--decoder_layers', type=int, default=6)
    parser.add_argument('--decoder_size', type=int, default=2048)
    parser.add_argument('--decoder_heads', type=int, default=8)
    parser.add_argument('--decoder_dropout', type=float, default=0.1)

    parser.add_argument('--random_init', type=bool, default=False)

    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--dev_batch_size', type=int, default=32)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=8)
    parser.add_argument('--max_seq_length', type=int, default=128)

    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--bert_lr', type=float, default=3e-6)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    ########################## Input Parameters ###########################
    dataset = args.dataset
    data_dir = args.data_dir
    bert_load_path = args.bert_load_path \
        if len(args.bert_load_path) else args.bert_model
    bert_model = args.bert_model
    do_lower_case = args.do_lower_case
    train_batch_size = args.train_batch_size
    dev_batch_size = args.dev_batch_size
    per_gpu_eval_batch_size = args.per_gpu_eval_batch_size
    num_train_epochs = args.num_train_epochs
    warmup_proportion = args.warmup_proportion
    learning_rate = args.learning_rate
    adam_epsilon = args.adam_epsilon
    weight_decay = args.weight_decay
    local_rank = args.local_rank
    max_seq_length = args.max_seq_length
    max_grad_norm = args.max_grad_norm
    output_dir = args.output_dir
    output_vocab_f = os.path.join(args.data_dir, 'output_vocab.txt')
    if args.overwrite_output_vocab:
        output_vocab_f = args.output_vocab # overwrite the vocab to multilingual version so that we can do cross lingual
    #######################################################################
    if args.overwrite_output_vocab:
        print("Overwrite output vocab using:", output_vocab_f)
    if bert_load_path[0] == '/' or bert_load_path == '\\':
        print("Loading Model from:", bert_load_path)

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name='lr{}_batch{}'.format(learning_rate, train_batch_size))
    wandb.config.learning_rate = learning_rate
    wandb.config.train_batch_size = train_batch_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, "debug.log")))
    logger.info(args)

    # Load config and model for both eval and train
    if bert_model in ['bert-base-cased', 'bert-base-multilingual-cased']:
        tokenizer = BertTokenizer.from_pretrained(bert_load_path, do_lower_case=do_lower_case)
        cls_token_segment_id = 1
    elif bert_model in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(bert_load_path, add_prefix_space=True)
        cls_token_segment_id = 0
    elif bert_model in ['xlm-roberta-base', 'xlm-roberta-large']:
        tokenizer = XLMRobertaTokenizer.from_pretrained(bert_load_path)
        cls_token_segment_id = 0
    elif bert_model in ['facebook/bart-base', 'facebook/bart-large']:
        tokenizer = BartTokenizer.from_pretrained(bert_load_path, add_prefix_space=True)
        cls_token_segment_id = 0
    elif bert_model in ['facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
        tokenizer = MBart50Tokenizer.from_pretrained(bert_load_path)
        cls_token_segment_id = 0
    elif bert_model in ['t5-large', 't5-base', 't5-small']:
        tokenizer = T5Tokenizer.from_pretrained(bert_load_path)
        cls_token_segment_id = 0
    elif bert_model in ['google/mt5-large']:
        tokenizer = MT5Tokenizer.from_pretrained(bert_load_path)
        cls_token_segment_id = 0

    logger.info('num_special_tokens_to_add {}'.format(tokenizer.num_special_tokens_to_add()))

    output_vocab, all_outputs, num_ptrs, num_slots, num_intents = read_output_vocab(tokenizer, output_vocab_f, args.dataset)
    # logger.info(output_vocab)
    # logger.info(all_outputs)
    logger.info('num_ptrs: {}, num_slots: {}, num_intents: {}'.format(num_ptrs, num_slots, num_intents))
    outputs_map = {word: i for i, word in enumerate(all_outputs)}
    id2token = {i: word for i, word in enumerate(all_outputs)}
    # processor = DataProcessor(output_vocab)
    # label_list = processor.get_labels()
    # num_labels = len(label_list)
    # label_map = {i : label for i, label in enumerate(label_list)}
    # logger.info('num_labels: {}'.format(num_labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # For debug:
    # device = "cpu"
    # n_gpu = 0


    if bert_model in ['bert-base-cased', 'bert-base-multilingual-cased']:
        config = BertConfig.from_pretrained(bert_load_path)
    elif bert_model in ['roberta-base', 'roberta-large']:
        config = RobertaConfig.from_pretrained(bert_load_path)
    elif bert_model in ['xlm-0-base', 'xlm-roberta-large']:
        config = XLMRobertaConfig.from_pretrained(bert_load_path)
    elif bert_model in ['facebook/bart-base', 'facebook/bart-large']:
        config = BartConfig.from_pretrained(bert_load_path)
    elif bert_model in ['facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
        config = MBartConfig.from_pretrained(bert_load_path)
    elif bert_model in ['t5-large', 't5-base', 't5-small']:
        config = T5Config.from_pretrained(bert_load_path)
    elif bert_model in ['google/mt5-large']:
        config = MT5Config.from_pretrained(bert_load_path)

    config.bert_model = bert_model
    config.num_ptrs = num_ptrs
    config.all_outputs = all_outputs
    config.output_vocab = output_vocab
    config.decoder_dropout = args.decoder_dropout

    config.smoothing = args.smoothing
    config.use_decode_emb = (args.use_decode_emb == 1)
    config.use_avg_span_extractor = (args.use_avg_span_extractor == 1)
    config.use_schema_token_mask = (args.use_schema_token_mask == 1)

    if bert_model in ['facebook/bart-base', 'facebook/bart-large', 'facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
        if args.random_init:
            model = PtrBART(config)
        else:
            model, loading_info = PtrBART.from_pretrained(
                        bert_load_path, config=config, output_loading_info=True, ignore_mismatched_sizes=True)
    elif bert_model in ['t5-large', 't5-base', 't5-small', 'google/mt5-large']:
        if args.random_init:
            model = PtrT5(config)
        else:
            model, loading_info = PtrT5.from_pretrained(
                        bert_load_path, config=config, output_loading_info=True, ignore_mismatched_sizes=True)
    else: # include other model AND local models
        config.decoder_layers = args.decoder_layers
        config.decoder_size = args.decoder_size
        config.decoder_heads = args.decoder_heads
        model, loading_info = PtrRoberta.from_pretrained(
                    bert_load_path, config=config, output_loading_info=True, ignore_mismatched_sizes=True)
    # logger.info('loading_info')
    # logger.info('missing_keys: {}'.format(loading_info['missing_keys']))
    # logger.info('unexpected_keys: {}'.format(loading_info['unexpected_keys']))
    # logger.info('error_msgs: {}'.format(loading_info['error_msgs']))

    ###### Device and Parallel
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)


    if args.mode == 'train':
        model.train()

        #### Train Features and DataLoader
        print('Loading Train')
        train_sampler, train_dataloader, train_examples = create_sampler_dataloader(
            config.use_decode_emb,
            output_vocab, data_dir, 'train', train_batch_size,
            tokenizer, cls_token_segment_id, outputs_map, max_seq_length, local_rank)
        train_num = len(train_examples)
        num_train_optimization_steps = math.ceil(train_num / train_batch_size) * num_train_epochs
        #### Dev Features and DataLoader
        print('Loading Dev')
        if "dev" in args.eval_on:
            dev_split = 'dev'
        else:
            dev_split = 'test.py'
        dev_sampler, dev_dataloader, dev_examples = create_sampler_dataloader(
            config.use_decode_emb,
            output_vocab, data_dir, dev_split, dev_batch_size,
            tokenizer, cls_token_segment_id, outputs_map, max_seq_length)
        print('Done')

        #### Optimizer
        param_optimizer = list(model.named_parameters())
        # for n, p in param_optimizer:
        #     logger.info('{}, {}, {}'.format(n, p.size(), p.requires_grad))
        no_decay = ['bias','LayerNorm.weight','norm.a_2', 'norm.b_2']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     ]
        optimizer_grouped_parameters = [
            {'params': [], 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': [], 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': [], 'weight_decay': weight_decay, 'lr': args.bert_lr},
            {'params': [], 'weight_decay': 0.0, 'lr': args.bert_lr},
            ]
        for n, p in param_optimizer: 
            if any(nd in n for nd in no_decay):
                if n.startswith(model.base_model_prefix):
                    # no decay, bert
                    optimizer_grouped_parameters[3]['params'].append(p)
                else:
                    # no decay, not bert
                    optimizer_grouped_parameters[1]['params'].append(p)
            else:
                if n.startswith(model.base_model_prefix):
                    # decay, bert
                    optimizer_grouped_parameters[2]['params'].append(p)
                else:
                    # decay, not bert
                    optimizer_grouped_parameters[0]['params'].append(p)
        
        # for g in optimizer_grouped_parameters:
        #     print(len(g['params']), g['lr'], g['weight_decay'])
        # exit()
        if args.optimizer == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters, eps=adam_epsilon)
            warmup_steps = int(warmup_proportion * num_train_optimization_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_train_optimization_steps)
        elif args.optimizer == 'Adafactor':
            optimizer = Adafactor(optimizer_grouped_parameters, eps=(1e-30, 1e-3), clip_threshold=1.0, beta1=0.0, scale_parameter=False, relative_step=False, warmup_init=False)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_num)
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num epochs = %d", num_train_epochs)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        global_step = 0
        best_model = None
        best_epoch = None
        best_dev_loss = 1e6
        epoch = 0
        for _ in trange(int(num_train_epochs), desc="Epoch"):
            output_f = os.path.join(output_dir, 'dev_output.txt')
            dev_results = evaluate(model, dev_examples, dev_sampler, dev_dataloader, id2token, device, n_gpu, output_f, decode=False, use_decode_emb=config.use_decode_emb)
            dev_loss = dev_results['loss']
            logger.info('Epoch: {}, Dev loss: {}'.format(epoch, dev_loss))

            wandb.log({'dev_loss': dev_loss})

            # dev_results = evaluate(model, dev_examples, dev_sampler, dev_dataloader, id2token, num_slots, num_intents, device, n_gpu, output_f, decode=False)
            # dev_loss = dev_results['loss']
            # logger.info('Epoch: {}, Dev loss: {}'.format(epoch, dev_loss))

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                # Save Model
                # model_config = {"bert_model":bert_model,
                #                 "do_lower":do_lower_case, 
                #                 "max_seq_length":max_seq_length,
                #                 # "num_labels":len(label_list), 
                #                 # "label_map":label_map,
                #                 "num_ptrs": num_ptrs,
                #                 "all_outputs": all_outputs,
                #                 "output_vocab": output_vocab,
                #                 'decoder_layers': args.decoder_layers,
                #                 'decoder_size': args.decoder_size,
                #                 'decoder_heads': args.decoder_heads,
                #                 'decoder_dropout': args.decoder_dropout,
                #                 "epoch": epoch}
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                save_model(model, tokenizer, output_dir)

            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            train_sampler.shuffle()
            for idx, i in enumerate(train_sampler):
                batch = train_dataloader.dataset[i]
                batch = tuple(t.to(device) for t in batch)
                input_ids = batch[0]
                # input_ids, attention_mask, source_mask, token_type_ids, output_ids, \
                #                         output_ids_y, output_mask, ntokens, input_length, output_length = batch
                # print(batch[1].size())
                # print(batch[2].size())
                # print(batch[6].size())
                # print(batch[7].size())
                # exit()
                if len(batch) == 11:
                    outputs = model(input_ids=batch[0], attention_mask=batch[1], source_mask=batch[2],
                                    token_type_ids=batch[3], output_ids=batch[4],
                                    output_ids_y=batch[5], target_mask=batch[6], output_mask=batch[7], ntokens=batch[8],
                                    input_length=batch[9], output_length=batch[10])
                else:
                    outputs = model(input_ids=batch[0], attention_mask=batch[1], source_mask=batch[2],
                                    token_type_ids=batch[3], output_ids=batch[4],
                                    output_ids_y=batch[5], target_mask=batch[6], output_mask=batch[7], ntokens=batch[8],
                                    input_length=batch[9], output_length=batch[10],
                                    input_token_length=batch[11],
                                    span_indices=batch[12], span_indices_mask=batch[13],
                                    pointer_mask=batch[14], schema_token_mask=batch[15])

                loss = outputs
                # if (idx%100==0):
                #     print(loss)
                #     logger.info("Loss at iteration %d = %.4f", idx, loss.item())
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.optimizer == 'AdamW':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                elif args.optimizer == 'Adafactor':
                    optimizer.step()
                model.zero_grad()
                global_step += 1

            logger.info("Average loss at the end of Epoch = %.4f", tr_loss / nb_tr_steps)
            wandb.log({'train_loss': tr_loss / nb_tr_steps})
            epoch += 1

    logger.info('EVAL')
    if args.mode=='train':
        logger.info('1')
        dev_results = evaluate(best_model, dev_examples, dev_sampler, dev_dataloader, id2token, device, n_gpu, output_f, decode=False, use_decode_emb=config.use_decode_emb)
        dev_loss = dev_results['loss']
        logger.info('Epoch: {}, Dev loss: {}'.format(best_epoch, dev_loss))

        # logger.info('2')
        # dev_results = evaluate(best_model, dev_examples, dev_sampler, dev_dataloader, id2token, num_slots, num_intents, device, n_gpu, output_f, decode=True)
        # dev_loss = dev_results['loss']
        # logger.info('Epoch: {}, Dev loss: {}'.format(best_epoch, dev_loss))

    # Load the best model
    # with open(os.path.join(output_dir, "model_config.json")) as f:
    #     model_config = json.load(f)

    if bert_model in ['bert-base-cased', 'bert-base-multilingual-cased']:
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)
        PtrRoberta.config_class = BertConfig
    elif bert_model in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(output_dir, add_prefix_space=True)
        PtrRoberta.config_class = RobertaConfig
    elif bert_model in ['xlm-roberta-base', 'xlm-roberta-large']:
        tokenizer = XLMRobertaTokenizer.from_pretrained(output_dir)
        PtrRoberta.config_class = XLMRobertaConfig
    elif bert_model in ['facebook/bart-base', 'facebook/bart-large']:
        tokenizer = BartTokenizer.from_pretrained(output_dir, add_prefix_space=True)
        PtrBART.config_class = BartConfig
    elif bert_model in ['facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
        tokenizer = MBart50Tokenizer.from_pretrained(output_dir)
        PtrBART.config_class = MBartConfig
    elif bert_model in ['t5-large', 't5-base', 't5-small']:
        tokenizer = T5Tokenizer.from_pretrained(output_dir)
        PtrT5.config_class = T5Config
    elif bert_model in ['google/mt5-large']:
        tokenizer = MT5Tokenizer.from_pretrained(output_dir)
        PtrT5.config_class = MT5Config

    if bert_model in ['facebook/bart-base', 'facebook/bart-large', 'facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
        model, loading_info = PtrBART.from_pretrained(output_dir, output_loading_info=True)
    elif bert_model in ['t5-large', 't5-base', 't5-small', 'google/mt5-large']:
        model, loading_info = PtrT5.from_pretrained(output_dir, output_loading_info=True)
    else:
        # model, loading_info = PtrRoberta.from_pretrained(
        #             output_dir, bert_model=bert_model, num_ptrs=model_config['num_ptrs'],
        #             all_outputs=model_config['all_outputs'], output_vocab=model_config['output_vocab'],
        #             decoder_layers=model_config['decoder_layers'], decoder_size=model_config['decoder_size'], decoder_heads=model_config['decoder_heads'], decoder_dropout=model_config['decoder_dropout'],
        #             output_loading_info=True)
        model, loading_info = PtrRoberta.from_pretrained(output_dir, output_loading_info=True)
    logger.info('loading_info')
    logger.info('missing_keys: {}'.format(loading_info['missing_keys']))
    logger.info('unexpected_keys: {}'.format(loading_info['unexpected_keys']))
    logger.info('error_msgs: {}'.format(loading_info['error_msgs']))
    
    model.to(device)
    # multi-gpu evaluate
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # logger.info('3')
    # dev_results = evaluate(model, dev_sampler, dev_dataloader, id2token, num_slots, num_intents, device, n_gpu, output_f, decode=False)
    # dev_loss = dev_results['loss']
    # logger.info('Epoch: {}, Dev loss: {}'.format(epoch, dev_loss))

    # print('Start Eval')
    if "dev" in args.eval_on:
        data_sampler, data_dataloader, dev_examples = create_sampler_dataloader(
            config.use_decode_emb,
            output_vocab, data_dir, 'dev', dev_batch_size,
            tokenizer, cls_token_segment_id, outputs_map, max_seq_length, sort=False)
        output_f = os.path.join(output_dir, 'dev_output.txt')
        output_json = os.path.join(output_dir, 'dev_output.json')
        dev_num = len(dev_examples)

        # logger.info('4')
        # dev_results = evaluate(model, dev_examples, data_sampler, data_dataloader, id2token, device, n_gpu, output_f, decode=False, use_decode_emb=config.use_decode_emb)
        # logger.info('Dev Num: {}'.format(dev_num))
        # logger.info("Dev Results: ")
        # logger.info(dev_results)

        logger.info('Eval on Dev')
        dev_results = evaluate(model, dev_examples, data_sampler, data_dataloader, id2token, device, n_gpu, output_f, output_json=output_json, decode=True, use_decode_emb=config.use_decode_emb)
        logger.info('Dev Num: {}'.format(dev_num))
        logger.info("Dev Results: ")
        logger.info(dev_results)

        if dataset == 'MSPIDER':
            eval_cmd = 'python postprocess_eval.py --dataset=spider --split=dev --pred_file {} --remove_from'.format(output_json)
            subprocess.run(eval_cmd, shell=True)

        wandb.log({'dev_exact_match': dev_results['exact_match']})
    
    if "test.py" in args.eval_on:
        data_sampler, data_dataloader, test_examples = create_sampler_dataloader(
            config.use_decode_emb,
            output_vocab, data_dir, 'test.py', dev_batch_size,
            tokenizer, cls_token_segment_id, outputs_map, max_seq_length, sort=False)
        output_f = os.path.join(output_dir, 'test_output.txt')
        test_num = len(test_examples)
        test_results = evaluate(model, test_examples, data_sampler, data_dataloader, id2token, device, n_gpu, output_f, decode=True, use_decode_emb=config.use_decode_emb)
        logger.info('Eval on Test')
        logger.info('Test Num: {}'.format(test_num))
        logger.info("Test Results: ")
        logger.info(test_results)

        wandb.log({'test_exact_match': test_results['exact_match']})

    return

if __name__ == '__main__':
    main()
