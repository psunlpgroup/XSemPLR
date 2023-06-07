import argparse
import random, os, csv, logging, json, copy, math, operator
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

from transformers import (BertForSequenceClassification, BertTokenizer, BertConfig, 
                          BertPreTrainedModel, BertModel, AdamW, get_linear_schedule_with_warmup)

from src.data import DataProcessor, semParse_convert_examples_to_features, BucketSampler
# from src.ptrbert import PtrBert

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def get_exact_match(output_ids, pred_ids):
    output_ids = output_ids.detach().cpu().numpy()
    # y_intent_p = []
    # y_intent_t = []
    exact_match_res = []
    # y_intent_p_first = []
    # y_intent_t_first = []
    for i in range(output_ids.shape[0]):
        # intent_p = []
        # intent_t = []
        # y_intent_p_first.append(pred_ids[i][0][1])
        # y_intent_t_first.append(output_ids[i][1])
        for j in range(output_ids.shape[1]):
            if (output_ids[i][j]==2): 
                if (np.all(pred_ids[i][0][:j+1] == output_ids[i,:j+1])): exact_match_res.append(1)
                else: exact_match_res.append(0)
                break
            # elif (output_ids[i][j]<num_slots+num_intents+3 and output_ids[i][j]>=num_slots+3):
            #     intent_t.append(output_ids[i][j])
            #     if (j>=len(pred_ids[i][0])):
            #         # print("Terminated early. Appending -1")
            #         # print(output_ids[i][:j+5])
            #         # print(pred_ids[i][0])
            #         intent_p.append(-1)
            #     else:
            #         intent_p.append(pred_ids[i][0][j])
        # if (len(intent_p) > 0):
        #     y_intent_p += intent_p
        #     y_intent_t += intent_t
    return exact_match_res
    # (y_intent_p, y_intent_t, exact_match_res, y_intent_p_first, y_intent_t_first)


# MTOP
def evaluate(model, examples, eval_sampler, eval_dataloader, id2token, num_slots, num_intents, device, n_gpu, output_f, decode=False, output_json=None, use_decode_emb=False):
    eval_loss, nb_eval_steps, nb_eval_examples = 0, 0, 0
    eval_slot_accuracy, eval_intent_accuracy = 0, 0
    y_slot_true = []
    y_slot_pred = []
    y_intent_true = []
    y_intent_pred = []
    exact_matches = []
    y_intent_pred_first = []
    y_intent_true_first = []

    f = open(output_f, 'w')

    # for idx, i in enumerate(tqdm(eval_sampler, desc="Evaluating")):
    cnt_ex = 0
    if output_json:
        f_out = open(output_json, 'w')
    for idx, i in enumerate(eval_sampler):
        # logger.info("{}/{}: {}".format(idx, len(eval_sampler), i))
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

            y_slot_p, y_slot_t = get_slots_info(output_ids, pred_ids, num_slots)
            y_intent_p, y_intent_t, em, y_intent_p_first, y_intent_t_first = get_intent_n_exact_match_info(output_ids, pred_ids, num_slots, num_intents)
            y_slot_pred += y_slot_p
            y_slot_true += y_slot_t
            y_intent_pred += y_intent_p
            y_intent_true += y_intent_t
            exact_matches += em
            y_intent_pred_first += y_intent_p_first
            y_intent_true_first += y_intent_t_first

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
                        output_token_surface.append(tokens[int(token[4:])])
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
                        pred_token_surface.append(tokens[int(token[4:])])
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
        "accuracy_slots": np.mean(np.array(y_slot_true) == np.array(y_slot_pred)),
        "accuracy_intents": np.mean(np.array(y_intent_true) == np.array(y_intent_pred)),
        "accuracy_intent_first": np.mean(np.array(y_intent_true_first) == np.array(y_intent_pred_first)),
        "exact_match": np.mean(np.array(exact_matches)),
    }

    return results

def evaluate(model, examples, eval_sampler, eval_dataloader, id2token, num_slots, num_intents, device, n_gpu, output_f, decode=False, output_json=None):
    eval_loss, nb_eval_steps, nb_eval_examples = 0, 0, 0
    eval_slot_accuracy, eval_intent_accuracy = 0, 0
    y_slot_true = []
    y_slot_pred = []
    y_intent_true = []
    y_intent_pred = []
    exact_matches = []
    y_intent_pred_first = []
    y_intent_true_first = []

    f = open(output_f, 'w')

    # for idx, i in enumerate(tqdm(eval_sampler, desc="Evaluating")):
    cnt_ex = 0
    if output_json:
        f_out = open(output_json, 'w')
    for idx, i in enumerate(eval_sampler):
        # logger.info("{}/{}: {}".format(idx, len(eval_sampler), i))
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
            start_symbol = 1
            output_ids = torch.cat([
                torch.zeros(output_ids_y.size(0), 1, dtype=output_ids_y.dtype).fill_(start_symbol).to(output_ids_y.device), 
                output_ids_y], axis=1)

            y_slot_p, y_slot_t = get_slots_info(output_ids, pred_ids, num_slots)
            y_intent_p, y_intent_t, em, y_intent_p_first, y_intent_t_first = get_intent_n_exact_match_info(output_ids, pred_ids, num_slots, num_intents)
            y_slot_pred += y_slot_p
            y_slot_true += y_slot_t
            y_intent_pred += y_intent_p
            y_intent_true += y_intent_t
            exact_matches += em
            y_intent_pred_first += y_intent_p_first
            y_intent_true_first += y_intent_t_first

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
                        output_token_surface.append(tokens[int(token[4:])])
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
                        pred_token_surface.append(tokens[int(token[4:])])
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
        "accuracy_slots": np.mean(np.array(y_slot_true) == np.array(y_slot_pred)),
        "accuracy_intents": np.mean(np.array(y_intent_true) == np.array(y_intent_pred)),
        "accuracy_intent_first": np.mean(np.array(y_intent_true_first) == np.array(y_intent_pred_first)),
        "exact_match": np.mean(np.array(exact_matches)),
    }

    return results

def evaluate(model, examples, eval_sampler, eval_dataloader, id2token, num_slots, num_intents, device, n_gpu, output_f, decode=False, output_json=None):
    eval_loss, nb_eval_steps, nb_eval_examples = 0, 0, 0
    eval_slot_accuracy, eval_intent_accuracy = 0, 0
    y_slot_true = []
    y_slot_pred = []
    y_intent_true = []
    y_intent_pred = []
    exact_matches = []
    y_intent_pred_first = []
    y_intent_true_first = []

    f = open(output_f, 'w')

    # for idx, i in enumerate(tqdm(eval_sampler, desc="Evaluating")):
    cnt_ex = 0
    if output_json:
        f_out = open(output_json, 'w')
    for idx, i in enumerate(eval_sampler):
        logger.info("{}/{}: {}".format(idx, len(eval_sampler), i))
        batch = eval_dataloader.dataset[i]
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            input_ids, attention_mask, source_mask, token_type_ids, output_ids, \
                            output_ids_y, target_mask, output_mask, ntokens, input_length, output_length, input_token_length, span_indices, span_indices_mask, pointer_mask = batch
            outputs = model(input_ids=batch[0], attention_mask=batch[1], source_mask=batch[2],
                            token_type_ids=batch[3], output_ids=batch[4],
                            output_ids_y=batch[5], target_mask=batch[6], output_mask=batch[7], ntokens=batch[8],
                            input_length=batch[9], output_length=batch[10], decode=decode,
                            input_token_length=batch[11],
                            span_indices=batch[12], span_indices_mask=batch[13],
                            pointer_mask=batch[14])
            if decode:
                tmp_eval_loss, pred_ids = outputs
            else:
                tmp_eval_loss = outputs
                # pred_ids = [[[0]] for j in range(len(i))]
            
            # print('outputs')
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        if decode:
            start_symbol = 1
            output_ids = torch.cat([
                torch.zeros(output_ids_y.size(0), 1, dtype=output_ids_y.dtype).fill_(start_symbol).to(output_ids_y.device), 
                output_ids_y], axis=1)

            y_slot_p, y_slot_t = get_slots_info(output_ids, pred_ids, num_slots)
            y_intent_p, y_intent_t, em, y_intent_p_first, y_intent_t_first = get_intent_n_exact_match_info(output_ids, pred_ids, num_slots, num_intents)
            y_slot_pred += y_slot_p
            y_slot_true += y_slot_t
            y_intent_pred += y_intent_p
            y_intent_true += y_intent_t
            exact_matches += em
            y_intent_pred_first += y_intent_p_first
            y_intent_true_first += y_intent_t_first

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
                        output_token_surface.append(tokens[int(token[4:])])
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
                        pred_token_surface.append(tokens[int(token[4:])])
                    else:
                        pred_token_surface.append(token)

                f.write(example.db_id+'\n'+example.text_inp+'\n'+example.text_out+'\n')
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
                json.dump(data, f_out)
                f_out.write('\n')
                cnt_ex += 1
    f.close()
    if output_json:
        f_out.close()
    eval_loss = eval_loss / nb_eval_steps

    results = {
        "loss": eval_loss,
        "accuracy_slots": np.mean(np.array(y_slot_true) == np.array(y_slot_pred)),
        "accuracy_intents": np.mean(np.array(y_intent_true) == np.array(y_intent_pred)),
        "accuracy_intent_first": np.mean(np.array(y_intent_true_first) == np.array(y_intent_pred_first)),
        "exact_match": np.mean(np.array(exact_matches)),
    }

    return results

def get_slots_info(output_ids, pred_ids, num_slots):
    output_ids = output_ids.detach().cpu().numpy()
    y_slot_p = []
    y_slot_t = []
    for i in range(output_ids.shape[0]):
        slot_p = []
        slot_t = []
        for j in range(output_ids.shape[1]):
            if (output_ids[i][j]==2): break
            elif (output_ids[i][j]<num_slots+3 and output_ids[i][j]>2):
                slot_t.append(output_ids[i][j])
                if (j>=len(pred_ids[i][0])):
                    # print("Terminated early. Appending -1")
                    # print(output_ids[i][:j+5])
                    # print(pred_ids[i][0])
                    slot_p.append(-1)
                else:
                    slot_p.append(pred_ids[i][0][j])
        if (len(slot_p) > 0):
            y_slot_p += slot_p
            y_slot_t += slot_t
    return (y_slot_p, y_slot_t)

def get_intent_n_exact_match_info(output_ids, pred_ids, num_slots, num_intents):
    output_ids = output_ids.detach().cpu().numpy()
    y_intent_p = []
    y_intent_t = []
    exact_match_res = []
    y_intent_p_first = []
    y_intent_t_first = []
    for i in range(output_ids.shape[0]):
        intent_p = []
        intent_t = []
        y_intent_p_first.append(pred_ids[i][0][1])
        y_intent_t_first.append(output_ids[i][1])
        for j in range(output_ids.shape[1]):
            if (output_ids[i][j]==2): 
                if (np.all(pred_ids[i][0][:j+1] == output_ids[i,:j+1])): exact_match_res.append(1)
                else: exact_match_res.append(0)
                break
            elif (output_ids[i][j]<num_slots+num_intents+3 and output_ids[i][j]>=num_slots+3):
                intent_t.append(output_ids[i][j])
                if (j>=len(pred_ids[i][0])):
                    # print("Terminated early. Appending -1")
                    # print(output_ids[i][:j+5])
                    # print(pred_ids[i][0])
                    intent_p.append(-1)
                else:
                    intent_p.append(pred_ids[i][0][j])
        if (len(intent_p) > 0):
            y_intent_p += intent_p
            y_intent_t += intent_t
    return (y_intent_p, y_intent_t, exact_match_res, y_intent_p_first, y_intent_t_first)