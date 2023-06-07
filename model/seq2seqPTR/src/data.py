import argparse
import random, os, csv, logging, json, copy, math, operator
from dataclasses import dataclass

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Sampler)
from .utils import subsequent_mask

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from data_preprocess import read_database_schema

def convert_output_to_ids_spider(output, outputs_map):
    output_ids = []
    output_ids_y = []
    pointer_mask = []
    for w in output:
        output_ids_y.append(outputs_map[w])
        if '@ptr' in w:
            # output_ids.append(int(w[4:])+1) # TODO: we need to account for [CLS]
            output_ids.append(int(w[4:]))
            pointer_mask.append(1)
        else:
            output_ids.append(outputs_map[w])
            pointer_mask.append(0)
    return output_ids, output_ids_y, pointer_mask

def convert_output_to_ids(output, outputs_map):
    output_ids = []
    for w in output:
        output_ids.append(outputs_map[w])
    return output_ids

@dataclass(frozen=True)
class InputExample:
    """
    A single training/test.py example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test.py examples.
    """

    guid: str
    text_inp: str
    text_out: str
    db_id: str

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"

def sort_inp_len(inpex):
    return len(inpex.text_inp.split(" "))

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, source_mask=None,
                       token_type_ids=None, output_ids=None,
                       output_ids_y=None, target_mask=None, output_mask=None, ntokens=None,
                       input_length=None, output_length=None,
                       input_token_length=None,
                       span_indices=None, span_indices_mask=None,
                       pointer_mask=None, schema_token_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.source_mask = source_mask
        self.token_type_ids = token_type_ids
        self.output_ids = output_ids
        self.output_ids_y = output_ids_y
        self.target_mask = target_mask
        self.output_mask = output_mask
        self.ntokens = ntokens
        self.input_length = input_length
        self.output_length = output_length
        self.input_token_length = input_token_length
        self.span_indices = span_indices
        self.span_indices_mask = span_indices_mask
        self.pointer_mask = pointer_mask
        self.schema_token_mask = schema_token_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor(object):

    def __init__(self, output_vocab):
        self.output_vocab = output_vocab

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["input"].numpy().decode("utf-8"),
            str(tensor_dict["output"].numpy().decode("utf-8")),
        )

    def get_labels(self):
        return self.output_vocab

    def _create_examples(self, data_dir, set_type, sort=True):
        """Creates examples for the training and dev sets."""
        examples = []
        input_file_data = os.path.join(data_dir, "{}.tsv".format(set_type))
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            for i, inp in enumerate(f):
                inps = inp.split('\t')
                guid = "%s-%s" % (set_type, i)
                text_inp = inps[1].strip()
                text_out = inps[2].strip()
                if len(inps) == 4:
                    db_id = inps[3].strip()
                else:
                    db_id = ''
                examples.append(InputExample(guid=guid, text_inp=text_inp, text_out=text_out, db_id=db_id))
                
            # Sort these out before returning
            if sort:
                examples = sorted(examples, key=sort_inp_len)
            return examples


def create_sampler_dataloader(use_decode_emb, output_vocab, data_dir, split, batch_size, tokenizer, cls_token_segment_id, outputs_map, max_seq_length, sort=True, local_rank=-1):
    processor = DataProcessor(output_vocab)
    data_examples = processor._create_examples(data_dir, set_type=split, sort=sort)
    if 'spider' in data_dir:
        data_features, tensor_dataset = spider_convert_examples_to_features(
            use_decode_emb,
            examples=data_examples, tokenizer=tokenizer, outputs_map=outputs_map,
            output_vocab=output_vocab,
            max_seq_length=max_seq_length,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_on_left=False, 
            pad_token_segment_id=0,
            cls_token_segment_id=cls_token_segment_id)
    # elif ('mtop' in data_dir or 'mgeoquery' in data_dir) and 'old' not in data_dir:
    elif 'old' not in data_dir:
        data_features, tensor_dataset = mtop_convert_examples_to_features(
            use_decode_emb,
            examples=data_examples, tokenizer=tokenizer, outputs_map=outputs_map,
            output_vocab=output_vocab,
            max_seq_length=max_seq_length,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_on_left=False, 
            pad_token_segment_id=0,
            cls_token_segment_id=cls_token_segment_id)
    else:
        print('unknown datsets', data_dir)
        exit()
        data_features = semParse_convert_examples_to_features(
            examples=data_examples, tokenizer=tokenizer, outputs_map=outputs_map,
            max_seq_length=max_seq_length,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_on_left=False,
            pad_token_segment_id=0,
            cls_token_segment_id=cls_token_segment_id)

        all_input_ids = torch.tensor([f.input_ids for f in data_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in data_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in data_features], dtype=torch.bool)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in data_features], dtype=torch.long)
        all_output_ids = torch.tensor([f.output_ids for f in data_features], dtype=torch.long)
        all_output_ids_y = torch.tensor([f.output_ids_y for f in data_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in data_features], dtype=torch.bool)
        all_output_mask = torch.tensor([f.output_mask for f in data_features], dtype=torch.long)
        all_ntokens = torch.tensor([f.ntokens for f in data_features], dtype=torch.long)
        all_input_length = torch.tensor([f.input_length for f in data_features], dtype=torch.long)
        all_output_length = torch.tensor([f.output_length for f in data_features], dtype=torch.long)

        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_source_mask, all_token_type_ids,
                                    all_output_ids, all_output_ids_y, all_target_mask, all_output_mask, all_ntokens,
                                    all_input_length, all_output_length)
    # eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    # data_sampler =  BucketSampler(tensor_dataset, batch_size)#SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)

    if local_rank == -1:
        data_sampler = BucketSampler(tensor_dataset, batch_size)
    else:
        data_sampler = DistributedSampler(tensor_dataset)

    data_dataloader = DataLoader(tensor_dataset, sampler=data_sampler, batch_size=batch_size)

    return data_sampler, data_dataloader, data_examples


def semParse_convert_examples_to_features(
        examples,
        tokenizer,
        outputs_map,
        max_seq_length=512,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True):

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = example.text_inp.split(" ")
        output = example.text_out.split(" ")
        assert len(output)<=max_seq_length, "Length of output is larger than max_seq_length :: " + example.text_out

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        output += [sep_token]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            output += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            output = [cls_token] + output
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        output_ids = convert_output_to_ids(output, outputs_map)

        input_length = len(input_ids)
        output_length = len(output_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        output_mask = [1 if mask_padding_with_zero else 0] * len(output_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        padding_length_output = max_seq_length - len(output_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            output_ids = ([0] * padding_length_output) + output_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            output_mask = ([0 if mask_padding_with_zero else 1] * padding_length_output) + output_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            # pad input with pad_token
            input_ids += [pad_token] * padding_length
            # pad output always with 0
            output_ids += [0] * padding_length_output
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            output_mask += [0 if mask_padding_with_zero else 1] * padding_length_output
            segment_ids += [pad_token_segment_id] * padding_length

        #################
        inp_ids = torch.tensor(input_ids)
        out_ids = torch.tensor(output_ids)
        source_mask = (inp_ids != pad_token).unsqueeze(-2)   # size = [1, max_seq_length], dtype=bool
        trg = out_ids[:-1]
        trg_y = out_ids[1:]
        target_mask = (trg != 0).unsqueeze(-2)               # size = [1, max_seq_length-1], dytpe=bool
        target_mask = target_mask & Variable(subsequent_mask(trg.size(-1)).type_as(target_mask.data)) # size = [1, max_seq_length-1, max_seq_length-1], dytpe=bool
        target_mask = target_mask.squeeze(0) # size = [max_seq_length-1, max_seq_length-1], dytpe=bool

        source_mask = source_mask.data.numpy()
        output_ids_ = trg.data.numpy()
        output_ids_y = trg_y.data.numpy()
        target_mask = target_mask.data.numpy()
        ntokens = (trg_y!=0).data.sum().item()
        #################

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(output_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(output_ids) == max_seq_length

        # input_mask: [max_seq_length], int
        # source_mask: [1, max_seq_length], bool
        # output_mask: [max_seq_length], int
        # target_mask: [max_seq_length-1, max_seq_length-1], bool
        # print(len(input_mask))
        # print(source_mask.shape)
        # print(len(output_mask))
        # print(target_mask.shape)
        # exit()

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=input_mask, source_mask=source_mask,
                          token_type_ids=segment_ids, output_ids=output_ids_,
                          output_ids_y=output_ids_y, target_mask=target_mask, output_mask=output_mask, ntokens=ntokens,
                          input_length=input_length, output_length=output_length)
        )

    return features


def mtop_convert_examples_to_features(
        use_decode_emb,
        examples,
        tokenizer,
        outputs_map,
        output_vocab,
        max_seq_length=512,
        # cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True):

    features = []
    cnt_bad = 0
    for (ex_index, example) in enumerate(examples):
        tokens = example.text_inp.split(" ")
        output = example.text_out.split(" ")
        # if ex_index == 0:
        #     print(tokens)
        #     print(output)
        if not len(output)<=max_seq_length:
            cnt_bad += 1
            continue
        assert len(output)<=max_seq_length, "Length of output is larger than max_seq_length :: " + example.text_out

        input_tokenized = []
        span_indices = []
        schema_token_mask = []
        # cnt_sep = 0
        cnt_token = 0
        for token in tokens:
            start_idx = len(input_tokenized)
            end_idx = start_idx + len(tokenizer.tokenize(token))
            if end_idx + 1 > max_seq_length:
                break
            input_tokenized += tokenizer.tokenize(token)
            cnt_token += 1
            if token in [cls_token, sep_token]:
                schema_token_mask.append(0)
            else:
                schema_token_mask.append(1)
            span_indices.append([start_idx, end_idx])
            # if token == sep_token:
            #     cnt_sep += 1

        # assert(tokens[cnt_token-1] == sep_token and cnt_token == len(tokens))
        # add sep_token as the last one
        if tokens[cnt_token-1] != sep_token:
            start_idx = len(input_tokenized)
            input_tokenized += tokenizer.tokenize(sep_token)
            end_idx = len(input_tokenized) - 1
            span_indices.append([start_idx, end_idx])
            cnt_token += 1
            schema_token_mask.append(0)
        span_indices_mask = [1] * cnt_token

        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(input_tokenized)-1)
        input_ids = tokenizer.convert_tokens_to_ids(input_tokenized)
        if not len(input_ids) == len(segment_ids) == len(input_tokenized) == span_indices[-1][1]:
            cnt_bad += 1
            continue
        # assert(len(input_ids) == len(segment_ids) == len(input_tokenized) == span_indices[-1][1])
        assert(len(input_ids) <= max_seq_length)
        assert(cnt_token == len(span_indices) == len(span_indices_mask) == len(schema_token_mask))
        assert(cnt_token <= max_seq_length)

        # pad span_indices and span_indices_mask
        span_padding_length = max_seq_length - cnt_token
        span_indices += [[0,1]] * span_padding_length     # TODO: is this okay?
        span_indices_mask += [0] * span_padding_length
        schema_token_mask += [0] * span_padding_length
        # print(span_indices)
        # for i in range(max_seq_length):
        #     print(span_indices_mask[i], span_indices[i], tokens[i], input_tokenized[span_indices[i][0]:span_indices[i][1]])
        # exit()
        if sum(w not in outputs_map.keys() for w in output):
            cnt_bad += 1
            continue

        output_ids, output_ids_y, pointer_mask = convert_output_to_ids_spider(output, outputs_map)
        assert(len(output_ids) <= max_seq_length)
        # for i in range(len(output)):
        #     print(pointer_mask[i], output_ids[i], output_ids_y[i], output[i])
        # exit()

        input_length = len(input_ids)
        output_length = len(output_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        output_mask = [1 if mask_padding_with_zero else 0] * len(output_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # print(max_seq_length, len(input_ids), padding_length)
        padding_length_output = max_seq_length - len(output_ids)

        # pad input with pad_token
        input_ids += [pad_token] * padding_length
        # pad output always with 0
        output_ids += [0] * padding_length_output
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        output_mask += [0 if mask_padding_with_zero else 1] * padding_length_output
        segment_ids += [pad_token_segment_id] * padding_length

        output_ids_y += [0] * padding_length_output
        # pointer_mask += [1] * padding_length_output      # TODO: is this okay?
        pointer_mask += [0] * padding_length_output

        #################
        inp_ids = torch.tensor(input_ids)
        # print(use_decode_emb)
        if use_decode_emb:
            # print('here')
            # exit()
            out_ids = torch.tensor(output_ids_y)
        else:
            out_ids = torch.tensor(output_ids)
        source_mask = (inp_ids != pad_token).unsqueeze(-2)   # size = [1, max_seq_length], dtype=bool
        trg = out_ids[:-1]
        # trg_y = out_ids[1:]
        trg_y = torch.tensor(output_ids_y)[1:]
        target_mask = (trg != 0).unsqueeze(-2)               # size = [1, max_seq_length-1], dytpe=bool
        target_mask = target_mask & Variable(subsequent_mask(trg.size(-1)).type_as(target_mask.data)) # size = [1, max_seq_length-1, max_seq_length-1], dytpe=bool
        target_mask = target_mask.squeeze(0) # size = [max_seq_length-1, max_seq_length-1], dytpe=bool

        source_mask = source_mask.data.numpy()
        output_ids_ = trg.data.numpy()
        output_ids_y = trg_y.data.numpy()
        target_mask = target_mask.data.numpy()
        ntokens = (trg_y!=0).data.sum().item()
        #################

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(output_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(output_ids) == max_seq_length
        assert len(pointer_mask) == max_seq_length

        # assert(np.max(np.array(output_ids) * np.array(pointer_mask)) < len(tokens)-1)
        if not (np.max(np.array(output_ids) * np.array(pointer_mask)) < cnt_token-1):
            # This happens when the input, after truncating due to max_seq_length, does not contain a column in output
            cnt_bad += 1
            continue

        for oid in output_ids_y:
            if oid >= len(output_vocab):
                idx = oid - len(output_vocab)
                assert schema_token_mask[idx]==1, ex_index
        for idx,oid in enumerate(output_ids):
            if pointer_mask[idx] == 1 and oid != 0:
                assert schema_token_mask[oid]==1, ex_index

        # if ex_index == 0:
        #     print(output_ids_y)

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=input_mask, source_mask=source_mask,
                          token_type_ids=segment_ids, output_ids=output_ids_,
                          output_ids_y=output_ids_y, target_mask=target_mask, output_mask=output_mask, ntokens=ntokens,
                          input_length=input_length, output_length=output_length,
                          input_token_length=cnt_token,
                          span_indices=span_indices, span_indices_mask=span_indices_mask,
                          pointer_mask=pointer_mask, schema_token_mask=schema_token_mask)
        )
        # print(features[-1])

    print('cnt_bad', cnt_bad)

    tensor_dataset = TensorDataset(
            torch.tensor([f.input_ids for f in features], dtype=torch.long),
            torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            torch.tensor([f.source_mask for f in features], dtype=torch.bool),
            torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            torch.tensor([f.output_ids for f in features], dtype=torch.long),
            torch.tensor([f.output_ids_y for f in features], dtype=torch.long),
            torch.tensor([f.target_mask for f in features], dtype=torch.bool),
            torch.tensor([f.output_mask for f in features], dtype=torch.long),
            torch.tensor([f.ntokens for f in features], dtype=torch.long),
            torch.tensor([f.input_length for f in features], dtype=torch.long),
            torch.tensor([f.output_length for f in features], dtype=torch.long),
            torch.tensor([f.input_token_length for f in features], dtype=torch.long),
            torch.tensor([f.span_indices for f in features], dtype=torch.long),
            torch.tensor([f.span_indices_mask for f in features], dtype=torch.bool),
            torch.tensor([f.pointer_mask for f in features], dtype=torch.long),
            torch.tensor([f.schema_token_mask for f in features], dtype=torch.long)
        )

    return features, tensor_dataset


def spider_convert_examples_to_features(
        use_decode_emb,
        examples,
        tokenizer,
        outputs_map,
        output_vocab,
        max_seq_length=512,
        # cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True):


    database_schema, column_names_surface_form, column_names_embedder_input = read_database_schema('data/spider_data_removefrom/tables.json')

    features = []
    cnt_bad = 0
    for (ex_index, example) in enumerate(examples):
        # if ex_index != 8507:
        #     continue
        # if ex_index >= 32:
        #     break
        db_id = example.db_id
        tokens = example.text_inp.split(" ")
        output = example.text_out.split(" ")
        assert len(output)<=max_seq_length, "Length of output is larger than max_seq_length :: " + example.text_out

        input_tokenized = []
        span_indices = []
        schema_token_mask = []
        cnt_sep = 0
        cnt_token = 0
        for token in tokens:
            if cnt_sep == 0 or token in [cls_token, sep_token]:
                # utterance or special tokens
                start_idx = len(input_tokenized)
                end_idx = start_idx + len(tokenizer.tokenize(token))
                if end_idx + 1 > max_seq_length:
                    break
                input_tokenized += tokenizer.tokenize(token)
                cnt_token += 1
                schema_token_mask.append(0)
            else:
                # schema
                column_index = column_names_surface_form[db_id].index(token)
                column_name = column_names_embedder_input[db_id][column_index]
                start_idx = len(input_tokenized)
                end_idx = start_idx + len(tokenizer.tokenize(column_name))
                if end_idx + 1 > max_seq_length:
                    break
                input_tokenized += tokenizer.tokenize(column_name)
                cnt_token += 1
                schema_token_mask.append(1)
            span_indices.append([start_idx, end_idx])
            if token == sep_token:
                cnt_sep += 1
        # add sep_token as the last one
        if tokens[cnt_token-1] != sep_token:
            start_idx = len(input_tokenized)
            input_tokenized += tokenizer.tokenize(sep_token)
            end_idx = len(input_tokenized)
            span_indices.append([start_idx, end_idx])
            cnt_token += 1
            schema_token_mask.append(0)
        span_indices_mask = [1] * cnt_token

        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(input_tokenized)-1)
        input_ids = tokenizer.convert_tokens_to_ids(input_tokenized)
        assert(len(input_ids) == len(segment_ids) == len(input_tokenized) == span_indices[-1][1])
        assert(len(input_ids) <= max_seq_length)
        assert(cnt_token == len(span_indices) == len(span_indices_mask) == len(schema_token_mask))
        assert(cnt_token <= max_seq_length)

        # pad span_indices and span_indices_mask
        span_padding_length = max_seq_length - cnt_token
        span_indices += [[0,1]] * span_padding_length
        span_indices_mask += [0] * span_padding_length
        schema_token_mask += [0] * span_padding_length
        # for i in range(max_seq_length):
        #     print(span_indices_mask[i], span_indices[i], tokens[i], input_tokenized[span_indices[i][0]:span_indices[i][1]+1])
        # exit()

        output_ids, output_ids_y, pointer_mask = convert_output_to_ids_spider(output, outputs_map)
        assert(len(output_ids) <= max_seq_length)
        # for i in range(len(output)):
        #     print(pointer_mask[i], output_ids[i], output_ids_y[i], output[i])
        # exit()

        input_length = len(input_ids)
        output_length = len(output_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        output_mask = [1 if mask_padding_with_zero else 0] * len(output_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # print(max_seq_length, len(input_ids), padding_length)
        padding_length_output = max_seq_length - len(output_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            output_ids = ([0] * padding_length_output) + output_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            output_mask = ([0 if mask_padding_with_zero else 1] * padding_length_output) + output_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            # pad input with pad_token
            input_ids += [pad_token] * padding_length
            # pad output always with 0
            output_ids += [0] * padding_length_output
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            output_mask += [0 if mask_padding_with_zero else 1] * padding_length_output
            segment_ids += [pad_token_segment_id] * padding_length

            output_ids_y += [0] * padding_length_output
            pointer_mask += [1] * padding_length_output

        #################
        inp_ids = torch.tensor(input_ids)
        if use_decode_emb:
            # print('here')
            # exit()
            out_ids = torch.tensor(output_ids_y)
        else:
            out_ids = torch.tensor(output_ids)
        # out_ids = torch.tensor(output_ids)
        source_mask = (inp_ids != pad_token).unsqueeze(-2)   # size = [1, max_seq_length], dtype=bool
        trg = out_ids[:-1]
        # trg_y = out_ids[1:]
        trg_y = torch.tensor(output_ids_y)[1:]
        target_mask = (trg != 0).unsqueeze(-2)               # size = [1, max_seq_length-1], dytpe=bool
        target_mask = target_mask & Variable(subsequent_mask(trg.size(-1)).type_as(target_mask.data)) # size = [1, max_seq_length-1, max_seq_length-1], dytpe=bool
        target_mask = target_mask.squeeze(0) # size = [max_seq_length-1, max_seq_length-1], dytpe=bool

        source_mask = source_mask.data.numpy()
        output_ids_ = trg.data.numpy()
        output_ids_y = trg_y.data.numpy()
        target_mask = target_mask.data.numpy()
        ntokens = (trg_y!=0).data.sum().item()
        #################

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(output_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(output_ids) == max_seq_length
        assert len(pointer_mask) == max_seq_length

        assert(np.max(np.array(output_ids) * np.array(pointer_mask)) < len(tokens)-1)
        if not (np.max(np.array(output_ids) * np.array(pointer_mask)) < cnt_token-1):
            # This happens when the input, after truncating due to max_seq_length, does not contain a column in output
            cnt_bad += 1
            continue
        # input_mask: [max_seq_length], int
        # source_mask: [1, max_seq_length], bool
        # output_mask: [max_seq_length], int
        # target_mask: [max_seq_length-1, max_seq_length-1], bool
        # print(len(input_mask))
        # print(source_mask.shape)
        # print(len(output_mask))
        # print(target_mask.shape)
        # exit()

        # print(output_ids_)
        # print(output_ids_y)
        # print(output_mask)
        # print(pointer_mask)
        # exit()

        # print(output_ids_y)
        # print(schema_token_mask)
        # print(len(output_vocab))
        # if ex_index == 8507:
        #     print(len(output_vocab))
        #     print(db_id)
        #     print(tokens)
        #     print(tokens[126], schema_token_mask[126])
        #     # assert len(tokens) == len(schema_token_mask)
        #     print(cnt_token, len(tokens))
        #     for a, b in zip(tokens, schema_token_mask):
        #         print(a,b)
        #     print(output)
        #     print(output_ids)
        #     print(pointer_mask)
        #     print(np.max(np.array(output_ids) * np.array(pointer_mask)), cnt_token)
        #     print(output_ids_y)
            # print(schema_token_mask)
        bad_flag = 0
        for oid in output_ids_y:
            if oid >= len(output_vocab):
                idx = oid - len(output_vocab)
                if schema_token_mask[idx]!=1:
                    bad_flag = 1
                    break
                # assert schema_token_mask[idx]==1, ex_index
        if bad_flag:
            cnt_bad += 1
            continue

        # print(output_ids)
        # print(pointer_mask)
        # print(schema_token_mask)
        for idx,oid in enumerate(output_ids):
            if pointer_mask[idx] == 1 and oid != 0:
                if not schema_token_mask[oid]==1:
                    # This happens when the input, after truncating due to max_seq_length, does not contain a column in output
                    cnt_bad += 1
                    continue

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=input_mask, source_mask=source_mask,
                          token_type_ids=segment_ids, output_ids=output_ids_,
                          output_ids_y=output_ids_y, target_mask=target_mask, output_mask=output_mask, ntokens=ntokens,
                          input_length=input_length, output_length=output_length,
                          input_token_length=cnt_token,
                          span_indices=span_indices, span_indices_mask=span_indices_mask,
                          pointer_mask=pointer_mask, schema_token_mask=schema_token_mask)
        )
        # print(features[-1])

    print('cnt_bad', cnt_bad)

    tensor_dataset = TensorDataset(
            torch.tensor([f.input_ids for f in features], dtype=torch.long),
            torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            torch.tensor([f.source_mask for f in features], dtype=torch.bool),
            torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            torch.tensor([f.output_ids for f in features], dtype=torch.long),
            torch.tensor([f.output_ids_y for f in features], dtype=torch.long),
            torch.tensor([f.target_mask for f in features], dtype=torch.bool),
            torch.tensor([f.output_mask for f in features], dtype=torch.long),
            torch.tensor([f.ntokens for f in features], dtype=torch.long),
            torch.tensor([f.input_length for f in features], dtype=torch.long),
            torch.tensor([f.output_length for f in features], dtype=torch.long),
            torch.tensor([f.input_token_length for f in features], dtype=torch.long),
            torch.tensor([f.span_indices for f in features], dtype=torch.long),
            torch.tensor([f.span_indices_mask for f in features], dtype=torch.bool),
            torch.tensor([f.pointer_mask for f in features], dtype=torch.long),
            torch.tensor([f.schema_token_mask for f in features], dtype=torch.long)
        )

    return features, tensor_dataset


class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:min(i + self.batch_size, len(ids))] for i in range(0, len(ids), batch_size)]
        
    def shuffle(self):
        np.random.shuffle(self.bins)
        
    def num_samples(self):
        return len(self.data_source)
    
    def __len__(self):
        return len(self.bins)
    
    def __iter__(self):
        return iter(self.bins)
