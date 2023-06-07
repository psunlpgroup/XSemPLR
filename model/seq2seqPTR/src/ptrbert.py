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

from transformers import (PreTrainedModel, BertPreTrainedModel, BartPretrainedModel, T5PreTrainedModel,
                          BertModel, RobertaModel, XLMRobertaModel,
                          BartModel, MBartModel, T5Model, MT5Model,
                          BertConfig, RobertaConfig, XLMRobertaConfig,
                          BartConfig, MBartConfig, T5Config, MT5Config)

from .utils import Decoder, DecoderLayer, subsequent_mask, MultiHeadedAttention, PositionwiseFeedForward, Embeddings, PositionalEncoding, Generator, SimpleLossCompute, LabelSmoothing, AverageSpanExtractor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class BeamSearchNode(object):
    def __init__(self, ys, previousNode, wordId, logProb, length, ys_pointer_mask=None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.ys = ys
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.ys_pointer_mask = ys_pointer_mask

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    
    def __lt__(self, other):
        return True


def beam_decode(input_id, decoder_embed, decoder, encoder_outputs, encoder_outputs_excluding_sptokens, source_mask, generator, num_ptrs, bart_model=False):

    beam_width = 5
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []
    start_symbol = 1
    end_symbol = 2
    batch_size = 1
    # global num_ptrs
    # num_ptrs = num_ptrs
    
    # decoding goes sentence by sentence
    for idx in range(encoder_outputs.size(0)):
        # print(idx, encoder_outputs.size(0))

        encoder_output = encoder_outputs[idx, : , :].unsqueeze(0) # [1, 128, 768]
        encoder_output_excluding_sptokens = encoder_outputs_excluding_sptokens[idx, : , :].unsqueeze(0)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(input_id.data)
        
        node = BeamSearchNode(ys, None, start_symbol, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1
        
        tmpnodes = PriorityQueue()
        breaknow = False
        
        while not breaknow:
            
            nextnodes = PriorityQueue()
            
            # start beam search
            while nodes.qsize()>0:

                # fetch the best node
                score, n = nodes.get()
                prev_id = n.wordid
                ys = n.ys

                if (ys.shape[1]>100):
                    breaknow = True
                    break

                if n.wordid == end_symbol and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) == beam_width:
                        breaknow = True
                        break
                    else:
                        continue

                # decode for one step using decoder
                if not bart_model:
                    src_mask = source_mask[idx, :, :] # [1, max_inp_len]
                    decoder_output = decoder(decoder_embed(Variable(ys)), encoder_output, src_mask,
                                             Variable(subsequent_mask(ys.size(1), batch_size=batch_size).type_as(input_id.data)))
                    # decoder_output: [1, decoding_len, hidden_size]
                    # print('decoder_output', decoder_output.size())
                else:
                    # attention_mask = Variable(subsequent_mask(ys.size(1), batch_size=batch_size).type_as(input_id.data))
                    # BUG: attention_mask = torch.tensor([[1]]).type_as(input_id.data)
                    # [1, decoding_len]
                    decoder_attention_mask = torch.ones(1, ys.size(1)).type_as(input_id.data)
                    # print('attention_mask', attention_mask.size())
                    # [1, max_inp_len]
                    encoder_attention_mask = source_mask[idx, :].unsqueeze(0)
                    # print('encoder_attention_mask', encoder_attention_mask.size())
                    inputs_embeds = decoder_embed(Variable(ys))
                    # print('inputs_embeds', inputs_embeds.size())
                    
                    decoder_outputs = decoder(
                        input_ids=None,
                        attention_mask=decoder_attention_mask,
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_attention_mask,
                        # head_mask=decoder_head_mask,
                        # cross_attn_head_mask=cross_attn_head_mask,
                        # past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        # use_cache=use_cache,
                        # output_attentions=output_attentions,
                        # output_hidden_states=output_hidden_states,
                        return_dict=True,
                    )
                    decoder_output = decoder_outputs.last_hidden_state
                    # print('decoder_output', decoder_output.size())

                generator_scores = generator(decoder_output[:, -1])
                src_ptr_scores = torch.einsum('ac, adc -> ad', decoder_output[:, -1], encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])

                # Make src_ptr_scores of proper shape by appending 0's for non-existent pointers
                a, b = src_ptr_scores.shape
                src_ptr_scores_net = torch.zeros(a, num_ptrs).cuda()
                src_ptr_scores_net[:,:b] = src_ptr_scores

                all_scores = torch.cat((generator_scores, src_ptr_scores), axis=1)

                all_prob = F.log_softmax(all_scores, dim=-1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                top_log_prob, top_indexes = torch.topk(all_prob, beam_width)

                for new_k in range(beam_width):
                    decoded_t = top_indexes[0][new_k].view(1, -1).type_as(ys)
                    log_prob = top_log_prob[0][new_k].item()

                    ys2 = torch.cat([ys, decoded_t], dim=1)

                    node = BeamSearchNode(ys2, n, decoded_t.item(), n.logp + log_prob, n.leng + 1)
                    score = -node.eval()
                    nextnodes.put((score, node))


                # put them into queue
                for i in range(beam_width):
                    if nextnodes.qsize()>0:
                        score, nn = nextnodes.get()
                        nodes.put((score, nn))

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)
            if (len(utterances)==topk):
                break

        decoded_batch.append(utterances)

    return decoded_batch


def beam_decode_spider(config, decoder_embed, decoder, encoder_outputs, encoder_outputs_excluding_sptokens, schema_token_masks, source_mask, input_lengths, generator, num_ptrs, bart_model=False, encoder_inputs=None, beam_width=5, topk=1, start_symbol=1, end_symbol=2, batch_size=1):
    decoded_batch = []
    # decoding goes sentence by sentence
    for idx in range(encoder_outputs.size(0)):
        encoder_output = encoder_outputs[idx, : , :].unsqueeze(0) # [1, max_inp_len, d_model]
        encoder_output_excluding_sptokens = encoder_outputs_excluding_sptokens[idx, : , :].unsqueeze(0)
        schema_token_mask = schema_token_masks[idx, :].unsqueeze(0)  # [1, max_inp_len]
        input_length = input_lengths[idx]
        if bart_model:
            encoder_input = encoder_inputs[idx, : , :].unsqueeze(0)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        ys = torch.ones(batch_size, 1).fill_(start_symbol).long()
        ys_pointer_mask = torch.zeros(batch_size, 1).long()
        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(ys, None, start_symbol, 0, 1, ys_pointer_mask)
        nodes = PriorityQueue()
        nodes.put((-node.eval(), node))

        breaknow = False
        while not breaknow:
            nextnodes = PriorityQueue()
            # start beam search
            while nodes.qsize()>0:
                # fetch the best node
                score, n = nodes.get()
                if nodes.qsize() > 10000: # Yusen: I add this condition in case of dead loop
                    breaknow = True
                    break
                prev_id = n.wordid
                ys = n.ys
                ys_pointer_mask = n.ys_pointer_mask

                if (ys.shape[1]>100):
                    breaknow = True
                    break

                if n.wordid == end_symbol and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) == beam_width:
                        breaknow = True
                        break
                    else:
                        continue

                # decode for one step using decoder
                if not bart_model:
                    src_mask = source_mask[idx, :, :] # [1, max_inp_len] TODO: why not unsqueeze here?
                    ys = ys.to(encoder_output.device)

                    if config.use_decode_emb:
                        decoder_inputs_embeds = decoder_embed(ys)
                    else:
                        ys_pointer_mask = ys_pointer_mask.to(encoder_output.device)

                        decoder_inputs_embeds_target = decoder_embed(ys * (1-ys_pointer_mask))
                        decoder_inputs_embeds_target = (1-ys_pointer_mask).unsqueeze(-1) * decoder_inputs_embeds_target
                        decoder_inputs_embeds_pointer = torch.gather(input=encoder_output, dim=1, 
                                                                    index=(ys*ys_pointer_mask).unsqueeze(-1).expand(-1,-1,encoder_output.size(2)))
                        decoder_inputs_embeds_pointer = ys_pointer_mask.unsqueeze(-1) * decoder_inputs_embeds_pointer
                        decoder_inputs_embeds = decoder_inputs_embeds_target + decoder_inputs_embeds_pointer

                    decoder_output = decoder(decoder_inputs_embeds, encoder_output, src_mask,
                                             subsequent_mask(ys.size(1), batch_size).long().to(encoder_output.device))
                    # decoder_output: [1, decoding_len, d_model]
                    # print('decoder_output', decoder_output.size())
                else:
                    # [1, decoding_len]
                    decoder_attention_mask = torch.ones(1, ys.size(1)).long().to(encoder_output.device)
                    # [1, max_inp_len]
                    encoder_attention_mask = source_mask[idx, :].unsqueeze(0)

                    ys = ys.to(encoder_output.device)
                    if config.use_decode_emb:
                        decoder_inputs_embeds = decoder_embed(ys)
                    else:
                        ys_pointer_mask = ys_pointer_mask.to(encoder_output.device)

                        decoder_inputs_embeds_target = decoder_embed(ys * (1-ys_pointer_mask))
                        decoder_inputs_embeds_target = (1-ys_pointer_mask).unsqueeze(-1) * decoder_inputs_embeds_target

                        decoder_inputs_embeds_pointer = torch.gather(input=encoder_input, dim=1, 
                                                                     index=(ys*ys_pointer_mask).unsqueeze(-1).expand(-1,-1,encoder_input.size(2)))
                        decoder_inputs_embeds_pointer = ys_pointer_mask.unsqueeze(-1) * decoder_inputs_embeds_pointer

                        decoder_inputs_embeds = decoder_inputs_embeds_target + decoder_inputs_embeds_pointer
                        if hasattr(decoder, 'embed_scale'):
                            decoder_inputs_embeds = decoder_inputs_embeds * decoder.embed_scale

                    decoder_outputs = decoder(
                        input_ids=None,
                        attention_mask=decoder_attention_mask,
                        # encoder_hidden_states=encoder_outputs.last_hidden_state,
                        # encoder_attention_mask=attention_mask,
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_attention_mask,
                        inputs_embeds=decoder_inputs_embeds,
                        use_cache=False,
                        return_dict=True
                    )
                    decoder_output = decoder_outputs.last_hidden_state

                generator_scores = generator(decoder_output[:, -1])   # [1, target_vocab_size]
                # decoder_output[:, -1]: [1, d_model]
                # encoder_output_excluding_sptokens: [1, max_inp_len, d_model]
                # src_ptr_scores: [1, max_inp_len]
                src_ptr_scores = torch.einsum('ac, adc -> ad', decoder_output[:, -1], encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])

                if config.use_schema_token_mask:
                    # schema_token_mask: [1, max_inp_len]
                    src_ptr_scores = schema_token_mask * src_ptr_scores

                if config.smoothing == 0.0:
                    if config.use_schema_token_mask:
                        src_ptr_scores = src_ptr_scores + torch.log(schema_token_mask)
                    else:
                        src_ptr_scores[0, 0] = torch.log(torch.zeros(1))
                        src_ptr_scores[0, input_length-1:] = torch.log(torch.zeros(1))

                # src_ptr_scores = schema_token_mask * src_ptr_scores + torch.log(schema_token_mask)

                # Make src_ptr_scores of proper shape by appending 0's for non-existent pointers
                # a, b = src_ptr_scores.shape
                # src_ptr_scores_net = torch.zeros(a, num_ptrs).cuda()
                # src_ptr_scores_net[:,:b] = src_ptr_scores

                all_scores = torch.cat((generator_scores, src_ptr_scores), 1)
                all_prob = F.log_softmax(all_scores, dim=-1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                top_log_prob, top_indexes = torch.topk(all_prob, beam_width)

                for new_k in range(beam_width):
                    decoded_t = top_indexes[0][new_k].view(1, -1).type_as(ys)
                    log_prob = top_log_prob[0][new_k].item()

                    if config.use_decode_emb:
                        ys2 = torch.cat([ys, decoded_t], dim=1)
                        ys2_pointer_mask = ys_pointer_mask
                    else:
                        if decoded_t[0][0].item() < generator_scores.size(1):
                            ys2 = torch.cat([ys, decoded_t], dim=1)
                            ys2_pointer_mask = torch.cat([ys_pointer_mask, torch.zeros(1,1).long().to(ys_pointer_mask.device)], dim=1)
                        else:
                            ys2 = torch.cat([ys, decoded_t-generator_scores.size(1)], dim=1)
                            ys2_pointer_mask = torch.cat([ys_pointer_mask, torch.ones(1,1).long().to(ys_pointer_mask.device)], dim=1)      

                    node = BeamSearchNode(ys2, n, decoded_t.item(), n.logp + log_prob, n.leng + 1, ys2_pointer_mask)
                    score = -node.eval()
                    nextnodes.put((score, node))

                # put them into queue (This can be put to outer loop, but no harm)
                for i in range(beam_width):
                    if nextnodes.qsize()>0:
                        score, nn = nextnodes.get()
                        nodes.put((score, nn))

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)
            if (len(utterances)==topk):
                break

        decoded_batch.append(utterances)

    return decoded_batch


class PtrRoberta(PreTrainedModel):

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def __init__(self, config, bert_model, num_ptrs, all_outputs, output_vocab,
    #                    decoder_layers=6, decoder_size=2048, decoder_heads=8, decoder_dropout=0.1):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        d_model = config.hidden_size
        self.d_model = d_model
        # self.num_ptrs = num_ptrs

        bert_model = config.bert_model
        self.num_ptrs = config.num_ptrs
        all_outputs = config.all_outputs
        output_vocab = config.output_vocab
        decoder_layers = config.decoder_layers
        decoder_size = config.decoder_size
        decoder_heads = config.decoder_heads
        decoder_dropout = config.decoder_dropout

        if bert_model in ['bert-base-cased', 'bert-base-multilingual-cased']:
            PtrRoberta.config_class = BertConfig
            # PtrRoberta.load_tf_weights = load_tf_weights_in_bert
            PtrRoberta.base_model_prefix = "bert"
            PtrRoberta._keys_to_ignore_on_load_missing = [r"position_ids"]

            self.bert = BertModel(config)
        elif bert_model in ['roberta-base', 'roberta-large']:
            PtrRoberta.config_class = RobertaConfig
            PtrRoberta.base_model_prefix = "roberta"

            self.roberta = RobertaModel(config)
        elif bert_model in ['xlm-roberta-base', 'xlm-roberta-large']:
            PtrRoberta.config_class = XLMRobertaConfig
            PtrRoberta.base_model_prefix = "roberta"

            self.roberta = XLMRobertaModel(config)



        # Decoder
        N = decoder_layers # stack of N=6 identical layers
        # The dimensionality of input and output is dmodel=512 , and the inner-layer has dimensionality dff=2048
        d_ff = decoder_size
        h = decoder_heads # parallel attention layers, or heads
        dropout = decoder_dropout
        net_vocab = len(all_outputs)
        tgt_vocab = len(output_vocab)
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        # TODO
        self.use_schema_token_mask = config.use_schema_token_mask

        self.use_avg_span_extractor = config.use_avg_span_extractor
        if self.use_avg_span_extractor:
            self.average_span_extractor = AverageSpanExtractor()
        self.use_decode_emb = config.use_decode_emb
        if self.use_decode_emb:
            self.decoder_embed = nn.Sequential(Embeddings(d_model, net_vocab), c(position))
        else:
            self.decoder_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        self.generator = Generator(d_model, tgt_vocab)
        
        # Loss
        # TODO
        self.smoothing = config.smoothing
        criterion = LabelSmoothing(size=net_vocab, padding_idx=0, smoothing=self.smoothing)
        self.loss_compute = SimpleLossCompute(criterion)
        ###########

        print(self.config)

        self.init_weights()


    def forward_old(
        self,
        input_ids=None,
        attention_mask=None,
        source_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_ids=None, 
        output_attention_mask=None,
        output_ids_y=None,
        target_mask=None,
        output_mask=None,
        ntokens=None,
        input_length=None,
        output_length=None,
        decode=False,
        max_len=60, # 60 sufficient for valid
        start_symbol=1):
        
        # szs = []
        
        max_inp_len = torch.max(input_length).cpu().item()
        max_out_len = torch.max(output_length).cpu().item()
        
        input_ids = input_ids[:, :max_inp_len]
        attention_mask = attention_mask[:, :max_inp_len]  # [batch_size, max_inp_len]
        token_type_ids = token_type_ids[:, :max_inp_len]
        source_mask = source_mask[:,:,:max_inp_len]       # [batch_size, 1, max_inp_len]
        
        output_ids = output_ids[:, :max_out_len]
        output_ids_y = output_ids_y[:, :max_out_len]
        target_mask = target_mask[:, :max_out_len, :max_out_len]   # [batch_size, max_inp_len, max_inp_len]
        
        # print(max_inp_len, max_out_len)
        # print('input_ids')
        # print(input_ids)
        # print('attention_mask')
        # print(attention_mask)
        # print('output_ids')
        # print(output_ids)
        # print('output_ids_y')
        # print(output_ids_y)
        # print('output_mask')
        # print(output_mask)
        # print(attention_mask.size())
        # print(source_mask.size())
        # print(output_mask.size())
        # exit()
        
        if (self.config_class == BertConfig):
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds)
        else:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds)
        
        encoder_output = outputs[0]
        decoder_output = self.decoder(self.decoder_embed(output_ids), encoder_output, source_mask, target_mask)
        # print(encoder_output.size())
        # print(decoder_output.size())
        # exit()
        tgt_vocab_scores = self.generator(decoder_output)
        # norm = ntokens.data.sum()
        
        # Remove output for <start> and <end> tokens from encoder_output for excluding their pointer scores
        encoder_output_excluding_sptokens = encoder_output.clone()
        for i in range(encoder_output.shape[0]):
            encoder_output_excluding_sptokens[i,input_length[i]-1:,:] = 0.0 # removing output corresponding to <end> tokens for different seq lengths       
        encoder_output_excluding_sptokens = encoder_output_excluding_sptokens[:,1:-1,:] # removing output corresponding to <start> & <end> tokens as they all are aligned
        
        # Get pointer scores
        src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])#
        
        # Ensure that scores for padding are automatically zeroed out when all input_lengths are not same as appropriate 
        for i in range(input_length.shape[0]):
            if (torch.sum(src_ptr_scores[i,:,input_length[i]-2:]) != 0.0):
                print("max_inp_len: ", torch.max(input_length).cpu().item())
                print("src_ptr_scores.shape: ", src_ptr_scores.shape)
                print("input_length[i]: ", input_length[i])
                print("--", src_ptr_scores[i, 0, input_length[i]-2:])
            assert torch.sum(src_ptr_scores[i,:,input_length[i]-2:]) == 0.0
            
        # Make src_ptr_scores of proper shape by appending 0's for non-existent pointers
        a, b, c = src_ptr_scores.shape
        src_ptr_scores_net = torch.zeros(a, b, self.num_ptrs).cuda()
        src_ptr_scores_net[:,:,:c] = src_ptr_scores
        
        assert torch.sum(src_ptr_scores_net[:,:,c:]) == 0.0
        
        all_scores = torch.cat((tgt_vocab_scores, src_ptr_scores_net), axis=2)
        all_scores = F.log_softmax(all_scores, dim=-1) #---? dummy x, weights may not get updated effectively
        
        loss = self.loss_compute(all_scores, output_ids_y, ntokens)

        #########
        
        if not decode:
            return loss
        else:
            ys = beam_decode(input_ids, self.decoder_embed, self.decoder, encoder_output, encoder_output_excluding_sptokens,
                             source_mask, self.generator, self.num_ptrs)            
            return (loss, ys)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        source_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_ids=None, 
        output_attention_mask=None,
        output_ids_y=None,
        target_mask=None,
        output_mask=None,
        ntokens=None,
        input_length=None,
        output_length=None,
        input_token_length=None,
        span_indices=None,
        span_indices_mask=None,
        pointer_mask=None,
        schema_token_mask=None,
        decode=False):

        max_inp_len = torch.max(input_length).cpu().item()
        max_out_len = torch.max(output_length).cpu().item()
        
        input_ids = input_ids[:, :max_inp_len]
        attention_mask = attention_mask[:, :max_inp_len]  # [batch_size, max_inp_len]
        token_type_ids = token_type_ids[:, :max_inp_len]
        source_mask = source_mask[:,:,:max_inp_len]       # [batch_size, 1, max_inp_len]
        
        output_ids = output_ids[:, :max_out_len]
        output_ids_y = output_ids_y[:, :max_out_len]
        target_mask = target_mask[:, :max_out_len, :max_out_len]   # [batch_size, max_inp_len, max_inp_len]

        pointer_mask = pointer_mask[:, :max_out_len]  # [batch_size, max_out_len]
        
        if (self.config_class == BertConfig):
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
        else:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds)

        if self.use_avg_span_extractor:
            max_tok_len = torch.max(input_token_length).cpu().item()
            span_indices = span_indices[:, :max_tok_len]
            span_indices_mask = span_indices_mask[:, :max_tok_len]
            schema_token_mask = schema_token_mask[:, :max_tok_len]
            # Input:
            # sequence_tensor: [batch_size, max_inp_len, d_model]
            # span_indices: [batch_size, max_tok_len] list of list of span indices, e.g., [[0,0], [1,2], [3,5], [6,8], [9,9], [0,0], [0,0]]
            # span_indices_mask: [batch_size, max_tok_len] list of list of masks,   e.g., [1, 1, 1, 1, 1, 0, 0]
            # Output:
            # encoder_output: [batch_size, max_tok_len, d_model]
            encoder_output = self.average_span_extractor(sequence_tensor=outputs[0],
                                                        span_indices=span_indices,
                                                        span_indices_mask=span_indices_mask)
            source_mask = span_indices_mask.unsqueeze(1)
            input_length = input_token_length
        else:
            encoder_output = outputs[0]
            print(max_inp_len, encoder_output.size())

        # encoder_output: [batch_size, max_inp_len, hidden_size]
        # output_ids: [batch_size, max_out_len]
        # decoder_inputs_embeds: [batch_size, max_out_len, hidden_size]

        if self.use_decode_emb:
            decoder_inputs_embeds = self.decoder_embed(output_ids)
        else:
            decoder_inputs_embeds_target = self.decoder_embed(output_ids*(1-pointer_mask)) # [batch_size, max_out_len, d_model]
            decoder_inputs_embeds_target = (1-pointer_mask).unsqueeze(-1) * decoder_inputs_embeds_target
            # encoder_output: [batch_size, max_inp_len, d_model]
            # index:          [batch_size, max_out_len, d_model]
            decoder_inputs_embeds_pointer = torch.gather(input=encoder_output, dim=1, 
                                                        index=(output_ids*pointer_mask).unsqueeze(-1).expand(-1,-1,encoder_output.size(2))) # [batch_size, max_out_len, d_model]
            decoder_inputs_embeds_pointer = pointer_mask.unsqueeze(-1) * decoder_inputs_embeds_pointer
            decoder_inputs_embeds = decoder_inputs_embeds_target + decoder_inputs_embeds_pointer

        # TODO: pointer_mask: 1 if it is a pointer, 0 otherwise; decoder_size should be the same as d_model; change output_ids; double check padding
        # TODO: change beam_decode_spider()
        # TODO: remove two for loops below

        # decoder_output: [batch_size, max_out_len, d_model]
        decoder_output = self.decoder(decoder_inputs_embeds, encoder_output, source_mask, target_mask)
        
        # tgt_vocab_scores: [batch_size, max_out_len, tgt_size]
        tgt_vocab_scores = self.generator(decoder_output)
        
        # encoder_output_excluding_sptokens = encoder_output
        # encoder_output_excluding_sptokens = encoder_output.clone()
        # [batch_size, max_inp_len]
        # encoder_output_excluding_sptokens_mask = torch.arange(max_inp_len).expand(input_length.size(0),-1).to(input_length.device) < ((input_length-1).unsqueeze(-1))
        # print(max_inp_len)
        # print(input_length)
        # print(encoder_output_excluding_sptokens_mask)
        # exit()
        # encoder_output_excluding_sptokens = encoder_output_excluding_sptokens_mask.unsqueeze(-1) * encoder_output_excluding_sptokens
        # for i in range(encoder_output.shape[0]):
        #     encoder_output_excluding_sptokens[i,input_length[i]-1:,:] = 0.0 # removing output corresponding to <end> tokens for different seq lengths

        # TODO: [batch_size, max_inp_len-2, d_model] why do this? This does create something wrong for label alignment!!!
        # encoder_output_excluding_sptokens = encoder_output_excluding_sptokens[:,1:-1,:] # removing output corresponding to <start> & <end> tokens as they all are aligned

        # Get pointer scores
        # TODO: what if decoder_output size != encoder_output_excluding_sptokens size
        # src_ptr_scores: [batch_size, max_out_len, max_inp_len-2]
        # print(decoder_output.size())
        # print(encoder_output_excluding_sptokens.size())
        # print(max_inp_len)
        # exit()

        # [batch_size, max_out_len, d_model], [batch_size, max_inp_len, d_model] -> [batch_size, max_out_len, max_inp_len]
        # src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])

        # TODO: need to ensure pointer scores are non-zero only for schema tokens

        if self.use_schema_token_mask:
            encoder_output_excluding_sptokens = encoder_output
            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output)/np.sqrt(decoder_output.shape[-1])
            # src_ptr_scores: [batch_size, max_out_len, max_inp_len]
            # schema_token_mask: [batch_size, max_inp_len]
            src_ptr_scores = schema_token_mask.unsqueeze(1) * src_ptr_scores
        else:
            encoder_output_excluding_sptokens = encoder_output.clone()
            for i in range(encoder_output.shape[0]):
                encoder_output_excluding_sptokens[i,0,:] = 0.0                  # zero output corresponding to <start> tokens
                encoder_output_excluding_sptokens[i,input_length[i]-1:,:] = 0.0 # zero output after <end> tokens
            
            # Get pointer scores
            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])#
            
            # Ensure that scores for padding are automatically zeroed out when all input_lengths are not same as appropriate 
            for i in range(input_length.shape[0]):
                if (torch.sum(src_ptr_scores[i,:,input_length[i]-1:]) != 0.0):
                    print("max_inp_len: ", torch.max(input_length).cpu().item())
                    print("src_ptr_scores.shape: ", src_ptr_scores.shape)
                    print("input_length[i]: ", input_length[i])
                    print("--", src_ptr_scores[i, 0, input_length[i]-1:])
                assert torch.sum(src_ptr_scores[i,:,input_length[i]-1:]) == 0.0
        
        if self.smoothing == 0.0:
            if self.use_schema_token_mask:
                src_ptr_scores = src_ptr_scores + torch.log(schema_token_mask).unsqueeze(1)
            else:
                for i in range(src_ptr_scores.shape[0]):
                    src_ptr_scores[i, :, 0] = torch.log(torch.zeros(1))
                    src_ptr_scores[i, :, input_length[i]-1:] = torch.log(torch.zeros(1))
            assert(not torch.any(torch.isnan(src_ptr_scores)))
            a, b, c = src_ptr_scores.shape
            src_ptr_scores_net = torch.log(torch.zeros(a, b, self.num_ptrs).to(src_ptr_scores.device))
            src_ptr_scores_net[:,:,:c] = src_ptr_scores
        else:
            assert(not torch.any(torch.isnan(src_ptr_scores)))
            a, b, c = src_ptr_scores.shape
            src_ptr_scores_net = torch.zeros(a, b, self.num_ptrs).to(src_ptr_scores.device)
            src_ptr_scores_net[:,:,:c] = src_ptr_scores

        # all_scores: [batch_size, max_out_len, tgt_size+num_ptrs]
        all_scores = torch.cat((tgt_vocab_scores, src_ptr_scores_net), 2)
        all_scores = F.log_softmax(all_scores, dim=-1)
        loss = self.loss_compute(all_scores, output_ids_y, ntokens)
        assert(not torch.any(torch.isinf(loss)))
        assert(not torch.any(torch.isnan(loss)))
        
        # if schema_token_mask is not None and self.smoothing == 0.0:
        #     # if self.smoothing != 0.0:
        #     #     print('cannot have smoothing with 0 probabilities.')
        #     #     exit()
        #     encoder_output_excluding_sptokens = encoder_output
        #     src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])
        #     # src_ptr_scores: [batch_size, max_out_len, max_inp_len]
        #     # schema_token_mask: [batch_size, max_inp_len]
        #     src_ptr_scores = (schema_token_mask.unsqueeze(1) * src_ptr_scores +
        #                       torch.log(schema_token_mask).unsqueeze(1))
        #                     #  (1-schema_token_mask).unsqueeze(1) * torch.log(torch.zeros(src_ptr_scores.size()).to(src_ptr_scores.device)))
        #     # print(src_ptr_scores[0][0])
        #     # print(schema_token_mask[0])
        #     assert(not torch.any(torch.isnan(src_ptr_scores)))
        #     a, b, c = src_ptr_scores.shape
        #     src_ptr_scores_net = torch.log(torch.zeros(a, b, self.num_ptrs).to(src_ptr_scores.device))
        #     src_ptr_scores_net[:,:,:c] = src_ptr_scores

        #     # [batch_size, max_out_len, tgt_size+num_ptrs]
        #     all_scores = torch.cat((tgt_vocab_scores, src_ptr_scores_net), 2)
        #     all_scores = F.log_softmax(all_scores, dim=-1)
        #     # print(all_scores[0][0])
        #     # exit()
        #     # print(all_scores.size())
        #     # print(all_scores[0])
        #     # print(output_ids_y[0])
        #     # for j in range(output_ids_y[0].size(0)):
        #     #     print(output_ids_y[0][j])
        #     #     print(all_scores[0][j])
        #     #     print(all_scores[0][j][output_ids_y[0][j]])
        #     #     print()
        #     loss = self.loss_compute(all_scores, output_ids_y, ntokens)
        #     # print('loss', loss)
        #     assert(not torch.any(torch.isinf(loss)))
        #     assert(not torch.any(torch.isnan(loss)))
        # else:
        #     # print('here')
        #     # exit()
        #     encoder_output_excluding_sptokens = encoder_output.clone()
        #     for i in range(encoder_output.shape[0]):
        #         encoder_output_excluding_sptokens[i,0,:] = 0.0
        #         encoder_output_excluding_sptokens[i,input_length[i]-1:,:] = 0.0 # removing output corresponding to <end> tokens for different seq lengths
            
        #     # Get pointer scores
        #     src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])#
            
        #     # Ensure that scores for padding are automatically zeroed out when all input_lengths are not same as appropriate 
        #     for i in range(input_length.shape[0]):
        #         if (torch.sum(src_ptr_scores[i,:,input_length[i]-1:]) != 0.0):
        #             print("max_inp_len: ", torch.max(input_length).cpu().item())
        #             print("src_ptr_scores.shape: ", src_ptr_scores.shape)
        #             print("input_length[i]: ", input_length[i])
        #             print("--", src_ptr_scores[i, 0, input_length[i]-1:])
        #         assert torch.sum(src_ptr_scores[i,:,input_length[i]-1:]) == 0.0
                
        #     # Make src_ptr_scores of proper shape by appending 0's for non-existent pointers
        #     a, b, c = src_ptr_scores.shape
        #     src_ptr_scores_net = torch.zeros(a, b, self.num_ptrs).to(src_ptr_scores.device)
        #     src_ptr_scores_net[:,:,:c] = src_ptr_scores
            
        #     assert torch.sum(src_ptr_scores_net[:,:,c:]) == 0.0
            
        #     all_scores = torch.cat((tgt_vocab_scores, src_ptr_scores_net), 2)
        #     # TODO: wait, is this good???? This is because we are using label smoothing? check difference between CrossEntropyLoss, NLLoss, and KLDivLoss
        #     all_scores = F.log_softmax(all_scores, dim=-1)
            
        #     # TODO: check loss, especially alignment between all_scores and output_ids_y
        #     loss = self.loss_compute(all_scores, output_ids_y, ntokens)

        #########
        
        if not decode:
            return loss
        else:
            ys = beam_decode_spider(self.config, self.decoder_embed, self.decoder, encoder_output, encoder_output_excluding_sptokens,
                                    schema_token_mask, source_mask, input_length, self.generator, self.num_ptrs)            
            return (loss, ys)


class PtrBART_old(PreTrainedModel):
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

    def __init__(self, config, bert_model, num_ptrs, all_outputs, output_vocab):
        super().__init__(config)
        d_model = config.d_model
        # self.num_labels = config.num_labels # Includes both maxPtr# + intent labels + slot labels

        if bert_model in ['facebook/bart-base', 'facebook/bart-large']:
            PtrBART.config_class = BartConfig
            # https://huggingface.co/transformers/_modules/transformers/models/bart/modeling_bart.html#BartPretrainedModel
            PtrBART._keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]
            self.model = BartModel(config)
        elif bert_model in ['facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
            PtrBART.config_class = MBartConfig
            # PtrBART._keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]
            self.model = MBartModel(config)

        net_vocab = len(all_outputs)
        tgt_vocab = len(output_vocab)
        self.num_ptrs = num_ptrs

        # Decoder Input Embeddings
        dropout = 0.1
        position = PositionalEncoding(d_model, dropout)
        self.decoder_embed = nn.Sequential(Embeddings(d_model, net_vocab), position)

        # Use BART decoder
        # self.decoder = self.model.decoder

        # Decoder Output Embeddings
        self.generator = Generator(d_model, tgt_vocab)

        # Loss
        criterion = LabelSmoothing(size=net_vocab, padding_idx=0, smoothing=0.1)
        self.loss_compute = SimpleLossCompute(self.generator, criterion, None)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            source_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_ids=None, 
            output_attention_mask=None,
            output_ids_y=None,
            target_mask=None, 
            output_mask=None,
            ntokens=None,
            input_length=None,
            output_length=None,
            decode=False,
            max_len=60, # 60 sufficient for valid
            start_symbol=1):

        max_inp_len = torch.max(input_length).cpu().item()
        max_out_len = torch.max(output_length).cpu().item()
        
        input_ids = input_ids[:, :max_inp_len]
        attention_mask = attention_mask[:, :max_inp_len]
        # token_type_ids = token_type_ids[:, :max_inp_len]
        # source_mask = source_mask[:,:,:max_inp_len]
        
        output_ids = output_ids[:, :max_out_len]
        output_ids_y = output_ids_y[:, :max_out_len]

        output_mask = output_mask[:, :max_out_len]
        # target_mask = target_mask[:, :max_out_len, :max_out_len]

        # print(max_inp_len, max_out_len)
        # print('input_ids')
        # print(input_ids)
        # print('attention_mask')
        # print(attention_mask)
        # print('output_ids')
        # print(output_ids)
        # print('output_ids_y')
        # print(output_ids_y)
        # print('output_mask')
        # print(output_mask)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,      # TODO: Is this correct?
            decoder_input_ids=None,
            decoder_attention_mask=output_mask, # TODO: Is this correct?
            decoder_inputs_embeds=self.decoder_embed(output_ids),
            return_dict=True
            )
        encoder_output = outputs.encoder_last_hidden_state
        decoder_output = outputs.last_hidden_state
        # print(encoder_output.size())
        # print(decoder_output.size())
        # exit()

        tgt_vocab_scores = self.generator(decoder_output)
        # norm = ntokens.data.sum()
        
        # Remove output for <start> and <end> tokens from encoder_output for excluding their pointer scores
        encoder_output_excluding_sptokens = encoder_output.clone()
        for i in range(encoder_output.shape[0]):
            encoder_output_excluding_sptokens[i,input_length[i]-1:,:] = 0.0 # removing output corresponding to <end> tokens for different seq lengths       
        encoder_output_excluding_sptokens = encoder_output_excluding_sptokens[:,1:-1,:] # removing output corresponding to <start> & <end> tokens as they all are aligned
        
        # Get pointer scores
        src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])#
        
        # Ensure that scores for padding are automatically zeroed out when all input_lengths are not same as appropriate 
        for i in range(input_length.shape[0]):
            # if (torch.sum(src_ptr_scores[i,:,input_length[i]-2:]) != 0.0):
            #     print("max_inp_len: ", torch.max(input_length).cpu().item())
            #     print("src_ptr_scores.shape: ", src_ptr_scores.shape)
            #     print("input_length[i]: ", input_length[i])
            #     print("--", src_ptr_scores[i, 0, input_length[i]-2:])
            assert torch.sum(src_ptr_scores[i,:,input_length[i]-2:]) == 0.0
            
        # Make src_ptr_scores of proper shape by appending 0's for non-existent pointers
        a, b, c = src_ptr_scores.shape
        src_ptr_scores_net = torch.zeros(a, b, self.num_ptrs).cuda()
        src_ptr_scores_net[:,:,:c] = src_ptr_scores
        
        assert torch.sum(src_ptr_scores_net[:,:,c:]) == 0.0
        
        all_scores = torch.cat((tgt_vocab_scores, src_ptr_scores_net), axis=2)
        all_scores = F.log_softmax(all_scores, dim=-1) #---? dummy x, weights may not get updated effectively
        
        loss = self.loss_compute(all_scores, output_ids_y, ntokens)

        #########
        
        # TODO
        if not decode:
            return loss
        else:
            # source_mask=attention_mask
            ys = beam_decode(input_ids, self.decoder_embed, self.model.decoder, encoder_output, encoder_output_excluding_sptokens,
                             attention_mask, self.generator, self.num_ptrs, bart_model=True)            
            return (loss, ys)


class PtrBART(PreTrainedModel):
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        d_model = config.d_model # dimision of the model
        self.d_model = d_model

        bert_model = config.bert_model
        self.num_ptrs = config.num_ptrs

        if bert_model in ['facebook/bart-base', 'facebook/bart-large']:
            PtrBART.config_class = BartConfig
            # https://huggingface.co/transformers/_modules/transformers/models/bart/modeling_bart.html#BartPretrainedModel
            PtrBART._keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]
            self.model = BartModel(config)
        elif bert_model in ['facebook/mbart-large-50', 'facebook/mbart-large-50-one-to-many-mmt']:
            PtrBART.config_class = MBartConfig
            # PtrBART._keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]
            self.model = MBartModel(config)

        net_vocab = len(config.all_outputs)
        tgt_vocab = len(config.output_vocab)

        # TODO
        self.use_schema_token_mask = config.use_schema_token_mask

        self.use_avg_span_extractor = config.use_avg_span_extractor
        if self.use_avg_span_extractor:
            self.average_span_extractor = AverageSpanExtractor()

        # TODO: Decoder Input Embeddings
        # dropout = config.decoder_dropout
        # position = PositionalEncoding(d_model, dropout)
        # c = copy.deepcopy
        # self.use_decode_emb = config.use_decode_emb
        # if self.use_decode_emb:
        #     self.decoder_embed = nn.Sequential(Embeddings(d_model, net_vocab), c(position))
        # else:
        #     self.decoder_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))

        self.use_decode_emb = config.use_decode_emb
        # vanilla nn.Embedding for output vocab, with padding_idx 0
        if self.use_decode_emb:
            self.decoder_embed = nn.Embedding(net_vocab, d_model, padding_idx=0)
        else:
            self.decoder_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)

        # Decoder Output Embeddings
        self.generator = Generator(d_model, tgt_vocab)

        # Loss
        self.smoothing = config.smoothing
        criterion = LabelSmoothing(size=net_vocab, padding_idx=0, smoothing=self.smoothing)
        self.loss_compute = SimpleLossCompute(criterion)
        ###########

        print(self.config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            source_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_ids=None, 
            output_attention_mask=None,
            output_ids_y=None,
            target_mask=None, 
            output_mask=None,
            ntokens=None,
            input_length=None,
            output_length=None,
            input_token_length=None,
            span_indices=None,
            span_indices_mask=None,
            pointer_mask=None,
            schema_token_mask=None,
            decode=False):

        max_inp_len = torch.max(input_length).cpu().item()
        max_out_len = torch.max(output_length).cpu().item()
        
        input_ids = input_ids[:, :max_inp_len]
        attention_mask = attention_mask[:, :max_inp_len]
        # token_type_ids = token_type_ids[:, :max_inp_len]
        # source_mask = source_mask[:,:,:max_inp_len]
        
        output_ids = output_ids[:, :max_out_len]
        output_ids_y = output_ids_y[:, :max_out_len]
        output_mask = output_mask[:, :max_out_len]

        pointer_mask = pointer_mask[:, :max_out_len]  # [batch_size, max_out_len]

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if self.use_avg_span_extractor:
            max_tok_len = torch.max(input_token_length).cpu().item()
            span_indices = span_indices[:, :max_tok_len]
            span_indices_mask = span_indices_mask[:, :max_tok_len]
            schema_token_mask = schema_token_mask[:, :max_tok_len]
            # Input:
            # sequence_tensor: [batch_size, max_inp_len, d_model]
            # span_indices: [batch_size, max_tok_len] list of list of span indices, e.g., [[0,0], [1,2], [3,5], [6,8], [9,9], [0,0], [0,0]]
            # span_indices_mask: [batch_size, max_tok_len] list of list of masks,   e.g., [1, 1, 1, 1, 1, 0, 0]
            # Output:
            # encoder_output: [batch_size, max_tok_len, d_model]
            encoder_output = self.average_span_extractor(sequence_tensor=encoder_outputs.last_hidden_state,
                                                        span_indices=span_indices,
                                                        span_indices_mask=span_indices_mask)
            # source_mask = span_indices_mask.unsqueeze(1)
            source_mask = span_indices_mask
            input_length = input_token_length
        else:
            encoder_output = encoder_outputs.last_hidden_state
            print(max_inp_len, encoder_output.size())

        # TODO: need to check carefully here
        if self.use_decode_emb:
            decoder_inputs_embeds = self.decoder_embed(output_ids)
        # else:
        #     decoder_inputs_embeds_target = self.decoder_embed(output_ids*(1-pointer_mask)) # [batch_size, max_out_len, d_model]
        #     decoder_inputs_embeds_target = (1-pointer_mask).unsqueeze(-1) * decoder_inputs_embeds_target
        #     # encoder_output: [batch_size, max_inp_len, d_model]
        #     # index:          [batch_size, max_out_len, d_model]
        #     decoder_inputs_embeds_pointer = torch.gather(input=encoder_output, dim=1, 
        #                                                 index=(output_ids*pointer_mask).unsqueeze(-1).expand(-1,-1,encoder_output.size(2))) # [batch_size, max_out_len, d_model]
        #     decoder_inputs_embeds_pointer = pointer_mask.unsqueeze(-1) * decoder_inputs_embeds_pointer
        #     decoder_inputs_embeds = decoder_inputs_embeds_target + decoder_inputs_embeds_pointer
        else:
            # TODO: need to change to BART's decoder embeddings
            decoder_inputs_embeds_target = self.decoder_embed(output_ids*(1-pointer_mask)) # [batch_size, max_out_len, d_model]
            decoder_inputs_embeds_target = (1-pointer_mask).unsqueeze(-1) * decoder_inputs_embeds_target

            # [batch_size, max_inp_len, d_model]
            encoder_inputs = self.model.decoder.embed_tokens(input_ids)
            # [batch_size, max_tok_len, d_model]
            encoder_inputs = self.average_span_extractor(sequence_tensor=encoder_inputs,
                                                         span_indices=span_indices,
                                                         span_indices_mask=span_indices_mask)
            # [batch_size, max_out_len, d_model]
            decoder_inputs_embeds_pointer = torch.gather(input=encoder_inputs, dim=1, 
                                                         index=(output_ids*pointer_mask).unsqueeze(-1).expand(-1,-1,encoder_inputs.size(2)))
            decoder_inputs_embeds_pointer = pointer_mask.unsqueeze(-1) * decoder_inputs_embeds_pointer

            decoder_inputs_embeds = decoder_inputs_embeds_target + decoder_inputs_embeds_pointer
            decoder_inputs_embeds = decoder_inputs_embeds * self.model.decoder.embed_scale

        decoder_outputs = self.model.decoder(
            input_ids=None,
            attention_mask=output_mask,
            # encoder_hidden_states=encoder_outputs.last_hidden_state,
            # encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=span_indices_mask,
            inputs_embeds=decoder_inputs_embeds,
            # use_cache=self.model.config.use_cache,
            use_cache=False,
            return_dict=True
        )

        decoder_output = decoder_outputs.last_hidden_state

        tgt_vocab_scores = self.generator(decoder_output)

        if self.use_schema_token_mask:
            encoder_output_excluding_sptokens = encoder_output
            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output)/np.sqrt(decoder_output.shape[-1])
            # src_ptr_scores: [batch_size, max_out_len, max_inp_len]
            # schema_token_mask: [batch_size, max_inp_len]
            src_ptr_scores = schema_token_mask.unsqueeze(1) * src_ptr_scores
        else:
            encoder_output_excluding_sptokens = encoder_output.clone()
            for i in range(encoder_output.shape[0]):
                encoder_output_excluding_sptokens[i,0,:] = 0.0                  # zero output corresponding to <start> tokens
                encoder_output_excluding_sptokens[i,input_length[i]-1:,:] = 0.0 # zero output after <end> tokens
            
            # Get pointer scores
            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])#
            
            # Ensure that scores for padding are automatically zeroed out when all input_lengths are not same as appropriate 
            for i in range(input_length.shape[0]):
                if (torch.sum(src_ptr_scores[i,:,input_length[i]-1:]) != 0.0):
                    print("max_inp_len: ", torch.max(input_length).cpu().item())
                    print("src_ptr_scores.shape: ", src_ptr_scores.shape)
                    print("input_length[i]: ", input_length[i])
                    print("--", src_ptr_scores[i, 0, input_length[i]-1:])
                assert torch.sum(src_ptr_scores[i,:,input_length[i]-1:]) == 0.0

        if self.smoothing == 0.0:
            if self.use_schema_token_mask:
                src_ptr_scores = src_ptr_scores + torch.log(schema_token_mask).unsqueeze(1)
            else:
                for i in range(src_ptr_scores.shape[0]):
                    src_ptr_scores[i, :, 0] = torch.log(torch.zeros(1))
                    src_ptr_scores[i, :, input_length[i]-1:] = torch.log(torch.zeros(1))
            assert(not torch.any(torch.isnan(src_ptr_scores)))
            a, b, c = src_ptr_scores.shape
            src_ptr_scores_net = torch.log(torch.zeros(a, b, self.num_ptrs).to(src_ptr_scores.device))
            src_ptr_scores_net[:,:,:c] = src_ptr_scores
        else:
            assert(not torch.any(torch.isnan(src_ptr_scores)))
            a, b, c = src_ptr_scores.shape
            src_ptr_scores_net = torch.zeros(a, b, self.num_ptrs).to(src_ptr_scores.device)
            src_ptr_scores_net[:,:,:c] = src_ptr_scores

        # all_scores: [batch_size, max_out_len, tgt_size+num_ptrs]
        all_scores = torch.cat((tgt_vocab_scores, src_ptr_scores_net), 2)
        all_scores = F.log_softmax(all_scores, dim=-1)
        loss = self.loss_compute(all_scores, output_ids_y, ntokens)
        assert(not torch.any(torch.isinf(loss)))
        assert(not torch.any(torch.isnan(loss)))

        if not decode:
            return loss
        else:
            ys = beam_decode_spider(self.config, self.decoder_embed, self.model.decoder,
                                    encoder_output, encoder_output_excluding_sptokens,
                                    schema_token_mask, source_mask, input_length, self.generator, self.num_ptrs,
                                    bart_model=True, encoder_inputs=encoder_inputs)            
            return (loss, ys)


class PtrT5(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        d_model = config.d_model
        bert_model = config.bert_model
        self.num_ptrs = config.num_ptrs

        if bert_model in ['t5-large', 't5-base', 't5-small']:
            # https://huggingface.co/transformers/_modules/transformers/models/t5/modeling_t5.html#T5Model
            PtrT5._keys_to_ignore_on_load_missing = [
                r"encoder\.embed_tokens\.weight",
                r"decoder\.embed_tokens\.weight",
            ]
            PtrT5._keys_to_ignore_on_load_unexpected = [
                r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
            ]
            self.transformer = T5Model(config)
        elif bert_model in ['google/mt5-large']:
            # https://huggingface.co/transformers/_modules/transformers/models/mt5/modeling_mt5.html#MT5Model
            PtrT5.model_type = "mt5"
            PtrT5.config_class = MT5Config
            PtrT5._keys_to_ignore_on_load_missing = [
                r"encoder\.embed_tokens\.weight",
                r"decoder\.embed_tokens\.weight",
                r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
            ]
            PtrT5._keys_to_ignore_on_save = [
                r"encoder\.embed_tokens\.weight",
                r"decoder\.embed_tokens\.weight",
            ]
            self.transformer = MT5Model(config)

        net_vocab = len(config.all_outputs)
        tgt_vocab = len(config.output_vocab)

        self.use_schema_token_mask = config.use_schema_token_mask

        self.use_avg_span_extractor = config.use_avg_span_extractor
        if self.use_avg_span_extractor:
            self.average_span_extractor = AverageSpanExtractor()

        # Decoder Input Embeddings
        self.use_decode_emb = config.use_decode_emb
        # vanilla nn.Embedding for output vocab, with padding_idx 0
        if self.use_decode_emb:
            self.decoder_embed = nn.Embedding(net_vocab, d_model, padding_idx=0)
        else:
            self.decoder_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)

        # Decoder Output Embeddings
        self.generator = Generator(d_model, tgt_vocab)

        # Loss
        self.smoothing = config.smoothing
        criterion = LabelSmoothing(size=net_vocab, padding_idx=0, smoothing=self.smoothing)
        self.loss_compute = SimpleLossCompute(criterion)
        ###########

        print(self.config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            source_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_ids=None, 
            output_attention_mask=None,
            output_ids_y=None,
            target_mask=None, 
            output_mask=None,
            ntokens=None,
            input_length=None,
            output_length=None,
            input_token_length=None,
            span_indices=None,
            span_indices_mask=None,
            pointer_mask=None,
            schema_token_mask=None,
            decode=False):
        
        max_inp_len = torch.max(input_length).cpu().item()
        max_out_len = torch.max(output_length).cpu().item()
        
        input_ids = input_ids[:, :max_inp_len]
        attention_mask = attention_mask[:, :max_inp_len]
        
        output_ids = output_ids[:, :max_out_len]
        output_ids_y = output_ids_y[:, :max_out_len]
        output_mask = output_mask[:, :max_out_len]

        pointer_mask = pointer_mask[:, :max_out_len]  # [batch_size, max_out_len]

        encoder_outputs = self.transformer.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if self.use_avg_span_extractor:
            max_tok_len = torch.max(input_token_length).cpu().item()
            span_indices = span_indices[:, :max_tok_len]
            span_indices_mask = span_indices_mask[:, :max_tok_len]
            schema_token_mask = schema_token_mask[:, :max_tok_len]
            # Input:
            # sequence_tensor: [batch_size, max_inp_len, d_model]
            # span_indices: [batch_size, max_tok_len] list of list of span indices, e.g., [[0,0], [1,2], [3,5], [6,8], [9,9], [0,0], [0,0]]
            # span_indices_mask: [batch_size, max_tok_len] list of list of masks,   e.g., [1, 1, 1, 1, 1, 0, 0]
            # Output:
            # encoder_output: [batch_size, max_tok_len, d_model]
            encoder_output = self.average_span_extractor(sequence_tensor=encoder_outputs.last_hidden_state,
                                                        span_indices=span_indices,
                                                        span_indices_mask=span_indices_mask)
            # source_mask = span_indices_mask.unsqueeze(1)
            source_mask = span_indices_mask
            input_length = input_token_length
        else:
            encoder_output = encoder_outputs.last_hidden_state

        if self.use_decode_emb:
            decoder_inputs_embeds = self.decoder_embed(output_ids)
        else:
            decoder_inputs_embeds_target = self.decoder_embed(output_ids*(1-pointer_mask)) # [batch_size, max_out_len, d_model]
            decoder_inputs_embeds_target = (1-pointer_mask).unsqueeze(-1) * decoder_inputs_embeds_target

            # [batch_size, max_inp_len, d_model]
            encoder_inputs = self.transformer.decoder.embed_tokens(input_ids)
            # [batch_size, max_tok_len, d_model]
            encoder_inputs = self.average_span_extractor(sequence_tensor=encoder_inputs,
                                                         span_indices=span_indices,
                                                         span_indices_mask=span_indices_mask)
            decoder_inputs_embeds_pointer = torch.gather(input=encoder_inputs, dim=1, 
                                                         index=(output_ids*pointer_mask).unsqueeze(-1).expand(-1,-1,encoder_inputs.size(2))) # [batch_size, max_out_len, d_model]
            decoder_inputs_embeds_pointer = pointer_mask.unsqueeze(-1) * decoder_inputs_embeds_pointer

            decoder_inputs_embeds = decoder_inputs_embeds_target + decoder_inputs_embeds_pointer

        decoder_outputs = self.transformer.decoder(
            input_ids=None,
            attention_mask=output_mask,
            # encoder_hidden_states=encoder_outputs.last_hidden_state,
            # encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=span_indices_mask,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=False,
            return_dict=True
        )

        decoder_output = decoder_outputs.last_hidden_state

        tgt_vocab_scores = self.generator(decoder_output)

        if self.use_schema_token_mask:
            encoder_output_excluding_sptokens = encoder_output
            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output)/np.sqrt(decoder_output.shape[-1])
            # src_ptr_scores: [batch_size, max_out_len, max_inp_len]
            # schema_token_mask: [batch_size, max_inp_len]
            src_ptr_scores = schema_token_mask.unsqueeze(1) * src_ptr_scores
        else:
            encoder_output_excluding_sptokens = encoder_output.clone()
            for i in range(encoder_output.shape[0]):
                encoder_output_excluding_sptokens[i,0,:] = 0.0                  # zero output corresponding to <start> tokens
                encoder_output_excluding_sptokens[i,input_length[i]-1:,:] = 0.0 # zero output after <end> tokens
            
            # Get pointer scores
            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])#
            
            # Ensure that scores for padding are automatically zeroed out when all input_lengths are not same as appropriate 
            for i in range(input_length.shape[0]):
                if (torch.sum(src_ptr_scores[i,:,input_length[i]-1:]) != 0.0):
                    print("max_inp_len: ", torch.max(input_length).cpu().item())
                    print("src_ptr_scores.shape: ", src_ptr_scores.shape)
                    print("input_length[i]: ", input_length[i])
                    print("--", src_ptr_scores[i, 0, input_length[i]-1:])
                assert torch.sum(src_ptr_scores[i,:,input_length[i]-1:]) == 0.0

        if self.smoothing == 0.0:
            if self.use_schema_token_mask:
                src_ptr_scores = src_ptr_scores + torch.log(schema_token_mask).unsqueeze(1)
            else:
                for i in range(src_ptr_scores.shape[0]):
                    src_ptr_scores[i, :, 0] = torch.log(torch.zeros(1))
                    src_ptr_scores[i, :, input_length[i]-1:] = torch.log(torch.zeros(1))
            assert(not torch.any(torch.isnan(src_ptr_scores)))
            a, b, c = src_ptr_scores.shape
            src_ptr_scores_net = torch.log(torch.zeros(a, b, self.num_ptrs).to(src_ptr_scores.device))
            src_ptr_scores_net[:,:,:c] = src_ptr_scores
        else:
            assert(not torch.any(torch.isnan(src_ptr_scores)))
            a, b, c = src_ptr_scores.shape
            src_ptr_scores_net = torch.zeros(a, b, self.num_ptrs).to(src_ptr_scores.device)
            src_ptr_scores_net[:,:,:c] = src_ptr_scores

        # all_scores: [batch_size, max_out_len, tgt_size+num_ptrs]
        all_scores = torch.cat((tgt_vocab_scores, src_ptr_scores_net), 2)
        all_scores = F.log_softmax(all_scores, dim=-1)
        loss = self.loss_compute(all_scores, output_ids_y, ntokens)
        assert(not torch.any(torch.isinf(loss)))
        assert(not torch.any(torch.isnan(loss)))

        if not decode:
            return loss
        else:
            ys = beam_decode_spider(self.config, self.decoder_embed, self.transformer.decoder,
                                    encoder_output, encoder_output_excluding_sptokens,
                                    schema_token_mask, source_mask, input_length, self.generator, self.num_ptrs,
                                    bart_model=True, encoder_inputs=encoder_inputs)            
            return (loss, ys)