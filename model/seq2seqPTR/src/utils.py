import argparse
import random, os, csv, logging, json, copy, math, operator
from queue import PriorityQueue
from typing import Optional
from dataclasses import dataclass

import itertools

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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def get_range_vector(size: int, device) -> torch.Tensor:
  """
  """
  return torch.arange(0, size, dtype=torch.long).to(device)

def flatten_and_batch_shift_indices(indices: torch.LongTensor,
                                    sequence_length: int) -> torch.Tensor:
  """``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor,
  which has size ``(batch_size, sequence_length, embedding_size)``. This function returns a vector
  that correctly indexes into the flattened target. The sequence length of the target must be provided
  to compute the appropriate offset.
  Args:
    indices (torch.LongTensor):
  """
  if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
    raise ValueError("All the elements should be in range (0, {}), but found ({}, {})".format(
      sequence_length - 1, torch.min(indices).item(), torch.max(indices).item()))
  offsets = get_range_vector(indices.size(0), indices.device) * sequence_length
  for _ in range(len(indices.size()) - 1):
    offsets = offsets.unsqueeze(1)

  # (batch_size, d_1, ..., d_n) + (batch_size, 1, ..., 1)
  offset_indices = indices + offsets

  # (batch_size * d_1 * ... * d_n)
  offset_indices = offset_indices.view(-1)
  return offset_indices

def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
  """Select ``target`` of size ``(batch_size, sequence_length, embedding_size)`` with ``indices`` of
  size ``(batch_size, d_1, ***, d_n)``.
  Args:
    target (torch.Tensor): A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
  """
  if flattened_indices is None:
    flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

  # Shape: (batch_size * sequence_length, embedding_size)
  flattened_target = target.view(-1, target.size(-1))

  # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
  flattened_selected = flattened_target.index_select(0, flattened_indices)
  selected_shape = list(indices.size()) + [target.size(-1)]

  # Shape: (batch_size, d_1, ..., d_n, embedding_size)
  selected_targets = flattened_selected.view(*selected_shape)
  return selected_targets

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
  """
  ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
  masked. This performs a softmax on just the non-masked positions of ``vector``. Passing ``None``
  in for the mask is also acceptable, which is just the regular softmax.
  """
  if mask is None:
    result = torch.softmax(vector, dim=dim)
  else:
    mask = mask.float()
    while mask.dim() < vector.dim():
      mask = mask.unsqueeze(1)
    masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
    result = torch.softmax(masked_vector, dim=dim)
  return result

def weighted_sum(matrix: torch.Tensor,
                 attention: torch.Tensor) -> torch.Tensor:
  """
  Args:
    matrix ():
    attention ():
  """
  if attention.dim() == 2 and matrix.dim() == 3:
    return attention.unsqueeze(1).bmm(matrix).squeeze(1)
  if attention.dim() == 3 and matrix.dim() == 3:
    return attention.bmm(matrix)
  if matrix.dim() - 1 < attention.dim():
    expanded_size = list(matrix.size())
    for i in range(attention.dim() - matrix.dim() + 1):
      matrix = matrix.unsqueeze(1)
      expanded_size.insert(i + 1, attention.size(i + 1))
    matrix = matrix.expand(*expanded_size)
  intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
  return intermediate.sum(dim=-2)

class AverageSpanExtractor(nn.Module):
  def __init__(self):
    super(AverageSpanExtractor, self).__init__()

  def forward(self,
              sequence_tensor: torch.FloatTensor,
              span_indices: torch.LongTensor,
              span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
    # Shape (batch_size, num_spans, 1)
    span_starts, span_ends = span_indices.split(1, dim=-1)

    span_ends = span_ends - 1

    span_widths = span_ends - span_starts

    max_batch_span_width = span_widths.max().item() + 1

    # sequence_tensor (batch, length, dim)
    # global_attention_logits = self._global_attention(sequence_tensor)
    global_average_logits = torch.ones(sequence_tensor.size()[:2] + (1,)).float().to(sequence_tensor.device)

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width,
                                                    sequence_tensor.device).view(1, 1, -1)
    span_mask = (max_span_range_indices <= span_widths).float()

    # (batch_size, num_spans, 1) - (1, 1, max_batch_span_width)
    raw_span_indices = span_ends - max_span_range_indices
    span_mask = span_mask * (raw_span_indices >= 0).float()
    span_indices = torch.relu(raw_span_indices.float()).long()

    flat_span_indices = flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

    span_embeddings = batched_index_select(sequence_tensor, span_indices, flat_span_indices)

    span_attention_logits = batched_index_select(global_average_logits,
                                                       span_indices,
                                                       flat_span_indices).squeeze(-1)

    span_attention_weights = masked_softmax(span_attention_logits, span_mask)

    attended_text_embeddings = weighted_sum(span_embeddings, span_attention_weights)

    if span_indices_mask is not None:
      return attended_text_embeddings * span_indices_mask.unsqueeze(-1).float()

    return attended_text_embeddings


# Decoder from 'Attention is all you need'

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
def subsequent_mask(size, batch_size=1):
    "Mask out subsequent positions."
    attn_shape = (batch_size, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
#         return F.log_softmax(self.proj(x), dim=-1)
        return self.proj(x)

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, criterion, opt=None):
        # self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        norm = norm.data.sum()
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        return loss

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        # x:      [batch_size x max_out_len, num_label]
        # target: [batch_size x max_out_len]
        assert x.size(1) == self.size

        # true_dist: [batch_size x max_out_len, num_label]
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 3))   # TODO: why 3 here?
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        res = self.criterion(x, Variable(true_dist, requires_grad=False))
        return res