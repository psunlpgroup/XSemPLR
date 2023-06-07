#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Timothy_xie
# datetime： 2021/7/26 0:33
# ide： PyCharm

from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False,
                                                       cache_dir='/data/yfz5488/cache')
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location,
            cache_dir='/data/yfz5488/cache'
        )
        self.config = self.pretrain_model.config

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, labels):
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
