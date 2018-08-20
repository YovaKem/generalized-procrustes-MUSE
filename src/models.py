# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from .utils import load_embeddings, normalize_embeddings

class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params.src_lang, params.src_emb, params)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)
    params.tgt_dico = {}
    tgt_emb = {}
    # target embeddings
    if params.tgt_lang:
        tgt_lang_list = params.tgt_lang
        tgt_emb_list = params.tgt_emb
        assert len(tgt_emb_list) == len(tgt_lang_list)
        for lang, emb in zip(tgt_lang_list,tgt_emb_list):
            tgt_dico, _tgt_emb = load_embeddings(lang, emb, params)
            params.tgt_dico[lang] = tgt_dico
            tgt_emb[lang] = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
            tgt_emb[lang].weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    # mapping for each target language
    mapping = {lang: nn.Linear(params.emb_dim, params.emb_dim, bias=False) for lang in [params.src_lang]+tgt_lang_list}
    if getattr(params, 'map_id_init', True):
        for lang in [params.src_lang]+tgt_lang_list:
            mapping[lang].weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            for lang in tgt_lang_list:
                tgt_emb[lang].cuda()
        for lang in tgt_lang_list+[params.src_lang]:
            mapping[lang].cuda()
            if with_dis and lang in tgt_lang_list:
                discriminator[lang].cuda()

    # normalize embeddings
    normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        for lang in tgt_lang_list:
            normalize_embeddings(tgt_emb[lang].weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping, discriminator
