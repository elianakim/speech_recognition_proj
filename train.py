# -*- coding: utf-8 -*-

import os
import json
import argparse
import math
import torch
import torch.nn as nn
import torchaudio
import soundfile
import numpy as np
import editdistance
import pickle
from flask import Flask, request
from tqdm import tqdm

from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

device=None

## ===================================================================
## Load labels
## ===================================================================
def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char

        return char2index, index2char

## ===================================================================
## Data loader
## ===================================================================
class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_path, max_length, char2index):
        super(SpeechDataset, self).__init__()

        # load data from JSON
        with open(data_list, 'r') as f:
            data = json.load(f)

        # convert seconds to frames
        max_length *= 16000

        # sort data in length order and filter data less than max_length
        data = sorted(data, key=lambda d: d['len'], reverse=True)
        self.data = [x for x in data if x['len'] <= max_length]

        self.dataset_path = data_path
        self.char2index = char2index

    def __getitem__(self, index):
        # read audio using soundfile.read
        data_path = os.path.join(self.dataset_path, self.data[index]['file'])
        audio = soundfile.read(data_path)[0]

        # read transcript and convert to indices
        transcript = self.data[index]['text']
        transcript = self.parse_transcript(transcript)

        return torch.FloatTensor(audio), torch.LongTensor(transcript)

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return len(self.data)

class EvalSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_path, max_length):
        super(EvalSpeechDataset, self).__init__()

        # load data from JSON
        with open(data_list, 'r') as f:
            data = json.load(f)

        # convert seconds to frames
        max_length *= 16000

        # sort data in length order and filter data less than max_length
        data = sorted(data, key=lambda d: d['len'], reverse=True)
        self.data = [x for x in data if x['len'] <= max_length]

        self.dataset_path = data_path

    def __getitem__(self, index):
        data_path = os.path.join(self.dataset_path, self.data[index]['file'])
        audio = soundfile.read(data_path)[0]

        return torch.FloatTensor(audio)

    def __len__(self):
        return len(self.data)

## ===================================================================
## Define collate function
## ===================================================================
def pad_collate(batch):
    (xx, yy) = zip(*batch)

    ## compute lengths of each item in xx and yy
    x_lens = [len(x) for x in xx]  # < fill your code here >
    y_lens = [len(y) for y in yy]  # < fill your code here >

    ## zero-pad to the longest length
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)  # < fill your code here >
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)  # < fill your code here >

    return xx_pad, yy_pad, x_lens, y_lens

def eval_pad_collate(batch):
    x_lens = [len(x) for x in batch]
    xx_pad = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)  # < fill your code here >
    return xx_pad, x_lens

## ===================================================================
## Define sampler 
## ===================================================================
class BucketingSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        # Shuffle bins in random order
        np.random.shuffle(self.bins)

        # For each bin
        for ids in self.bins:
            # Shuffle indices in random order
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

## ===================================================================
## Test-Time Adaptation Modules
## ===================================================================
class TENT(nn.Module):
    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps

        # turn on grad for batch norm parameters only
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                # force use of batch stats in train and eval modes
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)

def configure_model(model):
    # set train mode as tent optimizes the model to minimize entropy
    model.train()
    # disable grad
    for param in model.parameters():
        param.requires_grad = False
    # turn on grad for batch norm parameters only
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            # force use of batch stats in train and eval modes
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model

def softmax_entropy(x, dim=2):
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def mcc_loss(x, dim=2, class_num=11):
    mcc_loss = 0.0
    for b in range(x.shape[0]): #batch
        p = x[b, :, :].softmax(1)
        cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) = (D, D)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        batch_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
        mcc_loss += batch_loss
    return mcc_loss / x.shape[0]

def collect_params(model):
    # collect the affine scale + shift parameters from batch norms
    # walk the model's modules and collect all batch normalization parameters
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f'{nm}.{np}')
    return params, names

## ===================================================================
## Conformer
## reference: https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
## ===================================================================
# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module
class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None, context_mask = None):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block
class ConformerBlock(nn.Module):
    def __init__(self, *, dim, dim_head = 64, heads = 8, ff_mult = 4, conv_expansion_factor = 2, conv_kernel_size = 31, attn_dropout = 0., ff_dropout = 0., conv_dropout = 0.):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

class Conv2dSubsampling(nn.Module):
    '''
    convolustional 2d subsampling to 1/4 length
    '''
    def __init__(self, in_channels, out_channels):
        super(Conv2dSubsampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

        return outputs

class ConformerEncoder(nn.Module):
    '''
    process the input with a convolution subsampling layer and then with a number of conformer blocks
    '''
    def __init__(self, input_dim, encoder_dim, num_layers, num_attention_heads, ff_mult, conv_expansion_factor,
                 conv_kernel_size, input_dropout, attn_dropout, ff_dropout, conv_dropout):
        super(ConformerEncoder, self).__init__()
        self.layers = nn.ModuleList([ConformerBlock(dim=encoder_dim, heads = num_attention_heads, dim_head= int(encoder_dim/num_attention_heads),
                                                    ff_mult=ff_mult, conv_expansion_factor=conv_expansion_factor, conv_kernel_size=conv_kernel_size,
                                                    attn_dropout=attn_dropout, ff_dropout=ff_dropout, conv_dropout=conv_dropout) for _ in range(num_layers)])

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

class Conformer(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 80, encoder_dim: int = 512, num_encoder_layers: int = 17, num_attention_heads: int = 8, feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2, input_dropout_p: float = 0.1, feed_forward_dropout_p: float = 0.1, attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1, conv_kernel_size: int = 17, half_step_residual: bool = True,):
        super(Conformer, self).__init__()

        cnns = [nn.Dropout(0.1),
                nn.Conv1d(40, 64, 3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()]
        cnns += [nn.Dropout(0.1),
                 nn.Conv1d(64, 256, 3, stride=1, padding=1),
                 nn.BatchNorm1d(256),
                 nn.ReLU()]
        cnns += [nn.Dropout(0.1),
                 nn.Conv1d(256, 256, 3, stride=1, padding=1),
                 nn.BatchNorm1d(256),
                 nn.ReLU()]

        self.cnns = nn.Sequential(*nn.ModuleList(cnns))

        self.encoder = ConformerEncoder(
            input_dim=input_dim, encoder_dim=encoder_dim, num_layers=num_encoder_layers, num_attention_heads=num_attention_heads,
            ff_mult=feed_forward_expansion_factor, conv_expansion_factor=conv_expansion_factor, conv_kernel_size=conv_kernel_size,
            input_dropout=input_dropout_p, attn_dropout=attention_dropout_p, ff_dropout=feed_forward_dropout_p, conv_dropout=conv_dropout_p
        )
        self.fc = nn.Linear(encoder_dim, num_classes)

        self.preprocess = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        # compute MFCC and perform mean variance normalisation
        with torch.no_grad():
            x = self.preprocess(inputs) + 1e-6
            x = self.instancenorm(x).detach()
        cnn_out = self.cnns(x).transpose(1, 2) # batch, seq, feature

        encoder_outputs = self.encoder(cnn_out)
        outputs = self.fc(encoder_outputs)
        return outputs

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

## ===================================================================
## Transformer-related functions and classes
## reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
## ===================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout):
        super().__init__()

        cnns = [nn.Dropout(0.1),
                nn.Conv1d(40, 64, 3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()]

        for i in range(2):
            cnns += [nn.Dropout(0.1),
                     nn.Conv1d(64, 64, 3, stride=1, padding=1),
                     nn.BatchNorm1d(64),
                     nn.ReLU()]

        # define CNN layers
        self.cnns = nn.Sequential(*nn.ModuleList(cnns))

        # define transformer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.preprocess = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # compute MFCC and perform mean variance normalisation
        with torch.no_grad():
            x = self.preprocess(x) + 1e-6
            x = self.instancenorm(x).detach()

        cnn_out = self.cnns(x).transpose(1, 2) # (batch, seq, feature)
        cnn_out = cnn_out.long() * math.sqrt(self.d_model)

        # src = self.encoder(cnn_out) * math.sqrt(self.d_model)
        pos_out = self.pos_encoder(cnn_out)
        encoded = self.transformer_encoder(pos_out)
        output = self.decoder(encoded)

        return output

## ===================================================================
## Baseline speech recognition model
## ===================================================================
class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_classes=11):
        super(SpeechRecognitionModel, self).__init__()

        cnns = [nn.Dropout(0.1),
                nn.Conv1d(40, 64, 3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()]

        for i in range(2):
            cnns += [nn.Dropout(0.1),
                     nn.Conv1d(64, 64, 3, stride=1, padding=1),
                     nn.BatchNorm1d(64),
                     nn.ReLU()]

        ## define CNN layers
        self.cnns = nn.Sequential(*nn.ModuleList(cnns))

        ## define RNN layers as self.lstm - use a 3-layer bidirectional LSTM with 256 output size and 0.1 dropout
        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=256, dropout=0.1, num_layers=3, bidirectional=True, batch_first=True)

        ## define the fully connected layer
        self.classifier = nn.Linear(512, n_classes)

        self.preprocess = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

    def forward(self, x):
        ## compute MFCC and perform mean variance normalisation
        with torch.no_grad():
            x = self.preprocess(x) + 1e-6
            x = self.instancenorm(x).detach()

        ## pass the network through the CNN layers
        cnn_out = self.cnns(x)

        ## pass the network through the RNN layers - check the input dimensions of nn.LSTM()
        lstm_out = self.lstm(cnn_out.transpose(1, 2))[0]

        ## pass the network through the classifier
        class_out = self.classifier(lstm_out)

        return class_out


## ===================================================================
## Train an epoch on GPU
## ===================================================================
def process_epoch(model, loader, criterion, optimizer, trainmode=True):
    # Set the model to training or eval mode
    if trainmode:
        model.train()
    else:
        model.eval()

    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2700)

    ep_loss = 0
    ep_cnt = 0

    with tqdm(loader, unit="batch") as tepoch:
        for data in tepoch:
            ## Load x and y
            x = data[0].cuda()
            y = data[1].cuda()
            y_len = torch.LongTensor(data[3]).cuda()

            # add some noise to x
            x = x + torch.normal(mean=0, std=torch.std(x) * 1e-3, size=x.shape).cuda()

            # forward pass
            logits = model(x)
            output = torch.nn.functional.log_softmax(logits, dim=2)
            output = output.transpose(0, 1)

            ## compute the loss using the CTC objective
            x_len = torch.LongTensor([output.size(0)]).repeat(output.size(1))
            loss = criterion(output, y, x_len, y_len)

            if trainmode:
                # backward pass
                loss.backward()

                # optimizer step
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent gradient exploding
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # keep running average of loss
            ep_loss += loss.detach() * len(x)
            ep_cnt += len(x)

            # print value to TQDM
            tepoch.set_postfix(loss=ep_loss.item() / ep_cnt)

    return ep_loss.item() / ep_cnt


## ===================================================================
## Greedy CTC Decoder
## ===================================================================
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """
        Given a sequence emission over labels, get the best path.
        """
        # find the index of the maximum probability output at each time step
        max_prob = torch.argmax(emission, dim=-1)

        # remove the repeats
        removed_repeats = torch.unique_consecutive(max_prob, dim=-1)

        # convert to numpy array
        removed_repeats = np.array(removed_repeats)

        # remove the blank symbols
        indices = [i for i in removed_repeats if i != self.blank]

        return indices


## ===================================================================
## Evaluation script
## ===================================================================
def process_eval(model, data_path, data_list, index2char, save_path=None, test_time_adaptation =False):
    if test_time_adaptation:
        # load model
        model = configure_model(model)

        # load data from JSON
        with open(data_list, 'r') as f:
            data = json.load(f)

        # HYPERPARAMETERS
        tta_batch_size = 100
        tta_lr = 2e-5
        em_coef=0.3
        not_blank=True # do not consider blank
        temp = 2.5

        # create dataloader for TENT
        dataset = EvalSpeechDataset(data_list, data_path, 10)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_sampler=BucketingSampler(dataset, tta_batch_size), # 100: batch size
                                             num_workers=6,
                                             collate_fn=eval_pad_collate)
        # collect parameters
        params, _ = collect_params(model)
        optimizer = torch.optim.Adam(params, lr=tta_lr, weight_decay=1e-5)

        # adapt using entropy minimization
        for epoch in range(0, 10):
            ep_loss = 0
            ep_cnt = 0
            with tqdm(loader, unit="batch") as tepoch:
                for audio in tepoch:
                    x = audio[0].cuda()

                    # add some noise to x
                    x = x + torch.normal(mean=0, std=torch.std(x) * 1e-3, size=x.shape).cuda()
                    logits = model(x)

                    predicted_ids = torch.argmax(logits, dim=-1)
                    non_blank = torch.where(predicted_ids != len(index2char), 1, 0).bool()

                    # adapt
                    loss = 0
                    if em_coef > 0:
                        if not_blank:
                            e_loss = softmax_entropy(logits / temp)[non_blank].mean(0).mean()
                        else:
                            e_loss = softmax_entropy(logits / temp).mean(0).mean()
                        loss += e_loss * em_coef
                    if 1 - em_coef > 0:
                        c_loss = mcc_loss(logits / temp, class_num=len(index2char) + 1)
                        loss += c_loss * (1 - em_coef)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # loss calculate
                    ep_loss += loss.detach() * len(x)
                    ep_cnt += len(x)
                    tepoch.set_postfix(loss=ep_loss.item() / ep_cnt)
            print('Epoch {:03d}, loss: {:.3f}'.format(epoch, ep_loss.item() / ep_cnt))
    else:
        # set model to evaluation mode
        model.eval()
        # load data from JSON
        with open(data_list, 'r') as f:
            data = json.load(f)

    # evaluate
    # initialise the greedy decoder
    greedy_decoder = GreedyCTCDecoder(blank=len(index2char))
    results = []

    for file in tqdm(data):
        # read the wav file and convert to PyTorch format
        audio, sample_rate = soundfile.read(os.path.join(data_path, file['file']))
        x = torch.FloatTensor(audio).cuda()
        x = x.unsqueeze(dim=0)
        # add some noise
        x = x + torch.normal(mean=0, std=torch.std(x) * 1e-3, size=x.shape).cuda()

        # forward pass through the model
        with torch.no_grad():
            logits = model(x)
            pred = torch.nn.functional.log_softmax(logits, dim=2)
            pred = pred.transpose(0, 1)

        # decode using the greedy decoder
        pred = greedy_decoder(pred.cpu().detach().squeeze())

        # convert to text
        out_text = ''.join([index2char[x] for x in pred])

        # keep log of the results
        file['pred'] = out_text
        if 'text' in file:
            file['edit_dist'] = editdistance.eval(out_text.replace(' ', ''), file['text'].replace(' ', ''))
            file['gt_len'] = len(file['text'].replace(' ', ''))
        results.append(file)

    # save results to json file
    with open(os.path.join(save_path, 'results.json'), 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    # print CER if there is ground truth
    if 'text' in file:
        cer = sum([x['edit_dist'] for x in results]) / sum([x['gt_len'] for x in results])
        print('Character Error Rate is {:.2f}%'.format(cer * 100))

## ===================================================================
## Deploy server script
## ===================================================================
def deploy_server(model, index2char, port):
    # initialise the greedy decoder
    greedy_decoder = GreedyCTCDecoder(blank=len(index2char))

    # create the Flask app
    app = Flask(__name__)

    @app.route('/query-window', methods=['POST'])
    def process_chunk():
        # unpack the received data
        data = pickle.loads(request.get_data())
        # convert to PyTorch format
        data = torch.FloatTensor(data).cuda()
        # forward pass through the model
        with torch.no_grad():
            logits = model(data)
            output = torch.nn.functional.log_softmax(logits, dim=2)
            output = output.transpose(0, 1)

        # decode using the greedy decoder
        pred = greedy_decoder(output.cpu().detach().squeeze())

        # join the index
        out_text = ''.join([index2char[x] for x in pred])
        print('Result:', out_text)

        return out_text

    app.run(host='0.0.0.0', debug=True, port=port, threaded=False)

## ===================================================================
## Main execution script
## ===================================================================
def main():
    parser = argparse.ArgumentParser(description='EE738 Exercise')

    ## gpu configuration
    parser.add_argument('--gpu_idx', type=int, default=0, help='index of gpu to use')

    ## related to data loading
    parser.add_argument('--max_length', type=int, default=10, help='maximum length of audio file in seconds')
    parser.add_argument('--train_list', type=str, default='./datasets/clovacall/cc_train.json')
    parser.add_argument('--val_list', type=str, default='./datasets/clovacall/cc_val.json')
    parser.add_argument('--train_path', type=str, default='./datasets/clovacall/wavs_train')
    parser.add_argument('--val_path', type=str, default='./datasets/clovacall/wavs_train')
    parser.add_argument('--labels_path', type=str, default='./datasets/char_dict/label.json')

    ## related to training
    parser.add_argument('--max_epoch', type=int, default=10, help='number of epochs during training')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=2222, help='random seed initialisation')

    ## relating to loading and saving
    parser.add_argument('--initial_model', type=str, default='', help='load initial model, e.g. for finetuning')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='location to save checkpoints')
    parser.add_argument('--log_suffix', type=str, default='debug', help='suffix used for logging data and models')

    ## related to inference and deploying server
    parser.add_argument('--eval', dest='eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--server', dest='server', action='store_true', help='Server mode')
    parser.add_argument('--port', type=int, default=10000, help='Port for the server')

    ## model
    parser.add_argument('--model', type=str, default='convlstm', help='model type: convlstm, transformer, conformer')
    parser.add_argument('--interval', type=int, default=10, help='interval to save and evaluate model')
    parser.add_argument('--tta', action='store_true', help='whether to do test time adaptation or not')

    args = parser.parse_args()

    # device configuration
    device = torch.device("cuda:{:d}".format(args.gpu_idx) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu_idx) # prevents unnecessary gpu memory allocation to cuda:0
    print(device)

    # load labels
    char2index, index2char = load_label_json(args.labels_path)

    ## make an instance of the model on GPU
    print("num classes:", len(char2index) + 1)
    if args.model == 'convlstm':
        model = SpeechRecognitionModel(n_classes=len(char2index) + 1).cuda()
    elif args.model == 'transformer':
        model = TransformerModel(ntoken=len(char2index) + 1, d_model=64, nhead=2, d_hid=512, nlayers=2, dropout=0.1).cuda()
    elif args.model == 'conformer':
        model = Conformer(num_classes=len(char2index) + 1, input_dim=64, encoder_dim=256, num_encoder_layers=3,
                          num_attention_heads=8).cuda()
    print('Model loaded. Number of parameters:', sum(p.numel() for p in model.parameters()))

    ## load from initial model
    if args.initial_model != '':
        model.load_state_dict(torch.load(args.initial_model))

    ## code for server
    if args.server:
        deploy_server(model, index2char, args.port)
        quit();

    # make directory for saving models and output
    assert args.save_path != ''
    assert args.log_suffix != ''
    args.log_path = os.path.join(args.save_path, args.log_suffix)
    os.makedirs(args.log_path, exist_ok=True)

    ## code for inference - this uses val_path and val_list
    if args.eval:
        process_eval(model, args.val_path, args.val_list, index2char, save_path=args.log_path, test_time_adaptation=args.tta)
        quit();

    # initialise seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # define datasets
    trainset = SpeechDataset(args.train_list, args.train_path, args.max_length, char2index)
    valset = SpeechDataset(args.val_list, args.val_path, args.max_length, char2index)

    # initiate loader for each dataset with 'collate_fn' argument
    # do not use more than 6 workers
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_sampler=BucketingSampler(trainset, args.batch_size),
                                              num_workers=6,
                                              collate_fn=pad_collate)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_sampler=BucketingSampler(valset, args.batch_size),
                                            num_workers=6,
                                            collate_fn=pad_collate)
    ## define the optimizer with args.lr learning rate and appropriate weight decay
    optimizer = None
    if args.model == 'convlstm' or args.model == 'transformer':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.model == 'conformer':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    ## set loss function with blank index
    criterion = torch.nn.CTCLoss(blank=len(index2char)).cuda()

    ## initialise training log file
    f_log = open(os.path.join(args.log_path, 'train.log'), 'w')
    f_log.write('{}\n'.format(args))
    f_log.flush()

    ## Train for args.max_epoch epochs
    for epoch in range(0, args.max_epoch):
        print('Training epoch', epoch)
        tloss = process_epoch(model, trainloader, criterion, optimizer, trainmode=True)
        print('Trained epoch', epoch, 'Loss', tloss)

        print('Validating epoch', epoch)
        vloss = process_epoch(model, valloader, criterion, optimizer, trainmode=False)
        print('Validated epoch', epoch, 'Loss', vloss)

        # save checkpoint to file # save at the last epoch
        if epoch == args.max_epoch - 1 or epoch % args.interval == 0:
            if epoch == 0:
                continue
            save_file = '{}/model{:05d}.pt'.format(args.log_path, epoch)
            print('Saving model {}'.format(save_file))
            torch.save(model.state_dict(), save_file)

        # write training progress to log
        f_log.write('Epoch {:03d}, train loss {:.3f}, val loss {:.3f}\n'.format(epoch, tloss, vloss))
        f_log.flush()

    f_log.close()

if __name__ == "__main__":
    main()