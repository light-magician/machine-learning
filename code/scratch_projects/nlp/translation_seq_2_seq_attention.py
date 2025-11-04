# coding: utf-8
# sequence to sequence network
get_ipython().run_cell_magic('capture', '', '# sequence to sequence network\n# in which two recurrent neural nets work together\n# to transform on sequence to another\n# an encoder network condenses an input seq to a vec\n# and a decoder net unfolds that vec into a new seq\n\n')
get_ipython().run_cell_magic('capture', '', '# this impl will include an attention mechanism\n# whih lets the decoder learn ot focus over a specific\n# range of the input sequence\n\n')
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
get_ipython().run_line_magic('clear', '')
get_ipython().run_cell_magic('capture', '', '# represent each word as a one hot encoding\n# -> a giant vector of zeros except for a single 1\n# the 1 is just the index of the word\n\n')
get_ipython().run_cell_magic('capture', '', '# start and end of seq tokens\nSOS_token = 0\nEOS_token = 1\n\n')
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n        \n')
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n        \n')
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n    \n')
get_ipython().run_line_magic('clear', '')
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n        \n    def addSentence(self, sentence):\n        for word in sentence.split(\' \'):\n            self.adWord(word)\n            \n    def addWord(self, word):\n        if word not in self.word2index:\n            self.word2index[word] = self.num_words\n            self.word2count[word] = 1\n            self.index2word[self.num_words] = word\n            self.num_words += 1\n        else:\n            self.word2count[word] += 1\n                \n')
get_ipython().run_cell_magic('capture', '', "# turn unicode string to ASCII\ndef unicodeToAscii(s):\n    return ''.join(\n    )\n")
get_ipython().run_cell_magic('capture', '', "# turn unicode string to ASCII\ndef unicodeToAscii(s):\n    return ''.join(\n        c for c in unicodedata.normalize('NFD', s)\n        if unicodedata.category(c) != 'Mn'\n    )\n    \n")
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sib(r"([.!?]), r" \1", s)
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sib(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()
    
def readLangs(lang1, lang2, reverse=False):
    # files will be in eglish and some other lang
    # split the file into lines, and split lines into pairs
    print("reading lines...")
    # read the file and split into lines, searching for langs
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    
get_ipython().run_line_magic('clear', '')
def readLangs(lang1, lang2, reverse=False):
    # files will be in eglish and some other lang
    # split the file into lines, and split lines into pairs
    print("reading lines...")
    # read the file and split into lines, searching for langs
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # split lines into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
def readLangs(lang1, lang2, reverse=False):
    # files will be in eglish and some other lang
    # split the file into lines, and split lines into pairs
    print("reading lines...")
    # read the file and split into lines, searching for langs
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # split lines into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # reverse pairs, make lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input lang = Lang(lang1)
def readLangs(lang1, lang2, reverse=False):
    # files will be in eglish and some other lang
    # split the file into lines, and split lines into pairs
    print("reading lines...")
    # read the file and split into lines, searching for langs
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # split lines into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # reverse pairs, make lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs
    
get_ipython().run_line_magic('clear', '')
get_ipython().run_cell_magic('capture', '', '# there are many samples, so to train quickly we trim dataset\n# to only short sentences\n# here max length is 10 words\nMAX_LENGTH = 10\neng_prefixes = (\n"i am ", "i m ", "he is ", "he is ", "she is ", "she s ", "you are ", "you re ",\n"we are ", "we re ", "they are ", "they re "\n)\n\n')
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and 
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)
        
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
    
def prepareData(lang1, lang2, reverse=False):
    # read text file, split into lines, split into pairs
    # normalize text, filter by len and content
    # make word lists from sentences in pairs
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("trimmed to %s sentence pairs" % len(pairs))
    print("counting words...")
    for pair in pairs:
        inpuit_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs
    
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
get_ipython().run_line_magic('clear', '')
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()
    
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
def prepareData(lang1, lang2, reverse=False):
    # read text file, split into lines, split into pairs
    # normalize text, filter by len and content
    # make word lists from sentences in pairs
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("trimmed to %s sentence pairs" % len(pairs))
    print("counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs
    
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n        \n    def addSentence(self, sentence):\n        for word in sentence.split(\' \'):\n            self.addWord(word)\n            \n    def addWord(self, word):\n        if word not in self.word2index:\n            self.word2index[word] = self.num_words\n            self.word2count[word] = 1\n            self.index2word[self.num_words] = word\n            self.num_words += 1\n        else:\n            self.word2count[word] += 1\n            \n')
get_ipython().run_line_magic('clear', '')
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
def prepareData(lang1, lang2, reverse=False):
    # read text file, split into lines, split into pairs
    # normalize text, filter by len and content
    # make word lists from sentences in pairs
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("trimmed to %s sentence pairs" % len(pairs))
    print("counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("counted words...")
    print(input_lang.name, input_lang.num_words)
    print(output_lang.name, output_lang.num_words)
    return input_lang, output_lang, pairs
    
get_ipython().run_line_magic('clear', '')
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
get_ipython().run_line_magic('save', 'translation_seq_2_seq_attention.py')
get_ipython().run_line_magic('clear', '')
get_ipython().run_cell_magic('capture', '', '# an RNN is a net that operates on a sequence and\n# uses its own output as input for subsequent steps\n# a Seq2Seq net operates with 2 RNNs encoder and decoder\n# encoder maps to lower dimensional space and decoder\n# maps back to higher dimensional space\n\n')
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        
def forward(self, input):
    embedded = self.dropout(self.embedding(input))
    output, hidden = self.gru(embedded)
    return output, hidden
    
get_ipython().run_cell_magic('capture', '', '# decoder will use last output of encoder called "context vector"\n# context vector is used as initial hidden state of decoder\n# at every step of decoding, the decoder is given an input token\n# and a hidden state\n# initial input token is SOS, initial hidden state is context vec\n\n')
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None
     def forward_step(self, input, hidden):
get_ipython().run_line_magic('clear', '')
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None
    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
        
get_ipython().run_cell_magic('capture', '', '# if we stopped here the model would be bad\n# lets go straight for the gold and add attention\n\n')
'''
Attention allows the decoder network to 'focus' on diferent part of the encoder's output
for every step of the decoder's own outputs. 
Attention requres the calculation of a set of 'attention weights'. These are multiplied
by the encoder output vectors to create a weighted combination.
The result 'attn_applied' should contain info about that specific part of the input
sequence, and thus help the decoder choose the right output words.
'''
