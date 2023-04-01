##############################################################################
#read data
##############################################################################
import json
def get_QA(path):
    """Function to read data, return question and answers list"""
    with open(path, 'r') as f:
        for line in f:
            l = json.loads(line)

    questions, answers = list(), list()

    #topicks that questions are from 
    topicks = [] 
    for i in range(len(l['data'])):
        topicks.append(l['data'][i]['title'])

    #extract questions and answers from data
    for i in range(len(l['data'])):
        for j in range(len(l['data'][i]['paragraphs'])):
            for k in range(len(l['data'][i]['paragraphs'][j]['qas'])):
                questions.append(l['data'][i]['paragraphs'][j]['qas'][k]['question'])
                answers.append(l['data'][i]['paragraphs'][j]['qas'][k]['answers'])

    #extract only answers from list
    a = []
    for item in answers:
        if not item:
            a.append(None)# some answers are empty lists
        else:
            a.append(item[0]['text'])
    return questions, a


############################################################################
#normalizing data
############################################################################
import re
import unicodedata

def unicodeToAscii(s):
    """Unicode string to plain ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """# Lowercase, trim, and remove non-letter characters"""
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s 


############################################################################

def filter_pairs(pairs, trim:int):
    """keep only sentences with lenght less than trim"""

    print('Read {} sentence pairs'.format(len(pairs)))
    print(f'Trimming pairs with sentence longer than MAX_LEN({trim})......')
    x = [pair for pair in pairs if len(pair[0].split()) < trim and len(pair[1].split()) < trim]
    print('trimmed to {} pairs'.format(len(x)))
    
    return x

############################################################################
#vizualization
############################################################################
def print_data_for_visualization(path = '.data/train-v2.0.json'):
    
    path = '.data/train-v2.0.json'
    with open(path, 'r') as f:
        for line in f:
            l = json.loads(line)
            #a = l['data']
    print(json.dumps(l, indent=2, sort_keys=True))

#############################################################################
# data to tensor
#############################################################################
import torch 
import itertools

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def encode(sentence, vocab):
    return [SOS_token] + [vocab.word2int[x] for x in sentence.split()] + [EOS_token]

def zero_padding(batch, fillvalue=PAD_token):
    """
    Flips input. batch.shape=(batch_size, sequence_len)
    returns (sequence_len, batch_size)
    """
    return list(itertools.zip_longest(*batch, fillvalue=fillvalue))


def input_tensor(x, vocab):
    encoded = [encode(sentence, vocab) for sentence in x]
    return (
        torch.LongTensor(zero_padding(encoded)), # input_var tensor
        torch.tensor(list(map(len,encoded)))   # lengths 
    )

def output_tensor(x, vocab):
    encoded = [encode(sentence, vocab) for sentence in x]
    seq_len = max(map(len, encoded))
    return(
        torch.LongTensor(zero_padding(encoded)),
        seq_len
    )

def pairs2TrainData(batch_pairs, vocab):
    batch_pairs.sort(key=lambda x: len(x[0].split()), reverse=True)
    input_batch, output_batch = [], []
    for pair in batch_pairs:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_tensor(input_batch, vocab)
    output, max_target_len = output_tensor(output_batch, vocab)
    return inp, lengths, output, max_target_len

############################################################################
#custom data loaders
############################################################################
def data_loaders(data, vocab, batch_size):
    """
    return: all from above function as genetrator object 
    input tensor , shape(seq, batch_size)
    lenghts tensor 
    target tensor
    max target len
    """
    for i in range(0, len(data), batch_size):
        yield pairs2TrainData(data[i:i+batch_size], vocab)

################################################################################
#encoder
################################################################################
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size,embeddings, embedding_dim, hidden_size, dropout , n_layers) -> None:
        super(Encoder, self).__init__()
        self.input_size = input_size # len(vocab)
        self.hidden_size = hidden_size
        self.drop_prob = dropout
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim


        self.embed = embeddings
        #LSTM
        self.lstm = nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_size,
            num_layers = self.n_layers,
            batch_first = False, # input shape (seq_len, batch_size, embedding_dim)
            dropout = (0 if n_layers == 1 else dropout),
        )
        self.dropout = nn.Dropout(p=self.drop_prob)

    
    def forward(self, x, lengths, hidden=None):
        #run input trought embeddings with dropout
        # x.shape = seq_len, batch_size
        embedding = self.dropout(self.embed(x)) #x.shape (seq_len, batch_size, embedding_dim)
        # Pack padded batch of sequences for RNN  
        packed = nn.utils.rnn.pack_padded_sequence(embedding, lengths=lengths)
        #only take hidden state
        output, hidden = self.lstm(packed, hidden) # hidden.shape (n_layers, batch_size, hidden_size)
        #unpack padding
        output,_ = nn.utils.rnn.pad_packed_sequence(output)
        output = self.dropout(output)

        return output, hidden # return embedding matrix reuse it in decoder
    
######################################################################################
#DECODER
######################################################################################
import torch.nn.functional as F
import torch

class Decoder(nn.Module):
    def __init__(self, embeddings, hidden_size, embeding_dim, output_size, dropout, n_layers) -> None:
        super(Decoder, self).__init__()

        #keep:
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dim = embeding_dim
        
        #Define layers
        self.embed = embeddings #use same embedings as encoder
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_size,
            num_layers = self.n_layers,
            batch_first = False, # input shape (seq_len, batch_size, embedding_dim)
            dropout = (0 if n_layers == 1 else dropout),
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, encoder_hidden):
        embedded = self.dropout(self.embed(x))
        #forward pass trought LSTM cells
        output, hidden = self.lstm(embedded, encoder_hidden)
        #forward fc
        output = self.dropout(self.fc(output))

        return output, hidden
    
#####################################################################################################
#Seq2seq
#####################################################################################################
import torch
import torch.nn as nn
from helper import Encoder, Decoder
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Seq2seq(nn.Module):
    def __init__(self, input_size, embeddings, embedding_dim, hidden_size, output_size, dropout = 0.25, n_layers=2) -> None:
        super(Seq2seq, self).__init__()

        print('Building encoder and decoder ...')
        self.encoder = Encoder(input_size, embeddings, embedding_dim, hidden_size, dropout, n_layers)
        self.decoder = Decoder(embeddings, hidden_size, embedding_dim ,output_size, dropout, n_layers)
        print('Models built and ready to go!')

    def forward(self, inputs, targets, max_target_len, inputs_lenghts, teacher_forcing_ratio = 0.5):
        # Forward pass through encoder
        _, encoder_hidden = self.encoder(inputs, inputs_lenghts)
        #initialize decoder inputs
        if self.encoder.lstm.batch_first:
            decoder_inputs = torch.ones(inputs.shape[0],1, device=device, dtype=torch.long) *SOS_token
        else:
            decoder_inputs = torch.ones(1,inputs.shape[1], device=device, dtype=torch.long)*SOS_token
        
        #tensor to append decoder outputs
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        #set decoder hidden state to encoder last hidden
        decoder_hidden=encoder_hidden

        # get 1 token from decoder on each iteration 
        for i in range(max_target_len):
            #forwar pass trought Decoder
            decoder_output, decoder_hidden = self.decoder(decoder_inputs, decoder_hidden)
            decoder_output.squeeze_(dim=0)
            #set decoder input to index of class with highest score 
            _, decoder_inputs = torch.max(decoder_output, dim=1)
            # shape input (seq_len, Batch_size)
            decoder_inputs.unsqueeze_(dim=0)
            #record token for this iteration all_tokens.shape = (seq_len, batch_size)
            all_tokens = torch.cat((all_tokens, decoder_inputs), dim=0)
            
            #teacher_forcing
            teacher_forsing = True if random.random() < teacher_forcing_ratio else False
            decoder_inputs = targets[i].unsqueeze_(0) if teacher_forsing else decoder_inputs

        return all_tokens

