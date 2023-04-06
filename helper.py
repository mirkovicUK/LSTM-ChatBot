##############################################################################
#read process data 
##############################################################################
import json
def get_QA(data_path):
    """
    Function to read json data, return list of question answer pairs    
    """
    with open(data_path, 'r') as f:
        for line in f:
            l = json.loads(line)

    questions, answers = list(), list()

    #topicks that questions are from not returned
    topicks = [] 
    for i in range(len(l['data'])):
        topicks.append(l['data'][i]['title'])

    #extract questions and answers from data
    for i in range(len(l['data'])):
        for j in range(len(l['data'][i]['paragraphs'])):
            for k in range(len(l['data'][i]['paragraphs'][j]['qas'])):
                questions.append(l['data'][i]['paragraphs'][j]['qas'][k]['question'])
                answers.append(l['data'][i]['paragraphs'][j]['qas'][k]['answers'])
    
    # build QA pairs
    pairs = list(zip(questions, answers))
    #remove pairs with empty answers
    for pair in pairs[:]:
        if len(pair[1]) == 0:
            pairs.remove(pair)

    # pick up only strings from answers
    answ = list()
    for pair in pairs:
        answ.append(pair[1][0]['text'])

    # build final pairs of QA
    p = []
    for i in range(len(pairs)):
        text = [[pairs[i][0]], [answ[i]]]
        p.append(text)


    return p
#############################################################################
import codecs
import csv
def to_csw(pairs, path):
    """Build csw file with pairs of questions and answers list
        use above function get_QA() to get pairs
    """
    delimiter ='\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    with open(path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=delimiter, lineterminator='\n')
        for i in range(len(pairs)):
            pair = [(pairs[i][0][0]), (pairs[i][1][0])]
            writer.writerow(pair)


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

import re

def normalizeString(s):
    """# Lowercase, trim, and remove non-letter characters"""
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s  

def get_pairs(data_path):
    """
    Read in QA.text normalize data remove empty data points after normalization
    """
    pairs = list()
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            pairs.append([normalizeString(s) for s in line.split('\t')])
    #after normalization some data point are empty 
    #remove empty pairs 
    for pair in pairs[:]:
        if len(pair[0]) == 0 or len(pair[1]) == 0:
            pairs.remove(pair)

    return pairs



############################################################################

def filter_pairs(pairs, trim):
    """keep only sentences with lenght less than trim"""

    print('Read {} sentence pairs'.format(len(pairs)))
    print(f'Trimming pairs with sentence longer than MAX_LEN({trim})......')
    x = [pair for pair in pairs if len(pair[0].split()) < trim[0] and len(pair[1].split()) < trim[1]]
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
    print(json.dumps(l))

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
    return inp.T, lengths, output.T, max_target_len

############################################################################
#custom data loaders
############################################################################
def data_loaders(data, vocab, batch_size):
    """
    return: all from above function as genetrator object 
    input tensor , shape(seq, batch_size)
    lenghts tensor 
    target tensor
    max target len for batch
    """
    for i in range(0, len(data), batch_size):
        yield pairs2TrainData(data[i:i+batch_size], vocab)


#############################################################################################

######################################################################################
#Embedings initializer
######################################################################################
import numpy as np
def initialize_embeddings(w2v, embeddings, vocab):
    """Initialize embeddings with pretrained weights , as per project requirements
    """
    counter = 0
    for word in vocab.word2int:
        try:
            embeddings.weight.data[vocab.word2int[word]] = torch.from_numpy(np.copy(w2v.wv[word]))
            counter +=1
        except KeyError:
            pass
    print(f'Initializet total {counter} embeddings with brown embedings, out of total {len(w2v.wv)}')
        
#####################################################################################
# chatbot comunication protocol 
######################################################################################
def evaluate(model, vocab, sentence, max_len=7):
    idx = [SOS_token] + [vocab.word2int[x] for x in sentence.split()] + [EOS_token]
    input_tensor = torch.LongTensor(idx).unsqueeze(0)
    lenghts_tensor = torch.LongTensor([input_tensor.shape[0]])
    targets = None
    output = model(input_tensor,targets, max_len, lenghts_tensor)
    _, output = torch.max(output, dim=-1)
    words = output.tolist()

    a = []
    print(words)
    for ii in words[0]:
        word = vocab.int2word[ii]
        a.append(word)
    return a

def evaluateInput(model, vocab):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            print(input_sentence)
            output_sentence = evaluate(model, vocab, input_sentence)
            
            out = [word for word in output_sentence if word!='EOS' and word!='PAD' and word!='SOS']
            print('Bot:', ' '.join(out))

        except KeyError:
            print("Error: Encountered unknown word.")