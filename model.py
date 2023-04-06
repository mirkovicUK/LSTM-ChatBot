import torch
import torch.nn as nn
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_dim, n_layers, dropout):
        
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = True, 
            dropout = (0 if n_layers == 1 else dropout),
        )        
    
    def forward(self, inputs, lengths, hidden=None):
        embedding = self.embedding(inputs)
        #Pack padded batch for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True)
        # get hidden tuple from lstm
        _, hidden = self.lstm(packed, hidden)
        
        return hidden[0] # return hidden state only

class Decoder(nn.Module):
      
    def __init__(self, hidden_size, output_size, embedding_dim, n_layers, dropout):
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = True, 
            dropout = (0 if n_layers == 1 else dropout),
        )

        self.fc = nn.Linear(hidden_size, output_size)     
        
    def forward(self, inputs, encoder_hidden):
        #embeddings
        embedding = self.embedding(inputs)
        #lstm
        output,hidden = self.lstm(embedding, encoder_hidden)
        # fc layer
        output = self.fc(output)
        
        return output, hidden
        
       

class Seq2Seq(nn.Module):
    
    def __init__(
            self, 
            encoder_input_size, 
            encoder_hidden_size, 
            embedding_dim, 
            n_layers, 
            dropout, 
            decoder_hidden_size, 
            decoder_output_size):
        
        self.output_size= decoder_output_size
        
        super(Seq2Seq, self).__init__()
        
        print('Building encoder and decoder ...')
        self.encoder = Encoder(
            input_size = encoder_input_size, 
            hidden_size = encoder_hidden_size,
            embedding_dim = embedding_dim,
            n_layers = n_layers,
            dropout = dropout
            )
        
        self.decoder = Decoder(
            hidden_size = decoder_hidden_size,
            output_size=decoder_output_size,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        print('Models built and ready to go!')
    
    def forward(self, inputs, targets, max_target_len, inputs_lenghts, teacher_forcing_ratio = 0.5):      
        # Forward pass through encoder
        encoder_hidden = self.encoder(inputs,inputs_lenghts)
        #encoder returns only hidden state so build hidden for decoder
        decoder_hidden = (encoder_hidden, torch.zeros_like(encoder_hidden))
        #initialize decoder inputs
        batch_size = inputs.shape[0]
        decoder_inputs = torch.ones(batch_size,1, device=device, dtype=torch.long)
        #tensor to append decoder outputs
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        
        # get 1 token from decoder on each iteration 
        for i in range(max_target_len):
            #forward decoder
            decoder_output, decoder_hidden = self.decoder(decoder_inputs, decoder_hidden)
            #pick class index with highest score
            _, decoder_inputs = torch.max(decoder_output, dim=-1)
            #record decoder token from this iteration
            all_tokens = torch.cat((all_tokens, decoder_output), dim=1)
            #teacher forcing only when model is in training mode
            if self.encoder.training and self.decoder.training:
                teacher_forsing = True if random.random() < teacher_forcing_ratio else False
                if teacher_forsing:
                    decoder_inputs = targets[:,i].unsqueeze(1) 
                else: decoder_inputs

        return all_tokens