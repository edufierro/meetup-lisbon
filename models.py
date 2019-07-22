# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class OurAwsomeRNN(nn.Module):
    """
    Our simple RNN model in pytorch
    For prediction, it only uses the last hidden layer
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, device):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embeddings
        @param hidden_dim: size of our hidden layers
        @param output_dim: size of the expected output
        @param device: device to be used by pytorch
        """
        super(OurAwsomeRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.device = device
        
        # See the full documentation at: https://pytorch.org/docs/stable/nn.html
        self.embedding = nn.Embedding(self.vocab_size + 3, self.emb_dim, padding_idx=0)
        self.rnn = nn.RNN(self.emb_dim, self.hidden_dim, num_layers=1, batch_first=True)   
        self.linear = nn.Linear(self.hidden_dim, 1)     # Why map to 1?
        self.sigmoid = nn.Sigmoid()     
    
    def forward(self, x):
        """
        Our forwad function in pytorch.  
        """
        
        batch_size = x.size()[0]
        embeddings = self.embedding(x)            # [BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE]
        h_0 = self.init_hidden(batch_size)        # [NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]
        _, h_n = self.rnn(embeddings, h_0)        # h_n: [1, BATCH_SIZE, HIDDEN_SIZE] 
        output = self.linear(h_n)                 # output: [1, BATCH_SIZE, 1] 
        return self.sigmoid(output.view(-1))      # output: [BATCH_SIZE] 
    
    def init_hidden(self, batch_size, num_layers=1):
        """
        Initialize our first hidden layer with zeros.
        This might be unecessary, pytorch can take care of this for us. See documentation. 
        But remember, we can initialize with other things!
        """

        hidden = torch.zeros(num_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden


class OurAwsomeLSTMWithTwoLayers(nn.Module):
    """
    Our simple LSTM model in pytorch. 
    This model has TWO layers, and a dropout of 0.5
    For prediction, it only uses the last hidden layer
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, device, num_layers=2):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embeddings
        @param hidden_dim: size of our hidden layers
        @param output_dim: size of the expected output
        @param device: device to be used by pytorch
        @param num_layers: Number of layers
        """
        super(OurAwsomeLSTMWithTwoLayers, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.device = device
        
        # See the full documentation at: https://pytorch.org/docs/stable/nn.html
        self.embedding = nn.Embedding(self.vocab_size + 3, self.emb_dim, padding_idx=0) 
        self.LSTM = nn.LSTM(self.emb_dim, self.hidden_dim, 
                            batch_first=True, num_layers=self.num_layers, dropout=0.5)   
        self.linear = nn.Linear(self.hidden_dim, 1)     # Why map to 1?
        self.sigmoid = nn.Sigmoid()      
    
    def forward(self, x):
        """
        Our forwad function in pytorch.  
        """
        
        batch_size = x.size()[0]
        embeddings = self.embedding(x)                             # [BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE]
        h_0 = self.init_hidden(batch_size, self.num_layers)        # [NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]
        c_0 = self.init_hidden(batch_size, self.num_layers)        # [NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]
        output, touple_hn_cn = self.LSTM(embeddings, (h_0, c_0))   # h_n: [NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]; 
                                                                   # output: [BATCH_SIZE, MAX_TIMESTEPS, HIDDEN_SIZE]
        output = self.linear(output[:,-1,:])                       # [BATCH_SIZE, 1] (From ONLY the last hidden layer of last layer)
        return self.sigmoid(output.view(-1))                       # [BATCH_SIZE] 
    
    def init_hidden(self, batch_size, num_layers):
        """
        Initialize our first hidden layer with zeros.
        This might be unecessary, pytorch can take care of this for us. See documentation. 
        But remember, we can initialize with other things!
        """

        hidden = torch.zeros(num_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden
    
        
    def init_cell_state(self, batch_size, num_layers):
        """
        Initialize our first cell state with zeros.
        This might be unecessary, pytorch can take care of this for us. See documentation. 
        """

        cell_state = torch.zeros(num_layers, batch_size, self.hidden_dim).to(self.device)
        return cell_state


class OurAwsomeRNNWithAllConnections(nn.Module):
    """
    Our simple RNN model in pytorch.
    For this model, we will use ALL the hidden layers available in our RNN to propagate information.
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, device, max_len):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embeddings
        @param hidden_dim: size of our hidden layers
        @param output_dim: size of the expected output
        @param device: device to be used by pytorch
        """
        super(OurAwsomeRNNWithAllConnections, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.device = device
        
        # See the full documentation at: https://pytorch.org/docs/stable/nn.html
        self.embedding = nn.Embedding(self.vocab_size + 3, self.emb_dim, padding_idx=0)
        self.rnn = nn.RNN(self.emb_dim, self.hidden_dim, num_layers=1, batch_first=True)  
        self.linear1 = nn.Linear(self.hidden_dim, 1) 
        self.linear2 = nn.Linear(self.max_len, 1) 
        self.sigmoid = nn.Sigmoid()     

    def forward(self, x):
        """
        Our forwad function in pytorch.  
        """
        
        batch_size = x.size()[0]
        embeddings = self.embedding(x)            # [BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE]
        h_0 = self.init_hidden(batch_size)        # [NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]
        output, _ = self.rnn(embeddings, h_0)     # output: [BATCH_SIZE, MAX_LEN, HIDDEN_SIZE] 
        output = self.linear1(output)             # output: [BATCH_SIZE, MAX_LEN, 1] 
        output = self.linear2(output.view(batch_size,self.max_len)) # [BATCH_SIZE, 1] from [BATCH_SIZE, MAX_LEN]
        return self.sigmoid(output.view(-1))      # output: [BATCH_SIZE] 
    
    def init_hidden(self, batch_size, num_layers=1):
        """
        Initialize our first hidden layer with zeros.
        This might be unecessary, pytorch can take care of this for us. See documentation. 
        But remember, we can initialize with other things!
        """

        hidden = torch.zeros(num_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden
    
