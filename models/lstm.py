import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, units):
        super(AttentionBlock, self).__init__()
        self.W1 = nn.Linear(in_features = embed_dim, out_features = units)
        self.W2 = nn.Linear(in_features=units, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, img_features, hidden):
        
        hidden = hidden.unsqueeze(dim=1)
        # hidden = hidden.double()
        #print("feature and hidden shape",img_features.shape, hidden.shape)
        pre_score1 = self.W1(img_features)
        pre_score2 = self.W2(hidden)
        combined_score = self.tanh(pre_score1 + pre_score2)
        
        attention_weights = self.softmax(self.V(combined_score))
        context_vector = attention_weights * img_features
        context_vector = torch.sum(context_vector, dim=1)
        
        return context_vector, attention_weights

class LSTM(nn.Module):
    def __init__(self, embed_dim, lstm_hidden_size, lstm_layers=1, vocab_size=100, use_gpu=True, use_attention=True):
        super(LSTM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.vocab_size = vocab_size
        self.use_gpu = use_gpu
        self.use_attention = use_attention
        
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = lstm_hidden_size,
                            num_layers = lstm_layers, batch_first = True)
        
        self.linear = nn.Linear(lstm_hidden_size, self.vocab_size)        
        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        if self.use_attention:
            self.attention = AttentionBlock(embed_dim, lstm_hidden_size)

        
    def forward(self, image_features, image_captions):
        image_features = torch.Tensor.repeat_interleave(image_features, repeats=5 , dim=0)
        image_features = image_features.unsqueeze(1)
        
        hidden = torch.zeros((image_features.shape[0], self.lstm_hidden_size))
        if self.use_gpu:
            hidden = hidden.cuda()
        
        if self.use_attention:
            image_features, _ = self.attention(image_features, hidden)
        
        if type(image_captions) == torch.nn.utils.rnn.PackedSequence:
            embedded_captions = torch.nn.utils.rnn.PackedSequence(self.embed(image_captions.data), image_captions.batch_sizes)
        else:
            embedded_captions = self.embed(image_captions
            
        lstm_outputs, _ = self.lstm(embedded_captions)        
        lstm_outputs = self.linear(lstm_outputs.data) 
        
        return lstm_outputs