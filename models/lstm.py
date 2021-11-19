import torch
import torch.nn as nn

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

        
    def forward(self, image_features, image_captions):
        # image_features = torch.Tensor.repeat_interleave(image_features, repeats=5 , dim=0)
        image_features = image_features.unsqueeze(1)
        
        if type(image_captions) == torch.nn.utils.rnn.PackedSequence:
            embedded_captions = torch.nn.utils.rnn.PackedSequence(self.embed(image_captions.data), image_captions.batch_sizes)
        else:
            embedded_captions = self.embed(image_captions)

        lstm_outputs, _ = self.lstm(embedded_captions)        
        lstm_outputs = self.linear(lstm_outputs.data) 
        
        return lstm_outputs