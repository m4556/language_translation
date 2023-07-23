import torch
import random
import torch.nn as nn
from preprocessing import german, english, train_data, valid_data, test_data


class Encoder(nn.Module):
    
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        
        # x shape: (seq_length, N) where N is batch size
        embedding = self.dropout(self.embedding(x))   # shape: (seq_length, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding)  # shape: (seq_length, N, hidden_size)
        
        return hidden, cell

class Decoder(nn.Module):
    
    def __init__(
            
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden, cell):

        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))  # shape: (1, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))  # shape: (1, N, hidden_size)
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs