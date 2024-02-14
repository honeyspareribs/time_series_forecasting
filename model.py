import torch
import torch.nn as nn
import math

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim=1, output_dim=1, num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, inputs, h0=None, c0=None):
        batch_size = inputs.size(0)
        if h0 == None and c0 == None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(torch.device("cuda:0"))
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(torch.device("cuda:0"))
        lstm_out, (h_n, c_n)  = self.lstm(inputs.view(-1, batch_size, 1), (h0, c0))
        y_pred = self.fc(lstm_out)
        y_pred = y_pred.view(batch_size, -1)
        return y_pred, (h_n, c_n)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GPTDecoder(nn.TransformerDecoder):
    def forward(self, tgt, mask=None):
        # 将 memory 设置为 None
        memory = torch.zeros_like(tgt).to(torch.device("cuda:0"))
        return super().forward(tgt, memory, tgt_mask=mask)

class TransformerModel(nn.Module):
    def __init__(self, hidden_size, nhead, input_dim = 1, output_dim = 1, num_layers = 1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.input_fc = nn.Linear(input_dim, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead=nhead)
        self.transformer_decoder = GPTDecoder(self.transformer_decoder_layer, num_layers)

        self.output_fc = nn.Linear(hidden_size, output_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(torch.device("cuda:0"))
        return mask

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, batch_size, self.input_dim)
        x = self.input_fc(x)
        x = self.positional_encoding(x)
        
        # 这里不需要memory参数, 只需要输入x和mask
        x = self.transformer_decoder(x, mask=self.generate_square_subsequent_mask(x.size(0)))
        x = self.output_fc(x)
        x = x.view(batch_size, -1)
        return x, None
