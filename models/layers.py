import math
import torch
import torch.nn as nn
from .utils.layer import BasicBlock

class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
            BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1600, downsample=True),
            BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
            BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
            BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
        )
        self.multi_scale = nn.ModuleList([
            nn.Sequential( 
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            ),
            nn.Sequential(
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),       
            ),
        ])
        self.out = nn.Linear(out_dim*3, out_dim)

    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        x = wav_data
        msi = 0
        ms = []
        for i in range(len(self.feat_extractor)):
            x = self.feat_extractor[i](x)
            if i == 2 or i == 4:
                ms.append(self.multi_scale[msi](x))
                msi += 1
        ms.append(x)
        out = self.out(torch.cat(ms, dim=1).transpose(-1, -2))
        return out
    
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, out_dim)
        )
    def forward(self, inputs):
        out = self.mlp(inputs)
        return out


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=60): 
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1) # (1, repeat_num, period, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # print(self.pe.shape, x.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)