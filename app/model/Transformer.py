import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L].to(x.device)


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=1, latent_dim=8):
        super().__init__()
        self.enc_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model, max_len=1000)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(d_model, latent_dim)

        self.from_latent = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.dec_out = nn.Linear(d_model, input_dim)

    def encode(self, x):
        x = self.enc_in(x) * math.sqrt(self.enc_in.out_features)
        x = self.pos(x)
        x = x.permute(1, 0, 2)
        h = self.encoder(x)
        h = h.permute(1, 2, 0)
        pooled = self.pool(h).squeeze(-1)
        z = self.to_latent(pooled)
        return z

    def decode(self, z, seq_len):
        h = self.from_latent(z)
        h_seq = h.unsqueeze(1).repeat(1, seq_len, 1)
        h_seq = self.pos(h_seq)
        h_seq = h_seq.permute(1, 0, 2)
        out = self.decoder(h_seq)
        out = out.permute(1, 0, 2)
        x_rec = self.dec_out(out)
        return x_rec

    def forward(self, x):
        z = self.encode(x)
        rec = self.decode(z, x.size(1))
        return rec, z