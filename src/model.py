import torch 
import torch.nn as nn 

class MotionDenoiser(nn.Module):
    def __init__(self,feature_dim=63, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_len=196):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim,d_model)
        self.output_proj = nn.Linear(d_model,feature_dim)

        # transformer encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)

        self.d_model = d_model
        self.max_len = max_len

        # Frame positional encoding
        positions = torch.arange(max_len).unsqueeze(1)
        inv_freq  = 10000 ** (-torch.arange(0, d_model, 2).float() / d_model)
        angles    = positions * inv_freq
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)                         
        pe[:, 1::2] = torch.cos(angles)                           
        self.register_buffer("pos_enc", pe) 

        # Diffusion time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.SiLU(),
            nn.Linear(d_model,d_model)
        )
    
    def timestep_embedding(self,t):
        inv_freq = 10000 ** (
            -torch.arange(0, self.d_model, 2, device=t.device).float() / self.d_model
        )
        args = t[:, None].float() * inv_freq[None, :]

        emb = torch.cat(
            [torch.sin(args), torch.cos(args)],
            dim=1
        ) 
        return emb

    def forward(self,x_t,t,mask):
        h = self.input_proj(x_t) 
        # adding frame positional encoding
        h = h + self.pos_enc
        # adding diffusien time encoding
        t_emb = self.time_mlp(self.timestep_embedding(t))
        h = h + t_emb.unsqueeze(1)
        # Transformer encoder
        h = self.encoder(h,src_key_padding_mask=~mask)

        out = self.output_proj(h)
        return out 
    
B = 4
x = torch.randn(B, 196, 63)
t = torch.randint(0, 200, (B,))
mask = torch.ones(B, 196, dtype=torch.bool)
out = MotionDenoiser()(x, t, mask)
print(out.shape)   # (4, 196, 63) bekliyoruz