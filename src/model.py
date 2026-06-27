import torch 
import torch.nn as nn 

class MotionDenoiser(nn.Module):
    def __init__(self,feature_dim=63, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_len=196, text_dim=512):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim,d_model)
        self.output_proj = nn.Linear(d_model,feature_dim)

        # --- text kosullandirma (CLIP cikisini d_model'e tasi) ---
        self.text_proj = nn.Linear(text_dim, d_model)
        # CFG icin "bos text" temsili (ogrenilen vektor); text dusurulen orneklerde kullanilir
        self.null_cond = nn.Parameter(torch.zeros(d_model))

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

    def forward(self, x_t, t, mask, text_emb=None, drop=None):
        # x_t:(B,T,D)  t:(B,)  mask:(B,T) True=gercek  text_emb:(B,512) ya da None  drop:(B,) bool
        B = x_t.shape[0]

        h = self.input_proj(x_t)            # (B,T,d)
        h = h + self.pos_enc                # frame positional encoding (kosul token'ina EKLENMEZ)

        t_emb = self.time_mlp(self.timestep_embedding(t))   # (B,d)  diffusion-t

        # --- text kosulu (CFG dropout dahil) ---
        null = self.null_cond.unsqueeze(0).expand(B, -1)    # (B,d)  "bos text"
        if text_emb is None:                                # tamamen kosulsuz
            text_cond = null
        else:
            text_cond = self.text_proj(text_emb)            # (B,d)
            if drop is not None:                            # dropping: o orneklerde null kullan
                text_cond = torch.where(drop.unsqueeze(1), null, text_cond)

        # --- MDM z_tk: text + time tek bir kosul token'i, dizinin BASINA eklenir ---
        z_tk = (t_emb + text_cond).unsqueeze(1)             # (B,1,d)
        h = torch.cat([z_tk, h], dim=1)                     # (B,T+1,d)

        # padding mask'i da uzat: kosul token'i her zaman gercek (dikkate alinsin)
        cond_valid = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mask_ext = torch.cat([cond_valid, mask], dim=1)     # (B,T+1)

        h = self.encoder(h, src_key_padding_mask=~mask_ext) # (B,T+1,d)
        h = h[:, 1:]                                        # kosul token'inin cikisini AT -> (B,T,d)

        out = self.output_proj(h)                           # (B,T,D)
        return out
    
