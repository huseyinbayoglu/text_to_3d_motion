"""
Sunucu/uygulama icin yeniden kullanilabilir motion uretici.
Modeli BIR KEZ yukler; her cagrida bellekte three.js-uyumlu JSON (dict) dondurur.
generate.py'daki recover_from_ric + chain_for'u tekrar kullanir.
"""
import numpy as np
import torch

from model import MotionDenoiser
from diffusion import GaussianDiffusion
from text_encoder import CLIPTextEncoder
from generate import recover_from_ric, chain_for


class MotionGenerator:
    def __init__(self, ckpt, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = ck["args"]
        self.margs = m
        self.mean = ck["mean"]
        self.std  = ck["std"]
        self.feature_dim = m.get("feature_dim", 263)

        self.text_encoder = CLIPTextEncoder(device=self.device)
        self.model = MotionDenoiser(
            feature_dim=self.feature_dim, d_model=m["d_model"], nhead=m["nhead"],
            num_layers=m["num_layers"], max_len=m["max_len"], text_dim=self.text_encoder.dim)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.diffusion = GaussianDiffusion(T=m["T"])
        # diffusion buffer'larini cihaza tasi
        for a in ["betas", "alphas", "cum_alphas", "sqrt_cumprod_alpha", "sqrt_one_minus"]:
            setattr(self.diffusion, a, getattr(self.diffusion, a).to(self.device))

    @torch.no_grad()
    def generate(self, prompt, seq_len=120, guidance=2.5, joints_num=22, fps=20, ddim_steps=50):
        """prompt -> three.js JSON dict (tek motion).
        ddim_steps>0 -> hizli DDIM (~1 sn); 0 -> tam DDPM (1000 adim, yavas ama referans)."""
        text_emb = None
        if prompt:
            text_emb = self.text_encoder.encode([prompt])          # (1,512)
        if ddim_steps and ddim_steps > 0:
            samples = self.diffusion.ddim_sample(self.model, n=1, seq_len=seq_len,
                                                 feature_dim=self.feature_dim,
                                                 text_emb=text_emb, guidance=guidance,
                                                 steps=int(ddim_steps))
        else:
            samples = self.diffusion.sample(self.model, n=1, seq_len=seq_len,
                                            feature_dim=self.feature_dim,
                                            text_emb=text_emb, guidance=guidance)
        denorm = samples.cpu().numpy() * self.std + self.mean       # (1,T,D)
        if self.feature_dim == 63:                                  # Path A: ham pozisyon
            positions = denorm.reshape(1, seq_len, joints_num, 3)
        else:                                                       # Path B: 263/251 -> recover
            positions = recover_from_ric(torch.from_numpy(denorm).float(), joints_num).numpy()
        return {
            "prompt": prompt,
            "fps": fps,
            "joints_num": joints_num,
            "chain": chain_for(joints_num),
            "frames": np.round(positions[0], 4).tolist(),          # (T, J, 3)
        }
