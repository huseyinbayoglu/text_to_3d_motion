"""
Donmus (frozen) CLIP text encoder — MDM ile ayni (ViT-B/32, 512-dim).
Egitilmez; sadece caption -> sabit ozellik vektoru cikarir.
Kurulum: pip install git+https://github.com/openai/CLIP.git ftfy regex
"""
import torch
import clip


class CLIPTextEncoder:
    def __init__(self, device, model_name="ViT-B/32"):
        self.device = device
        model, _ = clip.load(model_name, device=device, jit=False)
        model.eval()
        for p in model.parameters():          # tum CLIP'i dondur (gradyan yok)
            p.requires_grad_(False)
        self.model = model
        self.dim = model.text_projection.shape[1]   # ViT-B/32 -> 512

    @torch.no_grad()
    def encode(self, texts):
        """texts: List[str] -> (B, 512) float tensor (device'ta)."""
        tokens = clip.tokenize(texts, truncate=True).to(self.device)  # (B,77)
        feat = self.model.encode_text(tokens).float()                 # (B,512)
        return feat
