import torch 


class GaussianDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T=T
        betas = torch.linspace(beta_start,beta_end,self.T)
        alphas = 1 - betas 
        cum_alphas = torch.cumprod(alphas, dim=0)

        self.betas, self.alphas, self.cum_alphas = betas, alphas, cum_alphas
        self.sqrt_cumprod_alpha = torch.sqrt(cum_alphas)
        self.sqrt_one_minus     = torch.sqrt(1 - cum_alphas)

    def q_sample(self,x_0,t,eps=None):
        if eps is None:
            eps = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_cumprod_alpha[t].view(-1, 1,1)   # (B,) -> (B,1,1)                                         
        sqrt_om = self.sqrt_one_minus[t].view(-1, 1,1)  
        return sqrt_ab * x_0 + sqrt_om * eps
    
    @torch.no_grad()
    def sample(self, model, n, seq_len=196, feature_dim=63, text_emb=None, guidance=2.5):
        model.eval()
        device = next(model.parameters()).device          # cihazi modelden al
        x = torch.randn(n, seq_len, feature_dim, device=device)
        mask = torch.ones(n, seq_len, dtype=torch.bool, device=device) # all the motion
        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, dtype=torch.long, device=device)

            # --- classifier-free guidance ---
            if text_emb is None:                        # kosulsuz uretim
                x0_hat = model(x, t, mask)
            else:                                       # cond + uncond -> blend
                x0_cond   = model(x, t, mask, text_emb)        # text'li
                x0_uncond = model(x, t, mask, None)            # null (text yok)
                x0_hat = x0_uncond + guidance * (x0_cond - x0_uncond)

            alpha_t        = self.alphas[ti].to(device)
            beta_t         = self.betas[ti].to(device)
            alpha_bar_t    = self.cum_alphas[ti].to(device)
            alpha_bar_prev = self.cum_alphas[ti - 1].to(device) if ti > 0 else torch.tensor(1.0, device=device)

            # posterior q(x_{t-1} | x_t, x0_hat): mean = c0*x0_hat + ct*x_t  (stabil, bolme patlamasi yok)
            c0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
            ct = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            mean = c0 * x0_hat + ct * x
            if ti > 0:
                var = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t   # posterior varyans (beta_tilde)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean                                # son adim: x = x0_hat
        return x

    @torch.no_grad()
    def ddim_sample(self, model, n, seq_len=196, feature_dim=63,
                    text_emb=None, guidance=2.5, steps=50, eta=0.0):
        """Hizli sampling: T=1000 yerine 'steps' adim atlar. x0-prediction -> eps turetilir.
        eta=0 -> deterministik DDIM. ~50 adim 1000-adim DDPM'e cok yakin, ~20x hizli."""
        model.eval()
        device = next(model.parameters()).device
        x = torch.randn(n, seq_len, feature_dim, device=device)
        mask = torch.ones(n, seq_len, dtype=torch.bool, device=device)

        # zaman alt-dizisi: T-1 ... 0 arasi 'steps' nokta (azalan)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long().tolist()
        for i, ti in enumerate(ts):
            t = torch.full((n,), ti, dtype=torch.long, device=device)

            # --- classifier-free guidance (sample ile ayni) ---
            if text_emb is None:
                x0_hat = model(x, t, mask)
            else:
                x0_cond   = model(x, t, mask, text_emb)
                x0_uncond = model(x, t, mask, None)
                x0_hat = x0_uncond + guidance * (x0_cond - x0_uncond)

            abar_t = self.cum_alphas[ti].to(device)
            # eps'i tahmin edilen x0'dan turet
            eps = (x - torch.sqrt(abar_t) * x0_hat) / torch.sqrt(1 - abar_t)

            if i < len(ts) - 1:
                abar_prev = self.cum_alphas[ts[i + 1]].to(device)
            else:
                abar_prev = torch.tensor(1.0, device=device)             # son adim -> temiz x0

            # DDIM guncelleme (eta=0 deterministik)
            sigma = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t)) * torch.sqrt(1 - abar_t / abar_prev)
            dir_xt = torch.sqrt(torch.clamp(1 - abar_prev - sigma ** 2, min=0.0)) * eps
            x = torch.sqrt(abar_prev) * x0_hat + dir_xt
            if eta > 0 and i < len(ts) - 1:
                x = x + sigma * torch.randn_like(x)
        return x