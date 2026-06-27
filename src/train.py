from dataset import TextToMotionDataset
from model import MotionDenoiser
from diffusion import GaussianDiffusion
from text_encoder import CLIPTextEncoder
from torch.utils.data import DataLoader
import torch
import argparse


def parse_args(): 
    p = argparse.ArgumentParser()
    p.add_argument("--joints_dir", type=str, default="data/KIT-ML/new_joints")
    p.add_argument("--split_file", type=str, default="data/KIT-ML/train.txt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--max_len", type=int, default=196)
    p.add_argument("--min_len", type=int, default=24)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="motion_denoiser.pt")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--lambda_vel", type=float, default=1.0)   # velocity loss agirligi (statik-poz'a karsi)
    p.add_argument("--cfg_prob", type=float, default=0.1)     # CFG: text'i null'a dusurme olasiligi
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print("device:", device)

    # dataset
    ds = TextToMotionDataset(args.joints_dir, args.split_file,
                            max_len=args.max_len, min_len=args.min_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"There is {len(ds)} trainable motions")


    args.feature_dim = ds.motions[0].shape[-1]
    print(f"feature dimension: {args.feature_dim}")

    # text encoder (donmus CLIP) — caption -> (B,512)
    text_encoder = CLIPTextEncoder(device=device)
    print(f"clip text dim: {text_encoder.dim}")

    # model
    model = MotionDenoiser(feature_dim=args.feature_dim, d_model=args.d_model, nhead=args.nhead,
                            num_layers=args.num_layers, max_len=args.max_len,
                            text_dim=text_encoder.dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f}M")

    #  diffusion (move to GPU)
    diffusion = GaussianDiffusion(T=args.T)
    for attr in ["betas", "alphas", "cum_alphas", "sqrt_cumprod_alpha", "sqrt_one_minus"]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    # training
    model.train()
    for epoch in range(args.epochs):
        running, n_batches = 0.0, 0
        for motion, mask, caption in dl:
            motion = motion.to(device)              # (B,T,feature_dim) = temiz x0
            mask   = mask.to(device)                # (B,T)
            B, T, D = motion.shape

            # caption -> CLIP ozelligi (gradyan yok); CFG dropout maskesi
            text_emb = text_encoder.encode(list(caption))           # (B,512)
            drop = torch.rand(B, device=device) < args.cfg_prob     # (B,) True=null kullan

            t   = torch.randint(0, args.T, (B,), device=device)
            eps = torch.randn_like(motion)
            x_t = diffusion.q_sample(motion, t, eps)       # forward
            pred = model(x_t, t, mask, text_emb, drop)     # x0 tahmini (text-kosullu)

            # 1) x0 loss (maskeli): x0 tahminini gercek x0 (=motion) ile karsilastir
            m = mask.unsqueeze(-1)                   # (B,T,1)
            loss_x0 = ((pred - motion) ** 2 * m).sum() / (m.sum() * D)

            # 2) velocity loss: kare-arasi farki denetle -> statik-poz'u cezalandir (artikulasyon)
            vel_pred = pred[:, 1:] - pred[:, :-1]                 # (B,T-1,D)
            vel_gt   = motion[:, 1:] - motion[:, :-1]
            mvel = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)     # ardisik iki kare de gercek
            loss_vel = ((vel_pred - vel_gt) ** 2 * mvel).sum() / (mvel.sum() * D)

            loss = loss_x0 + args.lambda_vel * loss_vel

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1
        epoch_loss = running / n_batches
        losses.append(epoch_loss)
        if epoch % args.log_every == 0:
            print(f"epoch {epoch:3d} | loss {epoch_loss:.4f}")
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
              torch.save({"model": model.state_dict(),
                          "mean": ds.mean, "std": ds.std,
                          "losses": losses,
                          "args": vars(args)}, args.save_path)

    # save (mean/std dahil -> sampling'de denormalize icin sart)
    """torch.save({"model": model.state_dict(),
                "mean": ds.mean, "std": ds.std,
                "args": vars(args)}, args.save_path)
    print("saved ->", args.save_path)"""


if __name__ == "__main__":
    main()