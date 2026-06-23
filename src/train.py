from dataset import TextToMotionDataset
from model import MotionDenoiser
from diffusion import GaussianDiffusion
from generate import recover_from_ric          # geometrik loss icin (251 -> pozisyon)
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
    p.add_argument("--lambda_geo", type=float, default=1.0)   # geometrik loss agirligi (251 icin)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print("device:", device)

    # dataset
    ds = TextToMotionDataset(args.joints_dir, args.split_file,
                            max_len=args.max_len, min_len=args.min_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"{len(ds)} motions")

    # feature_dim'i veriden turet (Path A new_joints -> 63, Path B new_joint_vecs -> 251)
    # args'a yaziyoruz ki checkpoint'e gitsin ve generate.py okuyabilsin
    args.feature_dim = ds.motions[0].shape[-1]
    print(f"feature_dim: {args.feature_dim}")

    # normalizasyon istatistikleri (geometrik loss'ta denormalize icin) - torch, device'ta
    mean_t = torch.tensor(ds.mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(ds.std,  dtype=torch.float32, device=device)

    # model
    model = MotionDenoiser(feature_dim=args.feature_dim, d_model=args.d_model, nhead=args.nhead,
                            num_layers=args.num_layers, max_len=args.max_len).to(device)
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
        running, running_geo, n_batches = 0.0, 0.0, 0
        for motion, mask in dl:
            motion = motion.to(device)              # (B,T,feature_dim)
            mask   = mask.to(device)                # (B,T)
            B, T, D = motion.shape

            t   = torch.randint(0, args.T, (B,), device=device)
            eps = torch.randn_like(motion)
            x_t = diffusion.q_sample(motion, t, eps)
            pred = model(x_t, t, mask)

            # 1) eps loss (maskeli; padding'i katma)
            m = mask.unsqueeze(-1)                   # (B,T,1)
            loss_eps = ((pred - eps) ** 2 * m).sum() / (m.sum() * D)

            loss = loss_eps
            loss_geo_val = 0.0
            # 2) geometrik loss (sadece 251-dim Path B icin):
            #    eps tahmininden temiz x0'i geri hesapla -> pozisyona cevir -> gercek pozisyonla MSE
            #    Boylece recover'in kullandigi kritik ozellikler dogrudan denetlenir.
            if args.feature_dim == 251:
                sqrt_ab = diffusion.sqrt_cumprod_alpha[t].view(-1, 1, 1)
                sqrt_om = diffusion.sqrt_one_minus[t].view(-1, 1, 1)
                x0_hat = (x_t - sqrt_om * pred) / sqrt_ab            # tahmin edilen temiz x0 (normalize)
                pos_hat = recover_from_ric(x0_hat * std_t + mean_t, 21)     # (B,T,21,3)
                with torch.no_grad():
                    pos_gt = recover_from_ric(motion * std_t + mean_t, 21)
                mpos = m.unsqueeze(-1)                               # (B,T,1,1)
                # ornek-basina maskeli pozisyon MSE'si
                per_sample = ((pos_hat - pos_gt) ** 2 * mpos).sum(dim=(1, 2, 3)) \
                             / (mpos.sum(dim=(1, 2, 3)) * 21 * 3)    # (B,)
                # alpha_bar_t ile agirlikla: yuksek t'de x0_hat patlar -> agirlik ~0
                w = diffusion.cum_alphas[t]                          # (B,)
                loss_geo = (w * per_sample).sum() / w.sum()
                loss = loss_eps + args.lambda_geo * loss_geo
                loss_geo_val = loss_geo.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            running_geo += loss_geo_val
            n_batches += 1
        epoch_loss = running / n_batches
        losses.append(epoch_loss)
        if epoch % args.log_every == 0:
            print(f"epoch {epoch:3d} | loss {epoch_loss:.4f} | geo {running_geo / n_batches:.4f}")
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