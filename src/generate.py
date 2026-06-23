"""
Egitilmis modelden motion uret + iskelet animasyonu (gif) kaydet.
Ornek: python src/generate.py --ckpt motion_denoiser.pt --n 4 --out_dir samples
Uretim CPU'da kosar (birkac ornek, T=1000 -> birkac dakika). Yeterli.
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                      # headless: ekran yok, dosyaya kaydet
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401  (3d projection icin gerekli)

from model import MotionDenoiser
from diffusion import GaussianDiffusion

# KIT iskelet baglanti semasi (HumanML3D repo: paramUtil.kit_kinematic_chain)
KIT_CHAIN = [[0, 11, 12, 13, 14, 15],
             [0, 16, 17, 18, 19, 20],
             [0, 1, 2, 3, 4],
             [3, 5, 6, 7],
             [3, 8, 9, 10]]


def render_motion(motion, out_path, fps=20):
    """motion: (T, 21, 3) numpy -> iskelet animasyonu gif olarak kaydet."""
    F = motion.shape[0]
    # eksen sinirlarini tum kareler icin sabitle (yoksa kamera titrer)
    x, depth, up = motion[..., 0], motion[..., 2], motion[..., 1]
    xlim = (x.min(), x.max()); ylim = (depth.min(), depth.max()); zlim = (up.min(), up.max())

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(f):
        ax.cla()
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_title(f"frame {f}/{F}")
        fr = motion[f]                                  # (21, 3)
        ax.scatter(fr[:, 0], fr[:, 2], fr[:, 1], s=20)  # (x, derinlik, yukari)
        for chain in KIT_CHAIN:
            ax.plot(fr[chain, 0], fr[chain, 2], fr[chain, 1], linewidth=2)

    ani = animation.FuncAnimation(fig, draw, frames=F, interval=1000 / fps)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="motion_denoiser.pt")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="samples")
    p.add_argument("--seq_len", type=int, default=196)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) checkpoint yukle (CPU'ya)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # ckpt'te numpy mean/std var
    margs = ckpt["args"]            # egitimdeki hiperparametreler
    mean  = ckpt["mean"]           # (63,)
    std   = ckpt["std"]            # (63,)

    # 2) modeli AYNI mimariyle kur + agirliklari yukle
    model = MotionDenoiser(feature_dim=63,
                           d_model=margs["d_model"],
                           nhead=margs["nhead"],
                           num_layers=margs["num_layers"],
                           max_len=margs["max_len"])
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()                  # sample() cihazi modelden alir

    # 3) diffusion (egitimdeki T ile)
    diffusion = GaussianDiffusion(T=margs["T"])

    # 4) uret -> (n, seq_len, 63) normalize edilmis
    samples = diffusion.sample(model, n=args.n, seq_len=args.seq_len, feature_dim=63)

    # 5) DENORMALIZE (sart!) -> gercek pozisyon olcegi
    samples = samples.cpu().numpy() * std + mean          # (n, seq_len, 63)

    # 6) (n, seq_len, 21, 3) seklinde ac
    samples = samples.reshape(args.n, args.seq_len, 21, 3)

    # 7) her ornegi render et
    for i in range(args.n):
        out_path = os.path.join(args.out_dir, f"sample_{i}.gif")
        render_motion(samples[i], out_path)
        print("saved ->", out_path)


if __name__ == "__main__":
    main()
