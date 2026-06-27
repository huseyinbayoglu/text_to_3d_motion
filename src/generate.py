"""
Egitilmis modelden motion uret + iskelet animasyonu (gif) kaydet.
Ornek: python src/generate.py --ckpt motion_denoiser_251.pt --n 4 --out_dir samples
Path A (63 = ham pozisyon) ve Path B (251 = vektor temsil) ile calisir.
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
from text_encoder import CLIPTextEncoder

# KIT iskelet baglanti semasi (HumanML3D repo: paramUtil.kit_kinematic_chain)
KIT_CHAIN = [[0, 11, 12, 13, 14, 15],
             [0, 16, 17, 18, 19, 20],
             [0, 1, 2, 3, 4],
             [3, 5, 6, 7],
             [3, 8, 9, 10]]


# ---------------- 251-dim -> (J,3) pozisyon cozumu (HumanML3D kanonik) ----------------
def qinv(q):
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask                                   # kuaterniyon tersi (birim icin konjuge)


def qrot(q, v):
    """q: (...,4) kuaterniyon, v: (...,3) vektor -> v'yi q ile dondur."""
    assert q.shape[-1] == 4 and v.shape[-1] == 3 and q.shape[:-1] == v.shape[:-1]
    shape = list(v.shape)
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).reshape(shape)


def recover_root_rot_pos(data):
    """data: (..., D) -> root rotasyon kuaterniyonu ve root pozisyonu (integrasyonla)."""
    rot_vel = data[..., 0]                             # root acisal hizi (Y ekseni)
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)        # acisal hizi entegre et -> aci

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)          # Y ekseni etrafinda donus

    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]       # zemin hizi (x,z)
    r_pos = qrot(qinv(r_rot_quat), r_pos)              # hizi root cercevesine dondur
    r_pos = torch.cumsum(r_pos, dim=-2)                # entegre et -> pozisyon
    r_pos[..., 1] = data[..., 3]                       # root yuksekligi (y) dogrudan
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    """data: (..., D) denormalize edilmis ozellikler -> (..., joints_num, 3) pozisyon."""
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]      # ric kismi (root haric lokal poz)
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # lokal pozisyonlari root rotasyonuyla dunyaya dondur
    positions = qrot(qinv(r_rot_quat)[..., None, :].expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]                   # root x ekle
    positions[..., 2] += r_pos[..., 2:3]                   # root z ekle
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)  # root'u basa ekle
    return positions


def render_motion(motion, out_path, fps=20):
    """motion: (T, 21, 3) numpy -> iskelet animasyonu gif olarak kaydet."""
    F = motion.shape[0]
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
    p.add_argument("--joints_num", type=int, default=21)   # KIT: 21
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt", type=str, default=None)     # None -> kosulsuz uretim
    p.add_argument("--guidance", type=float, default=2.5)  # CFG gucu (w)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) checkpoint yukle
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # ckpt'te numpy mean/std var
    margs = ckpt["args"]
    mean  = ckpt["mean"]
    std   = ckpt["std"]
    feature_dim = margs.get("feature_dim", 63)         # eski Path A ckpt'lerde yok -> 63

    # 2) text encoder + prompt'u encode et (verilmisse)
    text_encoder = CLIPTextEncoder(device=args.device)
    if args.prompt is not None:
        text_emb = text_encoder.encode([args.prompt])          # (1,512)
        text_emb = text_emb.expand(args.n, -1)                 # (n,512) ayni prompt, farkli gurultu
        print(f"prompt: {args.prompt!r} | guidance: {args.guidance}")
    else:
        text_emb = None                                        # kosulsuz

    # 3) modeli AYNI mimariyle kur + agirliklari yukle
    model = MotionDenoiser(feature_dim=feature_dim,
                           d_model=margs["d_model"],
                           nhead=margs["nhead"],
                           num_layers=margs["num_layers"],
                           max_len=margs["max_len"],
                           text_dim=text_encoder.dim)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    # 4) diffusion + uret -> (n, seq_len, feature_dim) normalize edilmis
    diffusion = GaussianDiffusion(T=margs["T"])
    samples = diffusion.sample(model, n=args.n, seq_len=args.seq_len, feature_dim=feature_dim,
                               text_emb=text_emb, guidance=args.guidance)

    # 4) DENORMALIZE (sart!)
    denorm = samples.cpu().numpy() * std + mean        # (n, seq_len, feature_dim)

    # 5) pozisyona cevir
    if feature_dim == 63:                              # Path A: zaten ham pozisyon
        positions = denorm.reshape(args.n, args.seq_len, args.joints_num, 3)
    else:                                              # Path B: 251 -> recover_from_ric
        positions = recover_from_ric(torch.from_numpy(denorm).float(), args.joints_num).numpy()

    # 6) render
    for i in range(args.n):
        out_path = os.path.join(args.out_dir, f"sample_{i}.gif")
        render_motion(positions[i], out_path)
        print("saved ->", out_path)


if __name__ == "__main__":
    main()
