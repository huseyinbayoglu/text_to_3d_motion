"""!git clone https://github.com/EricGuo5513/HumanML3D
!pip install gdown
!gdown --folder https://drive.google.com/drive/folders/1D3bf2G2o4Hv-Ale26YW18r1Wrh7oIAwK
"""
import numpy as np 
import os 
from torch.utils.data import Dataset
import torch 

joints_path = "data/KIT-ML/new_joints/"
joint_vecs_path = "data/KIT-ML/new_joint_vecs/"
texts_path = "data/KIT-ML/texts/"

class TextToMotionDataset(Dataset):
  def __init__(self,data_dir, split_file,max_len=196,min_len=24):
    self.max_len = max_len

    # 1) split id'lerini oku, bos satirlari at
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    # 2) motionlari yukle (dataset kucuk -> hepsini RAM'de tut), kisalari ele
    self.motions = []                      # her biri (L, 63), degisken uzunluk
    for mid in ids:
        path = os.path.join(data_dir, mid + ".npy")
        if not os.path.exists(path):
            continue
        m = np.load(path)                  # (L, 21, 3)
        m = m.reshape(m.shape[0], -1)      # (L, 63)  her kareyi duzlestir
        if m.shape[0] < min_len:           # cok kisa -> ele
            continue
        self.motions.append(m.astype(np.float32))

    # 3) normalizasyon: ozellik bazinda mean/std (sadece bu split'ten)
    stacked = np.concatenate(self.motions, axis=0)        # (toplam_kare, 63)
    self.mean = stacked.mean(axis=0).astype(np.float32)   # (63,)
    self.std  = (stacked.std(axis=0) + 1e-8).astype(np.float32)  # (63,), eps: sifira bolme

  def __len__(self):
    return len(self.motions)

  def __getitem__(self,idx):
    m = self.motions[idx]
    m = (m - self.mean) / self.std 
    L = m.shape[0] 
    if L > self.max_len:
      m = m[:self.max_len]
      L = self.max_len
    sequence = np.zeros((self.max_len, m.shape[1]), dtype=np.float32) # m.shape[1] = 63 here
    sequence[:L] = m  # padding 
    mask = np.zeros(self.max_len,dtype=bool)
    mask[:L] = True
    return torch.from_numpy(sequence), torch.from_numpy(mask)


