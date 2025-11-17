import numpy as np
from pathlib import Path

path = "/Users/zhengliwei/Projects/golf/data/train_with_channels.npz"  # 換成你的檔案路徑
d = np.load(path, allow_pickle=True, mmap_mode="r")

print("=== Keys ===")
print(list(d.files))
print("X shape =", d["X"].shape)

# 安全檢查 chan_names / chan_xyz
if "chan_names" in d.files:
    print("chan_names length =", len(d["chan_names"]))
    print("first 10 chan_names =", d["chan_names"][:10])
else:
    print("chan_names: NOT PRESENT in this NPZ")

if "chan_xyz" in d.files:
    print("chan_xyz shape   =", d["chan_xyz"].shape)
else:
    print("chan_xyz: NOT PRESENT in this NPZ")

# 如果你做了旁檔 channels_meta.npz，也順便讀看看
meta_p = Path("data/channels_meta.npz")
if meta_p.exists():
    m = np.load(meta_p, allow_pickle=True)
    print("\n[channels_meta.npz] found")
    if "chan_names" in m.files:
        print("sidecar chan_names len =", len(m["chan_names"]))
        print("sidecar first 10 names  =", m["chan_names"][:10])
    if "chan_xyz" in m.files:
        print("sidecar chan_xyz shape  =", m["chan_xyz"].shape)
else:
    print("\n[channels_meta.npz] not found (optional sidecar)")
