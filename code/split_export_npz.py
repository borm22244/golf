# tools/split_export_npz.py
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

SRC = "data/train_with_channels.npz"   # 原始含 chan_names 的檔
OUTDIR = Path("data/splits_su")        # 輸出資料夾
OUTDIR.mkdir(parents=True, exist_ok=True)

d = np.load(SRC, allow_pickle=True, mmap_mode="r")

# 只保留 S/U
keep = np.isin(d["resp_chars"], ["S","U"])
idx_all = np.where(keep)[0]
y       = (d["resp_chars"][keep] == "S").astype(np.int64)
groups  = d["subject_ids"][keep].astype(int)

# 70/15/15 by subject
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
trval, te = next(gss1.split(y, y, groups=groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)  # ≈ 15% of all
tr, va = next(gss2.split(y[trval], y[trval], groups=groups[trval]))

tr_idx = idx_all[trval[tr]]
va_idx = idx_all[trval[va]]
te_idx = idx_all[te]

np.savez_compressed(OUTDIR/"indices.npz", train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx)

# 會被「按樣本切分」的鍵（第 0 維 = N）
PER_TRIAL_KEYS = [
    "X","subject_ids","set_ids","resp_labels","resp_chars","task_ids","cond_ids","srate"
]
# 其餘保留原樣
STATIC_KEYS = ["subjects","sets","tasks","conds","C","T","chan_names","chan_xyz"]

def export_subset(name, idx):
    payload = {}
    for k in PER_TRIAL_KEYS:
        if k in d.files: payload[k] = d[k][idx]
    for k in STATIC_KEYS:
        if k in d.files: payload[k] = d[k]
    payload["n_trials"] = np.int32(len(idx))
    np.savez_compressed(OUTDIR/f"{name}.npz", **payload)
    print(f"Saved {name}.npz  N={len(idx)} S:U = {int((payload['resp_chars']=='S').sum())}:{int((payload['resp_chars']=='U').sum())}")

export_subset("train", tr_idx)
export_subset("val",   va_idx)
export_subset("test",  te_idx)
print("Done. Files at:", OUTDIR)
