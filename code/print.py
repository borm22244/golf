import numpy as np, json
from collections import Counter

path = "./all_trials_merged.npz"  # ← 換成你的路徑
d = np.load(path, allow_pickle=True)

print("=== Keys ===")
keys = list(d.keys())
print(keys)

def sh(x): 
    try: return list(x.shape)
    except: return None

def dt(x): 
    try: return str(x.dtype)
    except: return type(x).__name__

report = {}
for k in keys:
    a = d[k]
    report[k] = {"shape": sh(a), "dtype": dt(a)}

print("\n=== Basic Schema ===")
print(json.dumps(report, indent=2))

X = d.get("X")
y = d.get("y")
sp = d.get("split")

if X is not None:
    print(f"\n[X] shape={X.shape}, dtype={X.dtype}")
    print(f"[X] NaN? {np.isnan(X).any() if np.issubdtype(X.dtype, np.floating) else 'N/A'} "
          f"Inf? {np.isinf(X).any() if np.issubdtype(X.dtype, np.floating) else 'N/A'}")
    if X.ndim == 2:
        mean = np.nanmean(X, axis=0)
        std  = np.nanstd(X, axis=0)
        print("[X] feature mean/std (first 8):")
        for i in range(min(8, X.shape[1])):
            print(f"  f{i}: mean={float(mean[i]):.6g}, std={float(std[i]):.6g}")

if y is not None:
    print(f"\n[y] shape={y.shape}, dtype={y.dtype}")
    if np.issubdtype(y.dtype, np.integer):
        cnt = Counter(y.tolist())
        print("[y] class distribution (top 30):")
        for c,n in cnt.most_common(30):
            print(f"  {c}: {n}")
        if len(cnt) > 30: print("  ...")
    elif np.issubdtype(y.dtype, np.floating):
        print(f"[y] min={float(np.nanmin(y)):.6g}, max={float(np.nanmax(y)):.6g}, "
              f"mean={float(np.nanmean(y)):.6g}, std={float(np.nanstd(y)):.6g}")

print("\n=== split ===")
try:
    sp_dict = sp.item() if hasattr(sp, "item") else sp
except Exception:
    sp_dict = sp
if isinstance(sp_dict, dict):
    for k,v in sp_dict.items():
        try: n = len(v)
        except: n = int(v) if isinstance(v,(int,np.integer)) else None
        print(f"  {k}: {n}")
else:
    print("  not found or unexpected format:", type(sp_dict))
