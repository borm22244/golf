#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat

NEEDED = [
    "setname", "nbchan", "pnts", "trials", "srate", "times",
    "data", "chanlocs", "event", "epoch", "datfile"
]

def fprint(*a, **kw):
    kw.setdefault("flush", True)
    print(*a, **kw)

def matobj_to_dict(obj):
    if isinstance(obj, np.void):  # MATLAB struct -> dict
        out = {}
        for name in obj.dtype.names:
            out[name] = matobj_to_dict(obj[name])
        return out
    elif isinstance(obj, np.ndarray) and obj.dtype == np.object_:
        return np.vectorize(matobj_to_dict, otypes=[object])(obj)
    else:
        return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_path", required=True)
    ap.add_argument("--head", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.set_path)
    if not p.exists():
        raise FileNotFoundError(p)

    fprint(f"=== 讀檔：{p} ===")
    # 只讀必要變數，避開會出事的 char/cell
    M = loadmat(
        p,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=False,
        appendmat=False,
        variable_names=NEEDED,
    )

    # 有些變數可能不存在，先列出實際讀到的 keys
    keys = [k for k in M.keys() if not k.startswith("__")]
    fprint("[實際讀到的變數 keys] ", keys)

    # 擷取欄位（若不存在就給 None）
    get = lambda k: M.get(k, None)

    setname = get("setname")
    nbchan  = int(get("nbchan") or 0)
    pnts    = int(get("pnts") or 0)
    trials  = int((get("trials") or 1) or 1)
    srate   = float(get("srate") or 0.0)
    times   = get("times")
    data    = get("data")
    chanlocs= get("chanlocs")
    event   = get("event")
    epoch   = get("epoch")
    datfile = get("datfile")

    fprint(f"[基本資訊] setname={setname}, nbchan={nbchan}, pnts={pnts}, trials={trials}, srate={srate}")

    # 通道名稱
    ch_names = []
    if chanlocs is not None:
        try:
            # chanlocs 常見是 (1, nbchan) 的 struct 陣列
            arr = np.atleast_1d(chanlocs)
            for c in arr.flat:
                if isinstance(c, np.void):
                    d = matobj_to_dict(c)
                    nm = d.get("labels") or d.get("label") or ""
                elif isinstance(c, dict):
                    nm = c.get("labels") or c.get("label") or ""
                else:
                    nm = str(c)
                ch_names.append(str(nm))
        except Exception:
            pass
    if ch_names:
        fprint("ch_names(前10)：", ch_names[:10], "..." if len(ch_names) > 10 else "")
    else:
        fprint("ch_names：無法解析或不存在（略過）")

    # 時間軸
    if isinstance(times, np.ndarray):
        try:
            fprint(f"times shape={times.shape}, 範圍≈[{times.min():.3f}, {times.max():.3f}] ms")
        except Exception:
            fprint(f"times shape={times.shape}")

    # 資料本體：shape 通常是 (nbchan, pnts, trials)
    if isinstance(data, np.ndarray):
        fprint(f"\n[data] dtype={data.dtype}, shape={data.shape} (nbchan x pnts x trials)")
        head = min(args.head, data.shape[1])
        fprint(f"第一個 trial 的每通道前 {head} 個 sample（單位多半是 µV）")
        preview = np.array(data[:, :head, 0], copy=True)
        for i in range(min(5, preview.shape[0])):
            nm = ch_names[i] if i < len(ch_names) and ch_names[i] else f"ch{i}"
            fprint(f"{nm:>10s}: {np.array2string(preview[i, :], precision=2, separator=', ')}")
        if data.shape[0] > 5:
            fprint("... (其餘通道省略)")
    else:
        fprint("\n[data] 不存在或不是 ndarray（可能使用外部 .fdt 或檔案異常）")

    # 事件 / epoch
    if event is not None:
        try:
            ev = np.atleast_1d(event)
            fprint(f"\n[event] 事件數：{ev.size}（列前 5）")
            for i, e in enumerate(ev.flat[:5]):
                d = matobj_to_dict(e) if isinstance(e, np.void) else (e if isinstance(e, dict) else None)
                if isinstance(d, dict):
                    fprint(f"{i}: type={d.get('type')}, latency={d.get('latency')}, duration={d.get('duration')}")
                else:
                    fprint(f"{i}: {e}")
        except Exception as ex:
            fprint(f"\n[event] 存在但解析失敗：{ex}")
    else:
        fprint("\n[event] 無")

    if epoch is not None:
        try:
            ep = np.atleast_1d(epoch)
            fprint(f"[epoch] 數量：{ep.size}")
        except Exception:
            fprint("[epoch] 存在但解析失敗")
    else:
        fprint("[epoch] 無")

    # datfile（若 data 在外部 .fdt）
    fprint(f"\n[datfile] = {datfile}")

    fprint("\n✅ 完成")

if __name__ == "__main__":
    main()
