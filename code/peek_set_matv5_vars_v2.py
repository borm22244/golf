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

def to_plain(obj):
    """把 MATLAB v5 的 mat_struct / 結構陣列 轉成 Python 原生型別（遞迴）"""
    # mat_struct：有 __dict__ 的物件
    if hasattr(obj, "__dict__"):
        return {k: to_plain(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    # numpy.void（有些情境會這樣出現）
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: to_plain(obj[name]) for name in obj.dtype.names}
    # 物件陣列 / 任意 ndarray
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            return np.vectorize(to_plain, otypes=[object])(obj)
        else:
            return obj
    # 其他原生型別
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
    M = loadmat(
        p,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=False,
        appendmat=False,
        variable_names=NEEDED,
    )

    keys = [k for k in M.keys() if not k.startswith("__")]
    fprint("[實際讀到的變數 keys] ", keys)

    setname = M.get("setname", None)
    nbchan  = int(M.get("nbchan", 0) or 0)
    pnts    = int(M.get("pnts", 0) or 0)
    trials  = int((M.get("trials", 1) or 1))
    srate   = float(M.get("srate", 0.0) or 0.0)
    times   = M.get("times", None)
    data    = M.get("data", None)
    chanlocs= M.get("chanlocs", None)
    event   = M.get("event", None)
    epoch   = M.get("epoch", None)
    datfile = M.get("datfile", None)

    fprint(f"[基本資訊] setname={setname}, nbchan={nbchan}, pnts={pnts}, trials={trials}, srate={srate}")

    # ---- 通道名稱 ----
    ch_names = []
    if chanlocs is not None:
        plain = to_plain(chanlocs)
        arr = np.atleast_1d(plain)
        for c in arr.flat:
            if isinstance(c, dict):
                nm = c.get("labels") or c.get("label") or ""
            else:
                nm = str(c)
            ch_names.append(str(nm))
    if ch_names:
        fprint("ch_names(前10)：", ch_names[:10], "..." if len(ch_names) > 10 else "")
    else:
        fprint("ch_names：無法解析或不存在（略過）")

    # ---- 時間軸 ----
    if isinstance(times, np.ndarray):
        try:
            fprint(f"times shape={times.shape}, 範圍≈[{float(times.min()):.3f}, {float(times.max()):.3f}] ms")
        except Exception:
            fprint(f"times shape={times.shape}")
    else:
        fprint("times：無或非 ndarray（略過）")

    # ---- 資料本體 ----
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

    # ---- 事件 ----
    if event is not None:
        ev_plain = to_plain(event)
        ev_arr = np.atleast_1d(ev_plain)
        fprint(f"\n[event] 事件數：{ev_arr.size}（列前 5）")
        for i, e in enumerate(ev_arr.flat[:5]):
            if isinstance(e, dict):
                # EEGLAB: type 可能是字串或數字；latency 多半是以「點數」為單位
                typ = e.get("type")
                lat = e.get("latency")
                dur = e.get("duration")
                epo = e.get("epoch")
                fprint(f"{i}: type={typ}, latency={lat}, duration={dur}, epoch={epo}")
            else:
                fprint(f"{i}: {e}")
    else:
        fprint("\n[event] 無")

    # ---- epoch ----
    if epoch is not None:
        ep_plain = to_plain(epoch)
        ep_arr = np.atleast_1d(ep_plain)
        fprint(f"[epoch] 數量：{ep_arr.size}")
    else:
        fprint("[epoch] 無")

    fprint(f"\n[datfile] = {datfile}")
    fprint("\n✅ 完成")

if __name__ == "__main__":
    main()
