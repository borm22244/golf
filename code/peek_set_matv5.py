#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def fprint(*a, **kw):
    kw.setdefault("flush", True)
    print(*a, **kw)

def matobj_to_dict(obj):
    """把 numpy.void / MATLAB struct 轉成 Python dict（遞迴）"""
    if isinstance(obj, np.void):  # MATLAB struct
        out = {}
        for name in obj.dtype.names:
            out[name] = matobj_to_dict(obj[name])
        return out
    elif isinstance(obj, np.ndarray) and obj.dtype == np.object_:
        # 物件陣列：逐一轉
        return np.vectorize(matobj_to_dict, otypes=[object])(obj)
    else:
        return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_path", required=True, help="EEGLAB .set (MAT v5)")
    ap.add_argument("--head", type=int, default=10, help="每通道印出前幾個 sample")
    args = ap.parse_args()

    p = Path(args.set_path)
    if not p.exists():
        raise FileNotFoundError(p)

    fprint(f"=== 讀檔：{p} ===")
    # 專讀 v5：這個檔已證實是 MATLAB 5.0 MAT-file
    # squeeze_me 讓 (1,1) → 標量；chars_as_strings 讓 char 陣列變字串
    M = loadmat(
    p,
    squeeze_me=True,
    chars_as_strings=True,
    struct_as_record=False,
    appendmat=False,
    variable_names=['EEG'],   # 只讀 EEG，避開問題欄位
    )

    if "EEG" not in M:
        # 有些 .set 沒包在 EEG 變數名時，印出 keys 幫你判斷
        fprint("[錯誤] 檔內沒有 EEG 變數。可用 keys 參考：", list(M.keys()))
        return

    EEG = M["EEG"]
    # 轉成好取用的 dict（把 MATLAB struct 轉成 Python 結構）
    EEGd = matobj_to_dict(EEG)

    # 抓基本欄位
    nbchan = int(EEGd.get("nbchan"))
    pnts   = int(EEGd.get("pnts"))
    trials = int(EEGd.get("trials") or 1)
    srate  = float(EEGd.get("srate"))
    setname = EEGd.get("setname", "")
    fprint(f"[基本資訊] setname={setname}, nbchan={nbchan}, pnts={pnts}, trials={trials}, srate={srate}")

    # 通道名稱
    ch_names = []
    chanlocs = EEGd.get("chanlocs")
    if chanlocs is not None and isinstance(chanlocs, (list, np.ndarray)):
        for c in chanlocs:
            if isinstance(c, dict):
                nm = c.get("labels") or c.get("label") or ""
            else:
                # 防呆：如果不是 dict，就直接 str()
                nm = str(c)
            ch_names.append(str(nm))
    if ch_names:
        fprint("ch_names(前10)：", ch_names[:10], "..." if len(ch_names) > 10 else "")
    else:
        fprint("ch_names：從 chanlocs 無法解析（略過）")

    # 時間軸
    times = EEGd.get("times")
    if isinstance(times, np.ndarray):
        fprint(f"times shape={times.shape}, 範圍約 [{times.min():.3f}, {times.max():.3f}] ms")
    else:
        fprint("times：無或非 ndarray（略過）")

    # 資料本體：EEGLAB v5 內嵌 data，shape = (nbchan, pnts, trials)
    data = EEGd.get("data")
    if isinstance(data, np.ndarray):
        fprint(f"\n[data] type={type(data)}, dtype={data.dtype}, shape={data.shape} (nbchan x pnts x trials)")
        # 預覽：列出第一個 trial，前 head 個 sample，前 5 個通道
        head = min(args.head, data.shape[1])
        fprint(f"第一個 trial 的每通道前 {head} 個 sample（µV；EEGLAB 是 µV，多數時候 SciPy 讀進來就維持單位）")
        # 防止 Fortran-order 造成切片慢，這裡拷貝一份小片段
        preview = np.array(data[:, :head, 0], copy=True)
        for i in range(min(5, preview.shape[0])):
            nm = ch_names[i] if i < len(ch_names) else f"ch{i}"
            fprint(f"{nm:>10s}: {np.array2string(preview[i, :], precision=2, separator=', ')}")
        if data.shape[0] > 5:
            fprint("... (其餘通道省略)")
    else:
        fprint("\n[data] 不是 ndarray（不常見的情況），型別：", type(data))

    # 事件 / epoch 概述
    evt = EEGd.get("event")
    if evt is not None:
        if isinstance(evt, (list, np.ndarray)):
            fprint(f"\n[event] 事件數：{len(evt)}（列前 5）")
            for i, e in enumerate(evt[:5]):
                if isinstance(e, dict):
                    fprint(f"{i}: type={e.get('type')}, latency={e.get('latency')}, duration={e.get('duration')}")
                else:
                    fprint(f"{i}: {e}")
        else:
            fprint("\n[event] 型別：", type(evt), "（略過）")
    else:
        fprint("\n[event] 無")

    epo = EEGd.get("epoch")
    if epo is not None and isinstance(epo, (list, np.ndarray)):
        fprint(f"[epoch] 數量：{len(epo)}")
    else:
        fprint("[epoch] 無或非陣列")

    fprint("\n✅ 完成：以上就是把 .set（MAT v5）在 Python 端讀成結構並預覽的最小流程。")

if __name__ == "__main__":
    main()
