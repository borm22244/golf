#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把多個 EEGLAB .set (MAT v5) 檔案彙整成單一 JSON。
預設輸出「中小型摘要」，可選擇輸出 data 預覽或整份 data（可能非常大）。
"""

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Dict, Any, Union

import numpy as np
from scipy.io import loadmat

NEEDED = [
    "setname", "nbchan", "pnts", "trials", "srate", "times",
    "data", "chanlocs", "event", "epoch", "datfile"
]

def fprint(*a, **kw):
    kw.setdefault("flush", True)
    print(*a, **kw)

# ---- 核心修正：JSON 轉換器，處理 numpy/Path/NaN/Inf ----
def _json_default(o):
    # NumPy 標量 -> Python 標量
    if isinstance(o, np.generic):
        v = o.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    # NumPy 陣列 -> list（並處理 NaN/Inf）
    if isinstance(o, np.ndarray):
        arr = o.astype(object, copy=False)
        it = np.nditer(arr, flags=["refs_ok", "multi_index"], op_flags=["readwrite"])
        for x in it:
            xv = x.item()
            if isinstance(xv, float) and not math.isfinite(xv):
                x[...] = None
            elif isinstance(xv, np.generic):
                x[...] = xv.item()
        return arr.tolist()
    # Path -> str
    if isinstance(o, Path):
        return str(o)
    # 其他未知型別 -> 字串保底
    return str(o)

def to_plain(obj):
    """把 MATLAB v5 的 mat_struct / 結構陣列 轉成 Python 原生型別（遞迴）"""
    if hasattr(obj, "__dict__"):  # mat_struct
        return {k: to_plain(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, np.void) and getattr(obj, "dtype", None) is not None and obj.dtype.names:
        return {name: to_plain(obj[name]) for name in obj.dtype.names}
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            return np.vectorize(to_plain, otypes=[object])(obj)
        else:
            return obj
    return obj

def collect_paths(inputs: List[str], recursive: bool) -> List[Path]:
    """將路徑 / 目錄 / GLOB 展開成 .set 檔清單"""
    out: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file() and p.suffix.lower() == ".set":
            out.append(p.resolve())
        elif p.is_dir():
            pattern = "**/*.set" if recursive else "*.set"
            out.extend(sorted((p.glob(pattern))))
        else:
            # 當作 glob pattern
            base = Path(".")
            for cand in base.glob(inp):
                if cand.is_file() and cand.suffix.lower() == ".set":
                    out.append(cand.resolve())
                elif cand.is_dir():
                    pattern = "**/*.set" if recursive else "*.set"
                    out.extend(sorted((cand.glob(pattern))))
    # 去重並排序
    uniq = sorted({p.resolve() for p in out})
    return uniq

def simplify_chanlocs(chanlocs: Any) -> List[str]:
    names: List[str] = []
    if chanlocs is None:
        return names
    plain = to_plain(chanlocs)
    arr = np.atleast_1d(plain)
    for c in arr.flat:
        if isinstance(c, dict):
            nm = c.get("labels") or c.get("label") or ""
        else:
            nm = str(c)
        names.append(str(nm))
    return names

def _to_py_num(x):
    """保守轉換為純 Python 數值或 None（處理 numpy 標量與 NaN/Inf）"""
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x

def simplify_events(event: Any, max_items: int = None) -> List[Dict[str, Any]]:
    if event is None:
        return []
    ev_plain = to_plain(event)
    ev_arr = np.atleast_1d(ev_plain)
    out: List[Dict[str, Any]] = []
    it = ev_arr.flat
    if max_items is not None and max_items > 0:
        it = (e for i, e in enumerate(ev_arr.flat) if i < max_items)
    for e in it:
        if isinstance(e, dict):
            out.append({
                "type": (e.get("type") if not isinstance(e.get("type"), np.generic) else str(e.get("type"))),
                "latency": _to_py_num(e.get("latency")),
                "duration": _to_py_num(e.get("duration")),
                "epoch": _to_py_num(e.get("epoch")),
            })
        else:
            out.append({"raw": str(e)})
    return out

def times_stats(times: Any) -> Union[None, Dict[str, Any]]:
    if isinstance(times, np.ndarray):
        try:
            mn = float(np.nanmin(times))
            mx = float(np.nanmax(times))
            if not math.isfinite(mn): mn = None
            if not math.isfinite(mx): mx = None
            return {
                "count": int(times.size),
                "min_ms": mn,
                "max_ms": mx,
            }
        except Exception:
            return {"count": int(times.size)}
    return None

def extract_preview_data(data: Any, head: int, trials: int, ch_names: List[str]) -> Dict[str, Any]:
    """輸出小型預覽：每通道前 head 個 sample、前 preview_trials 個 trial"""
    if not isinstance(data, np.ndarray):
        return {"note": "data 不存在或不是 ndarray"}
    # data shape: (nbchan, pnts, trials)
    nbchan = int(data.shape[0])
    pnts   = int(data.shape[1]) if data.ndim >= 2 else 0
    trl    = int(data.shape[2]) if data.ndim >= 3 else 1
    take_trials = min(max(1, trials), trl)
    take_head   = min(max(1, head), pnts)

    preview: Dict[str, Any] = {}
    # 第一個 trial 的每通道前 N 點
    arr1 = np.array(data[:, :take_head, 0], copy=False)
    if arr1.dtype.kind not in ("f", "i"):
        arr1 = arr1.astype(float, copy=False)
    preview_first_trial = []
    for ch in range(arr1.shape[0]):
        nm = ch_names[ch] if ch < len(ch_names) and ch_names[ch] else f"ch{ch}"
        vals = arr1[ch, :].astype(float).tolist()
        preview_first_trial.append({
            "channel": nm,
            "values": vals,
        })
    preview["first_trial_head"] = preview_first_trial
    # 其他 trial 的第一通道前 N 點
    if take_trials > 1:
        arr_multi = np.array(data[0, :take_head, :take_trials], copy=False)
        if arr_multi.dtype.kind not in ("f", "i"):
            arr_multi = arr_multi.astype(float, copy=False)
        preview["channel0_multi_trials_head"] = {
            "channel": ch_names[0] if ch_names and ch_names[0] else "ch0",
            "values_2d": arr_multi.astype(float).tolist(),  # [head, trials]
        }
    preview["shape"] = tuple(int(s) for s in data.shape)
    preview["dtype"] = str(data.dtype)
    return preview

def extract_one_set(set_path: Path,
                    include_data: str,
                    head: int,
                    preview_trials: int,
                    max_events: int) -> Dict[str, Any]:
    """讀取單一 .set 並輸出摘要 / 預覽 / 全量資料"""
    M = loadmat(
        set_path,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=False,
        appendmat=False,
        variable_names=NEEDED,
    )

    def get(k, default=None):
        return M.get(k, default)

    setname = get("setname", None)
    if isinstance(setname, np.generic):
        setname = setname.item()

    nbchan  = int(get("nbchan", 0) or 0)
    pnts    = int(get("pnts", 0) or 0)
    trials  = int(get("trials", 1) or 1)
    srate   = float(get("srate", 0.0) or 0.0)
    times   = get("times", None)
    data    = get("data", None)
    chanlocs= get("chanlocs", None)
    event   = get("event", None)
    epoch   = get("epoch", None)
    datfile = get("datfile", None)
    if isinstance(datfile, np.generic):
        datfile = datfile.item()

    ch_names = simplify_chanlocs(chanlocs)
    ev_list  = simplify_events(event, max_items=max_events if max_events and max_events > 0 else None)
    ep_count = int(np.atleast_1d(to_plain(epoch)).size) if epoch is not None else 0
    ev_count = int(np.atleast_1d(to_plain(event)).size) if event is not None else 0

    out: Dict[str, Any] = {
        "path": str(set_path),
        "setname": setname,
        "nbchan": int(nbchan),
        "pnts": int(pnts),
        "trials": int(trials),
        "srate": float(srate),
        "times_stats": times_stats(times),
        "channel_names": [str(x) for x in ch_names],
        "event_count": int(ev_count),
        "events": ev_list,
        "epoch_count": int(ep_count),
        "datfile": datfile if datfile is None else str(datfile),
    }

    if include_data == "none":
        pass
    elif include_data == "preview":
        out["data_preview"] = extract_preview_data(data, head=head, trials=preview_trials, ch_names=ch_names)
    elif include_data == "full":
        if isinstance(data, np.ndarray):
            # 警告：非常大！使用者自負風險
            out["data"] = data.astype(float).tolist()
            out["data_shape"] = tuple(int(s) for s in data.shape)
            out["data_dtype"] = str(data.dtype)
        else:
            out["data"] = None
            out["data_note"] = "data 不存在或不是 ndarray"
    else:
        out["note"] = f"未知 include_data 模式：{include_data}"

    return out

def main():
    ap = argparse.ArgumentParser(description="把多個 .set 檔案彙整成單一 JSON")
    ap.add_argument("inputs", nargs="+", help="輸入路徑（檔案 / 目錄 / GLOB）。可多個。例： data/*.set  或  /path/to/dir")
    ap.add_argument("-o", "--out", required=True, help="輸出 JSON 路徑")
    ap.add_argument("-R", "--recursive", action="store_true", help="掃描目錄時遞迴搜尋 .set")
    ap.add_argument("--include-data", choices=["none", "preview", "full"], default="preview",
                    help="輸出 data 內容的方式：none=不含資料、preview=小型預覽、full=整份資料（非常大）")
    ap.add_argument("--head", type=int, default=10, help="預覽模式：每通道輸出前 N 個 sample（預設 10）")
    ap.add_argument("--preview-trials", type=int, default=3, help="預覽模式：最多輸出前 M 個 trial 的 channel0（預設 3）")
    ap.add_argument("--max-events", type=int, default=200, help="事件輸出最多筆數（避免 JSON 爆量；0 或負值表示全部）")
    ap.add_argument("--indent", type=int, default=2, help="JSON 縮排（預設 2）")
    args = ap.parse_args()

    paths = collect_paths(args.inputs, recursive=args.recursive)
    if not paths:
        raise SystemExit("找不到任何 .set 檔案，請確認輸入路徑 / GLOB。")

    fprint(f"找到 {len(paths)} 個 .set 檔案，開始處理…")

    results: List[Dict[str, Any]] = []
    for i, p in enumerate(paths, 1):
        fprint(f"[{i}/{len(paths)}] 讀取：{p}")
        try:
            item = extract_one_set(
                p,
                include_data=args.include_data,
                head=args.head,
                preview_trials=args.preview_trials,
                max_events=args.max_events,
            )
            results.append(item)
        except Exception as e:
            fprint(f"⚠️ 讀取失敗：{p} :: {e}")
            results.append({"path": str(p), "error": repr(e)})

    bundle = {
        "count": len(results),
        "include_data": args.include_data,
        "files": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # 核心修正：提供 default 讓 JSON 能序列化 numpy/Path/NaN/Inf 等
        json.dump(bundle, f, ensure_ascii=False, indent=args.indent, default=_json_default)

    fprint(f"\n✅ 完成，已輸出：{out_path}")

if __name__ == "__main__":
    main()
