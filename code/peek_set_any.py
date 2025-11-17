#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, struct
from pathlib import Path
import numpy as np

def fprint(*a, **kw):
    kw.setdefault("flush", True)
    print(*a, **kw)

def sniff_format(p: Path):
    # 回傳 "v7.3" 或 "v5"
    with p.open("rb") as fh:
        head = fh.read(8)
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "v7.3"
    if head.startswith(b"MATLAB 5.0 MAT-file"):
        return "v5"
    # 一些 EEGLAB .set 其實是 ASCII/Struct 容器，也讓後續後端判斷
    return "unknown"

def try_mne(path: str):
    try:
        import mne
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        info = raw.info
        data = raw.get_data()
        return {
            "backend": "mne",
            "sfreq": info.get("sfreq"),
            "nchan": info.get("nchan"),
            "ch_names": info.get("ch_names"),
            "n_times": raw.n_times,
            "data": data,  # (n_channels, n_times), in Volts
            "annotations": raw.annotations,
        }
    except Exception as e:
        return {"error": repr(e), "backend": "mne"}

def try_pymatreader(path: str):
    try:
        from pymatreader import read_mat
        d = read_mat(path, ignore_fields=[], variable_names=None)  # 可能走 scipy 或 h5py
        return {"backend": "pymatreader", "dict": d}
    except Exception as e:
        return {"error": repr(e), "backend": "pymatreader"}

def try_mat73(path: str):
    # 專讀 MATLAB v7.3 (HDF5)
    try:
        import mat73
        d = mat73.loadmat(path)
        return {"backend": "mat73", "dict": d}
    except Exception as e:
        return {"error": repr(e), "backend": "mat73"}

def read_fdt_sidecar(set_path: Path, EEG):
    """
    若 EEG['data'] 是外部 .fdt，依 nbchan/pnts/trials 嘗試讀前幾個 samples。
    """
    try:
        datfile = EEG.get("datfile") or EEG.get("datafile")
    except AttributeError:
        datfile = None
    if isinstance(datfile, (list, tuple)) and len(datfile) > 0:
        datfile = datfile[0]
    if not datfile:
        return None

    fdt_path = (set_path.parent / str(datfile)).resolve()
    if not fdt_path.exists():
        return f"找不到 .fdt：{fdt_path}"

    nbchan = int(EEG.get("nbchan", 0))
    pnts   = int(EEG.get("pnts", 0))
    trials = int(EEG.get("trials", 1) or 1)
    dtype  = np.float32  # EEGLAB 預設 float32
    total  = nbchan * pnts * trials
    if total <= 0:
        return "無法從 EEG 結構推算形狀（nbchan/pnts/trials）"

    # 只讀前面一小塊
    preview_samples = min(10, pnts)
    try:
        with open(fdt_path, "rb") as fh:
            arr = np.fromfile(fh, dtype=dtype, count=nbchan * preview_samples * trials)
        # reshape 成 (trials, nbchan, pnts)
        try:
            arr = arr.reshape(trials, nbchan, preview_samples, order="F")  # EEGLAB 多用 column-major
        except Exception:
            arr = arr.reshape(trials, nbchan, preview_samples, order="C")
        return {"fdt_path": str(fdt_path), "preview": arr, "nbchan": nbchan, "pnts": pnts, "trials": trials}
    except Exception as e:
        return f".fdt 預覽失敗：{e}"

def pretty_print_from_dict(set_path: Path, D):
    """
    從 dict 形式的原始 MATLAB 結構，盡力找 EEG 物件，印出基本內容。
    """
    keys = list(D.keys())
    fprint("\n[MAT keys]", keys[:20], "..." if len(keys) > 20 else "")
    EEG = None
    for k in ("EEG", "eeg", "EEG1"):
        if k in D:
            EEG = D[k]
            break
    if EEG is None:
        fprint("找不到 'EEG' 結構，印出頂層摘要即可。")
        return

    def get(k, default=None):
        try:
            return EEG.get(k, default)
        except AttributeError:
            # 有些時候是 object-like
            return getattr(EEG, k, default)

    fprint("\n[EEG 基本欄位]")
    nbchan = get("nbchan")
    pnts   = get("pnts")
    trials = get("trials")
    srate  = get("srate")
    chanlocs = get("chanlocs")
    fprint(f"nbchan={nbchan}, pnts={pnts}, trials={trials}, srate={srate}")
    if chanlocs is not None:
        try:
            names = []
            # chanlocs 可能是 list/dict/struct
            for c in chanlocs:
                nm = getattr(c, "labels", None) or getattr(c, "label", None) or c.get("labels") or c.get("label")
                if nm is not None:
                    names.append(str(nm))
                if len(names) >= 10:
                    break
            fprint("ch_names(前10):", names)
        except Exception:
            pass

    data = get("data")
    if isinstance(data, np.ndarray):
        fprint(f"data 是 numpy 陣列，shape={data.shape}, dtype={data.dtype}")
        # 只印前 5x5
        slc = data.ravel()[:25]
        fprint("data 前 25 元素:", np.array2string(slc, precision=3, separator=", "))
    else:
        fprint(f"data 不是直接的 ndarray（可能在 .fdt），型別：{type(data)}")

        fdt_preview = read_fdt_sidecar(set_path, EEG)
        if isinstance(fdt_preview, dict):
            pr = fdt_preview
            fprint(f"偵測到 .fdt：{pr['fdt_path']}")
            fprint(f"預估形狀：(trials={pr['trials']}, nbchan={pr['nbchan']}, pnts={pr['pnts']})")
            fprint("每通道前 10 個樣本 (trial 0, 前 5 通道)：")
            pv = pr["preview"]
            for ch in range(min(5, pv.shape[1])):
                fprint(f"ch{ch}: {np.array2string(pv[0, ch, :], precision=2, separator=', ')}")
        else:
            fprint(f"無法讀 .fdt 預覽：{fdt_preview}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_path", required=True)
    args = ap.parse_args()
    p = Path(args.set_path)
    fprint(f"=== 嘗試讀取：{p} ===")

    fmt = sniff_format(p)
    fprint(f"[檔頭偵測] MATLAB 格式：{fmt}")

    # 1) 先試 MNE（若成功最省事）
    r = try_mne(str(p))
    if "error" not in r:
        fprint("\n[用 MNE 讀取成功]")
        fprint(f"sfreq={r['sfreq']}, nchan={r['nchan']}, n_times={r['n_times']}")
        chn = r["ch_names"][:10] if r["ch_names"] else []
        fprint("ch_names(前10):", chn)
        data = r["data"]
        head = min(10, data.shape[1])
        fprint(f"\n每通道前 {head} 個樣本 (µV, 前 5 通道)：")
        for i in range(min(5, data.shape[0])):
            fprint(f"ch{i}: {np.array2string((data[i,:head]*1e6), precision=2, separator=', ')}")
        ann = r["annotations"]
        if ann is None or len(ann) == 0:
            fprint("\nAnnotations: 無")
        else:
            fprint(f"\nAnnotations: {len(ann)} 筆，前 5：")
            for i in range(min(5, len(ann))):
                fprint(f"{i}: onset={ann.onset[i]:.3f}s, dur={ann.duration[i]:.3f}s, desc={ann.description[i]}")
        fprint("\n✅ 完成（MNE）")
        return

    fprint("[MNE 失敗] ->", r.get("error"))

    # 2) 再試 pymatreader（會自動選 scipy/h5py）
    r2 = try_pymatreader(str(p))
    if "dict" in r2:
        fprint("\n[用 pymatreader 讀取成功，列印結構]")
        pretty_print_from_dict(p, r2["dict"])
        fprint("\n✅ 完成（pymatreader）")
        return
    else:
        fprint("[pymatreader 失敗] ->", r2.get("error"))

    # 3) 若是 v7.3，改用 mat73
    if fmt == "v7.3" or True:
        r3 = try_mat73(str(p))
        if "dict" in r3:
            fprint("\n[用 mat73 讀取成功，列印結構]")
            pretty_print_from_dict(p, r3["dict"])
            fprint("\n✅ 完成（mat73）")
            return
        else:
            fprint("[mat73 失敗] ->", r3.get("error"))

    fprint("\n❌ 仍無法讀取。請確認：")
    fprint("1) 同名 .fdt 是否和 .set 在同一資料夾")
    fprint("2) 檔案是否完整（不是 0 bytes，沒有被中斷複製）")
    fprint("3) 若是 EEGLAB 儲存，試著在 EEGLAB 重新另存一次（建議 v7.3/HDF5），或移除過長/非 ASCII 的事件描述後再存")

if __name__ == "__main__":
    main()
