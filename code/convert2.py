#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge multiple EEGLAB .set files into a single NPZ with per-trial dimensions.

Outputs
-------
X:            (N_trials, C, T) float32/float64
subject_ids:  (N_trials,) int32             # index into `subjects`
subjects:     (n_subjects,) object          # subject names (e.g., FA10)
set_ids:      (N_trials,) int32             # index into `sets`
sets:         (n_sets,) object              # set (recording) names
resp_labels:  (N_trials,) int8              # 1=S, 0=U, -1=?
resp_chars:   (N_trials,) object            # 'S'/'U'/'?'
task_ids:     (N_trials,) int32             # index into `tasks`
tasks:        (n_tasks,) object             # task strings (e.g., 'Apn')
cond_ids:     (N_trials,) int32             # index into `conds`
conds:        (n_conds,) object             # condition strings ('eas','mid','dif', ...)
srate:        (N_trials,) float32
C, T, n_trials: scalar meta

Usage
-----
python3 merge_set_to_npz.py ./eeg_sets -R -o all_trials_merged.npz --float32 --verbose
"""

import argparse
from pathlib import Path
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.io import loadmat

NEEDED = ["setname", "nbchan", "pnts", "trials", "srate", "data"]

# 解析主架構 (subject/task/cond)；resp 改由更穩健方法另抓
# e.g. "FA12_Apn_dif_ICAdone_U", "FA10-Apn-eas-ICAdone-S"
PAT = re.compile(
    r"(?P<subj>[A-Za-z]*\d+)"
    r"(?:[_\-](?P<task>[A-Za-z]+))?"
    r"(?:[_\-](?P<cond>[A-Za-z]+))?",
    re.IGNORECASE
)

def collect_set_paths(inputs: List[str], recursive: bool) -> List[Path]:
    paths: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file() and p.suffix.lower() == ".set":
            paths.append(p.resolve())
        elif p.is_dir():
            g = "**/*.set" if recursive else "*.set"
            paths.extend(sorted(p.glob(g)))
        else:
            paths.extend(sorted(Path(".").glob(inp)))
    paths = [x.resolve() for x in paths if x.is_file() and x.suffix.lower() == ".set"]
    return sorted(set(paths))

def _as_str_from_mat(val: Any) -> Optional[str]:
    """將 loadmat 讀到的 setname 轉成純 str（處理 0-D ndarray、np.str_ 等）。"""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, np.ndarray):
        try:
            return str(val.tolist())
        except Exception:
            pass
    if hasattr(val, "item"):
        try:
            v = val.item()
            return str(v)
        except Exception:
            return str(val)
    return str(val)

def _extract_resp_token(text: str) -> Optional[str]:
    """
    從字尾 token 找獨立的 S/U（處理尾端空白、底線、連字號、點號）。
    範例可抓到："..._S", "...-U", "... S", "..._U_epochs", "FA12_Apn_dif_ICAdone_U"
    """
    text = (text or "").strip()
    if not text:
        return None
    tokens = re.split(r"[_\-\.\s]+", text)
    for tok in reversed(tokens):
        t = tok.upper().strip()
        if t in ("S", "U"):
            return t
    m = re.search(r"(?:^|[_\-\.\s])(S|U)\s*$", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def parse_name(setname: Optional[str], path_stem: str) -> Tuple[str, str, str, str]:
    """
    回傳 (subject, task, cond, resp_char)。
    - subject/task/cond：優先從 setname 抓，抓不到退回 path.stem。
    - resp：會同時嘗試 setname 與 path.stem（以字尾 token 為主）。
    """
    name = (setname or "").strip()
    stem = (path_stem or "").strip()

    m = PAT.search(name) or PAT.search(stem)
    subj = (m.group("subj") or "").strip() if m else ""
    task = (m.group("task") or "").strip() if m else ""
    cond = ((m.group("cond") or "").strip().lower()) if m else ""

    resp = _extract_resp_token(name) or _extract_resp_token(stem)
    if resp not in ("S", "U"):
        resp = "?"
    return subj, task, cond, resp

def load_one_set(path: Path) -> Dict[str, Any]:
    M = loadmat(
        path,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=False,
        appendmat=False,
        variable_names=NEEDED,
    )
    def get(k, default=None): return M.get(k, default)
    setname = _as_str_from_mat(get("setname", None)) or path.stem

    nbchan  = int(get("nbchan", 0) or 0)
    pnts    = int(get("pnts", 0) or 0)
    trials  = int(get("trials", 1) or 1)
    srate   = float(get("srate", 0.0) or 0.0)
    data    = get("data", None)

    if not isinstance(data, np.ndarray):
        raise ValueError(f"data is not ndarray in {path}")

    # 期望 (C, T, trials)。若為 (T, C, trials) 則轉置；若無 trials 維則擴一維。
    if data.ndim == 2:
        data = data[:, :, None]
    if data.shape[0] != nbchan or data.shape[1] != pnts:
        if data.shape[0] == pnts and data.shape[1] == nbchan:
            data = np.transpose(data, (1, 0, 2))  # -> (C, T, trials)
        else:
            raise ValueError(f"Unexpected data shape {data.shape} vs nbchan={nbchan}, pnts={pnts}")

    return dict(
        path=str(path),
        setname=str(setname),
        nbchan=nbchan,
        pnts=pnts,
        trials=trials,
        srate=srate,
        data=data  # (C, T, trials)
    )

def main():
    ap = argparse.ArgumentParser(description="Merge .set files into a single NPZ with subject/task/cond/SU per-trial dimensions.")
    ap.add_argument("inputs", nargs="+", help="Files/dirs/globs (multiple allowed)")
    ap.add_argument("-R", "--recursive", action="store_true", help="Recurse into directories")
    ap.add_argument("-o", "--out", required=True, help="Output .npz path")
    ap.add_argument("--float32", action="store_true", help="Cast data to float32 (recommended)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    paths = collect_set_paths(args.inputs, args.recursive)
    if not paths:
        raise SystemExit("No .set files found.")

    X_list: List[np.ndarray] = []        # per-file -> (trials, C, T)

    subjects_vocab: List[str] = []
    sets_vocab: List[str] = []
    tasks_vocab: List[str] = []
    conds_vocab: List[str] = []          # 'eas','mid','dif', ...

    subj_to_id: Dict[str, int] = {}
    set_to_id: Dict[str, int] = {}
    task_to_id: Dict[str, int] = {}
    cond_to_id: Dict[str, int] = {}

    subject_ids: List[int] = []
    set_ids: List[int] = []
    task_ids: List[int] = []
    cond_ids: List[int] = []

    resp_labels: List[int] = []
    resp_chars: List[str] = []
    tasks: List[str] = []
    conds: List[str] = []
    srates: List[float] = []

    n_files = len(paths)
    for i, p in enumerate(paths, 1):
        info = load_one_set(p)
        setname = info["setname"]
        stem = Path(p).stem

        subj, task, cond, resp_char = parse_name(setname, stem)

        # Subject id
        subj_key = subj if subj else stem.split("_")[0]
        if subj_key not in subj_to_id:
            subj_to_id[subj_key] = len(subjects_vocab)
            subjects_vocab.append(subj_key)
        sid = subj_to_id[subj_key]

        # Set id
        if setname not in set_to_id:
            set_to_id[setname] = len(sets_vocab)
            sets_vocab.append(setname)
        setid = set_to_id[setname]

        # Task id
        task_key = task or ""
        if task_key not in task_to_id:
            task_to_id[task_key] = len(tasks_vocab)
            tasks_vocab.append(task_key)
        tid = task_to_id[task_key]

        # Cond id（小寫）
        cond_key = (cond or "").lower()
        if cond_key not in cond_to_id:
            cond_to_id[cond_key] = len(conds_vocab)
            conds_vocab.append(cond_key)
        cid = cond_to_id[cond_key]

        C, T, K = int(info["nbchan"]), int(info["pnts"]), int(info["trials"])
        arr = info["data"]  # (C, T, K)
        if args.float32 and arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        # (K, C, T)
        arr_trials = np.transpose(arr, (2, 0, 1))
        X_list.append(arr_trials)

        # resp: S->1, U->0, 其他->-1
        if resp_char == "S":
            rlab = 1
        elif resp_char == "U":
            rlab = 0
        else:
            rlab = -1

        # 複製 per-trial meta
        subject_ids.extend([sid] * K)
        set_ids.extend([setid] * K)
        task_ids.extend([tid] * K)
        cond_ids.extend([cid] * K)
        resp_labels.extend([rlab] * K)
        resp_chars.extend([resp_char] * K)
        tasks.extend([task_key] * K)
        conds.extend([cond_key] * K)
        srates.extend([float(info["srate"])] * K)

        if args.verbose:
            print(f"[{i}/{n_files}] {p.name} -> trials={K}, subj={subj_key}, task={task_key}, cond={cond_key}, resp={resp_char}")

    # 串接
    X = np.concatenate(X_list, axis=0)  # (N, C, T)

    subject_ids = np.asarray(subject_ids, dtype=np.int32)
    set_ids = np.asarray(set_ids, dtype=np.int32)
    task_ids = np.asarray(task_ids, dtype=np.int32)
    cond_ids = np.asarray(cond_ids, dtype=np.int32)

    resp_labels = np.asarray(resp_labels, dtype=np.int8)
    resp_chars = np.asarray(resp_chars, dtype=object)
    tasks = np.asarray(tasks, dtype=object)
    conds = np.asarray(conds, dtype=object)
    srates = np.asarray(srates, dtype=np.float32)

    N, C, T = X.shape
    assert subject_ids.shape[0] == N
    assert set_ids.shape[0] == N
    assert task_ids.shape[0] == N
    assert cond_ids.shape[0] == N
    assert resp_labels.shape[0] == N
    assert resp_chars.shape[0] == N
    assert tasks.shape[0] == N
    assert conds.shape[0] == N
    assert srates.shape[0] == N

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X,
        subject_ids=subject_ids,
        subjects=np.array(subjects_vocab, dtype=object),
        set_ids=set_ids,
        sets=np.array(sets_vocab, dtype=object),
        resp_labels=resp_labels,
        resp_chars=resp_chars,
        task_ids=task_ids,
        tasks=np.array(tasks_vocab, dtype=object),
        cond_ids=cond_ids,
        conds=np.array(conds_vocab, dtype=object),
        srate=srates,
        C=np.int32(C),
        T=np.int32(T),
        n_trials=np.int32(N),
    )
    print(f"✅ Saved: {out_path} | X={X.shape} | subjects={len(subjects_vocab)} | sets={len(sets_vocab)} | "
          f"tasks={len(tasks_vocab)} | conds={len(conds_vocab)} | trials={N}")

if __name__ == "__main__":
    main()
