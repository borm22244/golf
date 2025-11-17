#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lsl_tail.py — 接收 LSL 並持續列印
- 預設會找 type='EEG' 的 stream（可改 --type）
- 若同時存在 type='Markers'，也會啟用事件監聽
- 以 pull_chunk() 取批次，避免逐點列印太吃 CPU
"""

import argparse
import time
from typing import Optional, List, Tuple

from pylsl import StreamInlet, resolve_byprop, local_clock, StreamInfo  # pip install pylsl

def resolve_one(prop: str, value: str, timeout: float) -> Optional[StreamInfo]:
    streams = resolve_byprop(prop, value, timeout=timeout)
    return streams[0] if streams else None

def open_inlet(prop: str, value: str, timeout: float, max_buflen: float) -> Optional[StreamInlet]:
    info = resolve_one(prop, value, timeout)
    if info is None:
        return None
    inlet = StreamInlet(info, max_buflen=max_buflen)
    return inlet

def fmt_ts(ts: float) -> str:
    # 以本機 LSL 時鐘為基準，顯示相對延遲
    if ts is None:
        return "ts=?"
    lag = local_clock() - ts
    return f"ts={ts:.6f} lag={lag*1000:.1f}ms"

def main():
    ap = argparse.ArgumentParser(description="Receive LSL stream and continuously print chunks.")
    ap.add_argument("--type", default="EEG", help="LSL stream type to receive (default: EEG)")
    ap.add_argument("--chunk-timeout", type=float, default=0.1, help="pull_chunk timeout in seconds")
    ap.add_argument("--resolve-timeout", type=float, default=5.0, help="resolve stream timeout in seconds")
    ap.add_argument("--buflen", type=float, default=5.0, help="max buffer length in seconds")
    ap.add_argument("--limit", type=int, default=0, help="max printed chunks (0 = unlimited)")
    ap.add_argument("--print-samples", action="store_true", help="print actual numeric samples (可能很多)")
    ap.add_argument("--no-markers", action="store_true", help="do not try to open Markers stream")
    args = ap.parse_args()

    print(f"[INFO] Resolving type='{args.type}' ...")
    inlet = open_inlet("type", args.type, args.resolve_timeout, args.buflen)
    if inlet is None:
        print(f"[ERR ] No LSL stream found for type='{args.type}'."); return
    info = inlet.info()
    print(f"[OK  ] Connected EEG stream: name='{info.name()}', type='{info.type()}', "
          f"chs={info.channel_count()}, srate={info.nominal_srate()}Hz")

    marker_inlet: Optional[StreamInlet] = None
    if not args.no_markers:
        print(f"[INFO] Resolving type='Markers' ...")
        marker_inlet = open_inlet("type", "Markers", 1.0, args.buflen)
        if marker_inlet:
            print(f"[OK  ] Connected Markers stream: name='{marker_inlet.info().name()}'")
        else:
            print(f"[WARN] No Markers stream found (continue without markers).")

    printed = 0
    try:
        while args.limit == 0 or printed < args.limit:
            # 先抓 EEG chunk
            samples, timestamps = inlet.pull_chunk(timeout=args.chunk_timeout)
            if samples:
                printed += 1
                n = len(samples)
                chs = len(samples[0]) if samples and hasattr(samples[0], "__len__") else 1
                t0 = timestamps[0] if timestamps else None
                t1 = timestamps[-1] if timestamps else None
                print(f"[EEG] chunk#{printed:06d} n_samples={n} n_channels={chs} "
                      f"{fmt_ts(t0)} .. {fmt_ts(t1)}")
                if args.print_samples:
                    # 小心輸出量，僅示範前 3 筆
                    preview = samples[:3]
                    print("       first 3 samples:")
                    for i, row in enumerate(preview):
                        print(f"         {i:02d}: {row}")

            # 再抓 Markers（若有）
            if marker_inlet:
                mks, mk_ts = marker_inlet.pull_chunk(timeout=0.0)
                if mks:
                    for tag, ts in zip(mks, mk_ts):
                        # mks 可能是 [['228'], ['1']] 這種巢狀
                        label = tag[0] if isinstance(tag, (list, tuple)) and tag else tag
                        print(f"[MRK] {fmt_ts(ts)} label={label}")

            # 降低 CPU 佔用，且避免洗版太兇
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C). Bye.")

if __name__ == "__main__":
    main()
