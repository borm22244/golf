#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, time
from typing import Optional
from pylsl import StreamInlet, resolve_byprop, local_clock, StreamInfo

def wait_stream(prop: str, value: str, timeout: float, retry: float) -> StreamInfo:
    while True:
        ss = resolve_byprop(prop, value, timeout=timeout)
        if ss:
            return ss[0]
        print(f"[WAIT] No stream for {prop}={value!r}. Retry in {retry}s ...")
        time.sleep(retry)

def fmt_ts(ts: float) -> str:
    if ts is None:
        return "ts=?"
    lag_ms = (local_clock() - ts) * 1000.0
    return f"ts={ts:.6f} lag={lag_ms:.1f}ms"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid",  help="connect by source_id/uid (best)")
    ap.add_argument("--name", help="connect by name")
    ap.add_argument("--type", help="connect by type (default EEG if none given)")
    ap.add_argument("--chunk-timeout", type=float, default=0.5)
    ap.add_argument("--retry", type=float, default=2.0)
    ap.add_argument("--buflen", type=float, default=5.0)
    ap.add_argument("--print-samples", action="store_true")
    ap.add_argument("--markers", action="store_true", help="also connect type='Markers'")
    args = ap.parse_args()

    if not args.uid and not args.name and not args.type:
        args.type = "EEG"

    prop = "source_id" if args.uid else ("name" if args.name else "type")
    value = args.uid or args.name or args.type
    print(f"[INFO] Waiting for {prop}={value!r} ...")
    info = wait_stream(prop, value, timeout=args.chunk_timeout, retry=args.retry)

    inlet = StreamInlet(info, max_buflen=args.buflen)
    print(f"[OK] Connected: name={info.name()} type={info.type()} uid={info.uid()} "
          f"chs={info.channel_count()} sr={info.nominal_srate()}")

    marker_inlet: Optional[StreamInlet] = None
    if args.markers:
        print("[INFO] Waiting for type='Markers' ...")
        mi = wait_stream("type", "Markers", timeout=args.chunk_timeout, retry=args.retry)
        marker_inlet = StreamInlet(mi, max_buflen=args.buflen)
        print(f"[OK] Markers: name={mi.name()} uid={mi.uid()}")

    n = 0
    try:
        while True:
            samples, ts = inlet.pull_chunk(timeout=args.chunk_timeout)
            if samples:
                n += 1
                chs = len(samples[0]) if hasattr(samples[0], "__len__") else 1
                print(f"[EEG] chunk#{n:06d} n={len(samples)} chs={chs} "
                      f"{fmt_ts(ts[0])} .. {fmt_ts(ts[-1])}")
                if args.print_samples:
                    for i, row in enumerate(samples[:3]):
                        print(f"      {i:02d}: {row}")

            if marker_inlet:
                mks, mk_ts = marker_inlet.pull_chunk(timeout=0.0)
                for mk, t in zip(mks or [], mk_ts or []):
                    lab = mk[0] if isinstance(mk, (list, tuple)) and mk else mk
                    print(f"[MRK] {fmt_ts(t)} label={lab}")

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[INFO] Bye.")

if __name__ == "__main__":
    main()
