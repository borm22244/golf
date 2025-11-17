#!/usr/bin/env python3
from pylsl import resolve_streams
streams = resolve_streams(wait_time=5.0)  # 等 5 秒
if not streams:
    print("No LSL streams found.")
else:
    for i, s in enumerate(streams, 1):
        print(f"[{i}] name={s.name()} type={s.type()} ch={s.channel_count()} sr={s.nominal_srate()} uid={s.uid()}")
