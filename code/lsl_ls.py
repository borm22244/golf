#!/usr/bin/env python3
from pylsl import resolve_streams
streams = resolve_streams(wait_time=3.0)  # 等3秒
if not streams:
    print("No LSL streams found.")
else:
    for i, info in enumerate(streams, 1):
        print(f"[{i}] name={info.name()}  type={info.type()}  ch={info.channel_count()}  srate={info.nominal_srate()}  uid={info.uid()}")
