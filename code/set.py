#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小示範：讀取 EEGLAB .set，直接列印資料結構與前幾筆內容。
不做濾波、不重取樣，只看「讀進來長什麼樣子」。
"""

import argparse
from pathlib import Path
import numpy as np
import mne

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_path", required=True, help="EEGLAB .set 檔路徑（如 data/sample.set）")
    parser.add_argument("--head", type=int, default=5, help="要預覽的時間點數（samples）")
    parser.add_argument("--show_events", action="store_true", help="嘗試從 annotations 轉 events 並列印")
    args = parser.parse_args()

    set_path = Path(args.set_path)
    if not set_path.exists():
        raise FileNotFoundError(f"找不到檔案：{set_path}")

    print(f"\n=== 讀檔：{set_path} ===")
    # 讀 Raw（連續資料）；若是已切段之 .set，可改用 mne.read_epochs_eeglab
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    print("\n[Raw 物件]")
    print(raw)  # MNE 的摘要

    info = raw.info
    n_channels = info["nchan"]
    sfreq = info["sfreq"]
    ch_names = info["ch_names"]

    print("\n[基本資訊]")
    print(f"通道數 (n_channels): {n_channels}")
    print(f"取樣率 (sfreq, Hz): {sfreq}")
    print(f"通道名稱（前10個）: {ch_names[:10]}{'...' if len(ch_names) > 10 else ''}")

    n_times = raw.n_times
    duration_sec = n_times / sfreq if sfreq else None
    print(f"總點數 (n_times): {n_times}")
    print(f"推算時長 (sec): {duration_sec:.3f}" if duration_sec is not None else "推算時長 (sec): N/A")

    # 取出資料陣列：shape = (n_channels, n_times)
    # MNE 以 Volts 表示 EEG；若原檔是 µV，MNE 會轉換到 V。
    data = raw.get_data()  # numpy.ndarray
    print("\n[資料陣列 data]")
    print(f"type: {type(data)}, dtype: {data.dtype}, shape: {data.shape} (channels x times)")

    # 顯示前 head 個時間點（轉成 µV 方便人看）
    head = min(args.head, data.shape[1])
    print(f"\n[前 {head} 個 sample（轉 µV 顯示）]")
    preview = (data[:, :head] * 1e6)  # V -> µV
    # 只列印前最多 10 個通道，避免洗版
    max_show_ch = min(10, data.shape[0])
    for i in range(max_show_ch):
        print(f"{ch_names[i]:>10s}: {np.array2string(preview[i, :], precision=2, separator=', ')}")
    if data.shape[0] > max_show_ch:
        print("... (其餘通道省略)")

    # 列出 Annotations（EEGLAB 的 event 會被放在這裡）
    ann = raw.annotations
    print("\n[Annotations]")
    if ann is None or len(ann) == 0:
        print("（沒有 annotations）")
    else:
        print(f"總數：{len(ann)}")
        # 列前 10 筆
        for i in range(min(10, len(ann))):
            print(f"{i:>3d}: onset={ann.onset[i]:.3f}s, duration={ann.duration[i]:.3f}s, desc='{ann.description[i]}'")
        if len(ann) > 10:
            print("... (其餘省略)")

    # 嘗試把 annotations 轉成 events（數值碼）與 event_id（描述->碼）
    if args.show_events and ann is not None and len(ann) > 0:
        try:
            print("\n[Events（由 annotations 轉換）]")
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            print(f"events shape: {events.shape}（每列 = [sample_index, 0, event_code]）")
            # 只顯示前 10 筆
            for i in range(min(10, len(events))):
                print(f"{i:>3d}: sample={events[i,0]}, code={events[i,2]}")
            print(f"event_id 對照: {event_id}")
        except Exception as e:
            print(f"轉換 events 失敗：{e}")

    # 若檔案本身是 epoch 結構（有些 .set 會是已切段格式），可嘗試讀 epochs
    # 這段不一定成功，純粹示範
    try:
        epochs = mne.read_epochs_eeglab(set_path, verbose=False, preload=True)
        print("\n[嘗試讀取 Epochs 結構]")
        print(epochs)
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        print(f"epochs data shape: {X.shape}")
        print(f"epochs events shape: {epochs.events.shape}, event_id: {epochs.event_id}")
    except Exception:
        print("\n[Epochs] 此檔案似乎不是 epochs 形式（略過）")

    print("\n✅ 完成：以上就是 .set 讀進 Python 後的實際結構與內容預覽。")

if __name__ == "__main__":
    main()
