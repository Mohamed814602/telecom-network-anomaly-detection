"""
generate_data.py
----------------
Generates synthetic telecom network KPI time-series data with realistic
anomaly patterns. Simulates 3 cell towers over 60 days at 15-min intervals.

KPIs simulated:
  - latency_ms       : packet round-trip latency (ms)
  - throughput_mbps  : downlink throughput (Mbps)
  - packet_loss_pct  : % of packets lost
  - sinr_db          : Signal-to-Interference-plus-Noise Ratio (dB)
  - connected_users  : active UEs on cell

Anomaly types injected:
  - Latency spike      : sudden latency jump (congestion / backhaul issue)
  - Throughput drop    : sustained throughput degradation (hardware fault)
  - Packet loss burst  : bursty packet loss (interference / link instability)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

CELLS = ["CELL_001", "CELL_002", "CELL_003"]
START = "2024-01-01"
PERIODS = 60 * 24 * 4          # 60 days × 96 intervals/day (15-min)
FREQ = "15min"


# ── Baseline KPI generators ────────────────────────────────────────────────

def base_latency(n, rng):
    """Normal latency: 20–60 ms with daily cycle (higher at peak hours)."""
    t = np.arange(n)
    daily = 10 * np.sin(2 * np.pi * t / (96))          # 96 slots = 1 day
    noise = rng.normal(0, 3, n)
    return np.clip(35 + daily + noise, 5, 200)

def base_throughput(n, rng):
    """Throughput: 50–150 Mbps, inversely correlated with load."""
    t = np.arange(n)
    daily = -20 * np.sin(2 * np.pi * t / 96)
    noise = rng.normal(0, 5, n)
    return np.clip(100 + daily + noise, 10, 300)

def base_packet_loss(n, rng):
    """Packet loss: mostly near 0, occasional small spikes."""
    base = rng.exponential(0.3, n)
    return np.clip(base, 0, 2)

def base_sinr(n, rng):
    """SINR: 5–25 dB, slight daily variation."""
    t = np.arange(n)
    daily = 3 * np.sin(2 * np.pi * t / 96 + np.pi)
    noise = rng.normal(0, 1.5, n)
    return np.clip(15 + daily + noise, -5, 30)

def base_users(n, rng):
    """Connected users: 10–200, strong daily pattern."""
    t = np.arange(n)
    daily = 80 * np.clip(np.sin(2 * np.pi * (t / 96 - 0.2)), 0, 1)
    noise = rng.normal(0, 5, n)
    return np.clip(20 + daily + noise, 1, 300).astype(int)


# ── Anomaly injectors ──────────────────────────────────────────────────────

def inject_latency_spike(latency, start, duration=8):
    """Inject a sharp latency spike (congestion event)."""
    latency = latency.copy()
    latency[start:start+duration] += rng.uniform(80, 200, duration)
    return latency

def inject_throughput_drop(throughput, start, duration=24):
    """Inject sustained throughput degradation (hardware fault)."""
    throughput = throughput.copy()
    throughput[start:start+duration] *= rng.uniform(0.1, 0.3)
    return throughput

def inject_packet_loss_burst(packet_loss, start, duration=6):
    """Inject bursty packet loss (interference)."""
    packet_loss = packet_loss.copy()
    packet_loss[start:start+duration] += rng.uniform(5, 15, duration)
    return packet_loss


# ── Main generation ────────────────────────────────────────────────────────

def generate_cell_data(cell_id: str, n: int, rng) -> pd.DataFrame:
    timestamps = pd.date_range(START, periods=n, freq=FREQ)

    latency     = base_latency(n, rng)
    throughput  = base_throughput(n, rng)
    packet_loss = base_packet_loss(n, rng)
    sinr        = base_sinr(n, rng)
    users       = base_users(n, rng)

    labels = np.zeros(n, dtype=int)

    # Inject ~3–5 anomaly events per cell
    num_events = rng.integers(6, 10)
    # Spread anomalies across full timeline so test set (last 20%) also gets some
    segment_size = (n - 200) // num_events
    chosen = []
    for i in range(num_events):
        seg_start = 100 + i * segment_size
        seg_end   = seg_start + segment_size - 50
        if seg_end > seg_start + 10:
            chosen.append(int(rng.integers(seg_start, seg_end)))

    for start in chosen:
        anomaly_type = rng.choice(["latency", "throughput", "packet_loss"])
        if anomaly_type == "latency":
            dur = int(rng.integers(4, 12))
            latency = inject_latency_spike(latency, start, dur)
            labels[start:start+dur] = 1
        elif anomaly_type == "throughput":
            dur = int(rng.integers(12, 32))
            throughput = inject_throughput_drop(throughput, start, dur)
            labels[start:start+dur] = 1
        else:
            dur = int(rng.integers(3, 8))
            packet_loss = inject_packet_loss_burst(packet_loss, start, dur)
            labels[start:start+dur] = 1

    return pd.DataFrame({
        "timestamp":        timestamps,
        "cell_id":          cell_id,
        "latency_ms":       np.round(latency, 2),
        "throughput_mbps":  np.round(throughput, 2),
        "packet_loss_pct":  np.round(np.clip(packet_loss, 0, 100), 3),
        "sinr_db":          np.round(sinr, 2),
        "connected_users":  users,
        "anomaly":          labels,
    })


def main():
    frames = []
    for cell in CELLS:
        cell_rng = np.random.default_rng(SEED + hash(cell) % 1000)
        frames.append(generate_cell_data(cell, PERIODS, cell_rng))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    out = Path(__file__).parent / "raw_kpis.csv"
    df.to_csv(out, index=False)

    total     = len(df)
    anomalies = df["anomaly"].sum()
    print(f"Generated {total:,} rows across {len(CELLS)} cells")
    print(f"Anomaly rate: {anomalies}/{total} = {anomalies/total:.2%}")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
