import json
import os
import sys
import time

import pandas as pd
import yfinance as yf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("LOOKOUT_RAW", f"{BASE}/raw")
U = json.load(open(f"{BASE}/universe.json"))
NAMES = json.load(open(f"{BASE}/names.json"))


def save(sym, df):
    df = df.dropna(subset=["Close"])
    if len(df) < 2:
        return False
    rows = []
    for ts, r in df.iterrows():
        rows.append([ts.strftime("%Y-%m-%d"), float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]),
                     int(r["Volume"]) if pd.notna(r["Volume"]) else 0])
    out = {"symbol": sym, "name": NAMES.get(sym, sym), "currency": None, "rows": rows}
    json.dump(out, open(f"{RAW}/{sym.replace('^', '').replace('.', '_')}.json", "w"), separators=(",", ":"))
    return True


def grab(batch):
    data = yf.download(batch, period="2y", interval="1d", auto_adjust=False, group_by="ticker",
                       threads=True, progress=False)
    ok = []
    for s in batch:
        try:
            df = data[s] if isinstance(data.columns, pd.MultiIndex) else data
            if save(s, df):
                ok.append(s)
        except KeyError:
            pass
    return ok


def main():
    os.makedirs(RAW, exist_ok=True)
    syms = ["^NDX", "TDEX.BK"] + U["us"] + [t + ".BK" for t in U["set"]]
    todo = list(syms)
    for attempt in range(3):
        for i in range(0, len(todo), 25):
            batch = todo[i:i + 25]
            got = set(grab(batch))
            todo_left = [s for s in batch if s not in got]
            if todo_left:
                print("retry later:", " ".join(todo_left), file=sys.stderr)
            time.sleep(2)
        done = {s for s in syms if os.path.exists(f"{RAW}/{s.replace('^', '').replace('.', '_')}.json")}
        todo = [s for s in syms if s not in done]
        if not todo:
            break
        time.sleep(20 * (attempt + 1))
    print(f"fetched {len(syms) - len(todo)}/{len(syms)}; failed: {' '.join(todo)}")
    if "^NDX" in todo or "TDEX.BK" in todo or len(todo) > len(syms) * 0.15:
        sys.exit(1)


if __name__ == "__main__":
    main()
