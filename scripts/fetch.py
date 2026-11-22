import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("LOOKOUT_RAW", f"{BASE}/raw")
U = json.load(open(f"{BASE}/universe.json"))
HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def get(sym):
    last = None
    for attempt in range(4):
        host = HOSTS[attempt % 2]
        url = f"https://{host}/v8/finance/chart/{sym}?range=2y&interval=1d"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 + attempt * 3)
    raise last


def parse(sym, js):
    res = js["chart"]["result"][0]
    meta = res["meta"]
    off = meta.get("gmtoffset", 0)
    q = res["indicators"]["quote"][0]
    rows = []
    for i, ts in enumerate(res.get("timestamp") or []):
        c = q["close"][i]
        if c is None:
            continue
        d = (datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(seconds=off)).strftime("%Y-%m-%d")
        rows.append([d, q["open"][i], q["high"][i], q["low"][i], c, q["volume"][i]])
    dedup = {}
    for r in rows:
        dedup[r[0]] = r
    return {"symbol": sym, "name": meta.get("longName") or meta.get("shortName") or sym,
            "currency": meta.get("currency"), "rows": [dedup[k] for k in sorted(dedup)]}


def one(sym):
    try:
        d = parse(sym, get(sym))
        if len(d["rows"]) < 2:
            return sym, False
        json.dump(d, open(f"{RAW}/{sym.replace('^', '').replace('.', '_')}.json", "w"), separators=(",", ":"))
        return sym, True
    except Exception as e:
        print("fail", sym, e, file=sys.stderr)
        return sym, False


def main():
    os.makedirs(RAW, exist_ok=True)
    syms = ["^NDX", "TDEX.BK"] + U["us"] + [t + ".BK" for t in U["set"]]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(one, syms))
    bad = [s for s, ok in results if not ok]
    print(f"fetched {len(syms) - len(bad)}/{len(syms)}; failed: {' '.join(bad)}")
    if "^NDX" in bad or "TDEX.BK" in bad or len(bad) > len(syms) * 0.15:
        sys.exit(1)


if __name__ == "__main__":
    main()
