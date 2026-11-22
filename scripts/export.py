import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{BASE}/site/data"
d = json.load(open(f"{BASE}/out/analysis.json"))

stocks, episodes, epstock, sectors, sectorep, series, state = [], [], [], [], [], [], []
for m in ("us", "th"):
    mk = d[m]
    for s in mk["stocks"]:
        row = {k: v for k, v in s.items() if k not in ("own_dd", "ep")}
        row["m"] = m
        stocks.append(row)
        for e in s["ep"]:
            epstock.append(dict(m=m, t=s["t"], sector=s["sector"], **e))
    for i, e in enumerate(mk["episodes"]):
        episodes.append(dict(
            m=m, ep=i, peak=e["peak"], trough=e["trough"], recovered=e["recovered"],
            days_down=e["days_down"], days_to_recover=e["days_to_recover"],
            index_dd=e["index_dd"], ew_dd=e["ew_dd"], median_stock=e["median_stock"],
            share_worse=e["breadth_worse_than_index"], index_rebound_20d=e["index_rebound_20d"],
            held_best=", ".join(f"{x['t']} {x['ret']*100:+.0f}%" for x in e["best"][:3]),
            fell_most=", ".join(f"{x['t']} {x['ret']*100:+.0f}%" for x in e["worst"][:3]),
            bounced_first=", ".join(f"{x['t']} {x['rb']*100:+.0f}%" for x in e["rebound_leaders"][:3]) if e["recovered"] or e["index_rebound_20d"] else "",
        ))
        for s in e["sectors"]:
            sectorep.append(dict(m=m, ep=i, sector=s["sector"], ret=s["ret"], rebound=s["rebound"]))
    for s in mk["sectors"]:
        sectors.append({k: v for k, v in s.items() if k not in ("episodes", "series", "members")} |
                       dict(m=m, episodes_10pct=len(s["episodes"]), members=" ".join(s["members"])))
    sr = mk["series"]
    st = mk["state"]
    for dt, ix, ew, dd in zip(sr["dates"], sr["index"], sr["ew"], sr["dd"]):
        series.append(dict(m=m, date=dt, index=ix, ew=ew, dd=dd))
    if sr["dates"][-1] != st["last_date"]:
        series.append(dict(m=m, date=st["last_date"], index=st["index_last"], ew=None, dd=st["index_dd"]))
    s2 = dict(st)
    s2.update(m=m, index_label=mk["index_label"], n=mk["n"], excluded=", ".join(mk["missing"]))
    state.append(s2)

os.makedirs(OUT, exist_ok=True)
for name, rows in [("stocks", stocks), ("episodes", episodes), ("epstock", epstock), ("sectors", sectors),
                   ("sectorep", sectorep), ("series", series), ("state", state)]:
    json.dump(rows, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"), allow_nan=False)

hp = f"{OUT}/history.json"
hist = json.load(open(hp)) if os.path.exists(hp) else []
for s in state:
    have = {h["date"] for h in hist if h["m"] == s["m"]}
    for r in s.get("recent", []):
        if r["date"] not in have and r["date"] != s["last_date"]:
            hist.append(dict(m=s["m"], date=r["date"], index=r["index"], dd=r["dd"], r1=r["r1"], pct_above_50=None, pct_20_off_high=None))
    row = dict(m=s["m"], date=s["last_date"], index=s["index_last"], dd=s["index_dd"], r1=s["index_r1"],
               pct_above_50=s["pct_above_50"], pct_20_off_high=s["pct_20_off_high"])
    hist = [h for h in hist if not (h["m"] == row["m"] and h["date"] == row["date"])]
    hist.append(row)
hist.sort(key=lambda h: (h["m"], h["date"]))
json.dump(hist, open(hp, "w"), separators=(",", ":"), allow_nan=False)
json.dump({"generated": d["generated"]}, open(f"{OUT}/meta.json", "w"))
print("exported", len(stocks), "stocks;", d["generated"])
