import json
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("LOOKOUT_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
U = json.load(open(f"{BASE}/universe.json"))
SEC = json.load(open(f"{BASE}/sectors.json"))
TRIM = {"THAI.BK": "2025-08-04"}
RAW = os.environ.get("LOOKOUT_RAW", f"{BASE}/raw")


def load(sym):
    fn = f"{RAW}/{sym.replace('.', '_')}.json"
    if not os.path.exists(fn):
        return None, None
    d = json.load(open(fn))
    rows = [r for r in d["rows"] if r[4] is not None]
    s = pd.Series([r[4] for r in rows], index=pd.to_datetime([r[0] for r in rows]), dtype=float)
    s = s[~s.index.duplicated(keep="last")]
    if sym in TRIM:
        s = s[s.index >= TRIM[sym]]
    return s, d.get("name") or sym


def episodes(px, thresh):
    out = []
    peak_i, peak_v = 0, px.iloc[0]
    trough_i, trough_v = 0, px.iloc[0]
    in_dd = False
    for i in range(1, len(px)):
        v = px.iloc[i]
        if v >= peak_v:
            if in_dd and trough_v / peak_v - 1 <= -thresh:
                out.append((peak_i, trough_i, i))
            peak_i, peak_v = i, v
            trough_i, trough_v = i, v
            in_dd = False
        else:
            in_dd = True
            if v < trough_v:
                trough_i, trough_v = i, v
    if in_dd and trough_v / peak_v - 1 <= -thresh:
        out.append((peak_i, trough_i, None))
    return out


def rnd(x, n=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), n)


def market(key, index_sym, index_label, alt_label):
    tickers = U[key]
    sectors = SEC[key]
    syms = tickers if key == "us" else [t + ".BK" for t in tickers]
    closes, names, missing = {}, {}, []
    for t, s in zip(tickers, syms):
        px, nm = load(s)
        if px is None or len(px) < 60:
            missing.append(t)
            continue
        closes[t] = px
        names[t] = nm
    idx, _ = load(index_sym)
    df = pd.DataFrame(closes)
    cal = idx.index
    df = df.reindex(cal).ffill(limit=5)
    rets = df.pct_change(fill_method=None)
    ir = idx.pct_change()
    ew_ret = rets.mean(axis=1, skipna=True)
    ew = (1 + ew_ret.fillna(0)).cumprod() * 100

    eps = episodes(idx, 0.05)
    ep_list = []
    stock_ep = {t: [] for t in df.columns}
    for (p, tr, rec) in eps:
        pd_, td_ = cal[p], cal[tr]
        idx_dd = idx.iloc[tr] / idx.iloc[p] - 1
        rb_end = min(tr + 20, len(cal) - 1)
        idx_rb = idx.iloc[rb_end] / idx.iloc[tr] - 1
        ew_dd = ew.iloc[tr] / ew.iloc[p] - 1
        rows = []
        for t in df.columns:
            a, b, c = df[t].iloc[p], df[t].iloc[tr], df[t].iloc[rb_end]
            if np.isnan(a) or np.isnan(b):
                continue
            sdd = b / a - 1
            window = df[t].iloc[p:tr + 1]
            worst = window.min() / window.iloc[0] - 1
            rb = c / b - 1 if not np.isnan(c) else np.nan
            rec_days = None
            after = df[t].iloc[tr:]
            hit = np.where(after.values >= a)[0]
            if len(hit):
                rec_days = int(hit[0])
            r = dict(t=t, sector=sectors.get(t), ret=sdd, worst=worst, capture=sdd / idx_dd,
                     rs=sdd - idx_dd, rebound=rb, rb_rs=rb - idx_rb if not np.isnan(rb) else np.nan,
                     rec_days=rec_days)
            rows.append(r)
            stock_ep[t].append(dict(ep=len(ep_list), ret=sdd, capture=sdd / idx_dd, rs=sdd - idx_dd,
                                    rebound=rb, rec_days=rec_days))
        er = pd.DataFrame(rows)
        sec_tbl = er.groupby("sector").agg(ret=("ret", "mean"), rebound=("rebound", "mean"), n=("t", "count")).sort_values("ret")
        er = er.sort_values("ret")
        ep_list.append(dict(
            peak=str(pd_.date()), trough=str(td_.date()),
            recovered=str(cal[rec].date()) if rec is not None else None,
            days_down=int(tr - p), days_to_recover=int(rec - tr) if rec is not None else None,
            index_dd=rnd(idx_dd), ew_dd=rnd(ew_dd), index_rebound_20d=rnd(idx_rb),
            breadth_worse_than_index=rnd(float((er["ret"] < idx_dd).mean())),
            median_stock=rnd(float(er["ret"].median())),
            best=[dict(t=r.t, ret=rnd(r.ret)) for r in er.tail(5).iloc[::-1].itertuples()],
            worst=[dict(t=r.t, ret=rnd(r.ret)) for r in er.head(5).itertuples()],
            rebound_leaders=[dict(t=r.t, rb=rnd(r.rebound)) for r in er.sort_values("rebound", ascending=False).head(5).itertuples()],
            sectors=[dict(sector=s, ret=rnd(r.ret), rebound=rnd(r.rebound), n=int(r.n)) for s, r in sec_tbl.iterrows()],
        ))

    down = ir < 0
    up = ir > 0
    stocks = []
    for t in df.columns:
        sr = rets[t]
        m = sr.notna() & ir.notna()
        md, mu = m & down, m & up
        def beta(mask):
            if mask.sum() < 20:
                return np.nan
            x, y = ir[mask].values, sr[mask].values
            return np.cov(x, y)[0, 1] / np.var(x, ddof=1)
        bd, bu, ball = beta(md), beta(mu), beta(m)
        dcap = sr[md].mean() / ir[md].mean() if md.sum() else np.nan
        ucap = sr[mu].mean() / ir[mu].mean() if mu.sum() else np.nan
        corr = sr[m].corr(ir[m])
        px = df[t].dropna()
        runmax = px.cummax()
        dd = px / runmax - 1
        own = episodes(px, 0.10)
        own_list = []
        for (p, tr, rec) in own:
            own_list.append(dict(peak=str(px.index[p].date()), trough=str(px.index[tr].date()),
                                 depth=rnd(px.iloc[tr] / px.iloc[p] - 1),
                                 days_down=int(tr - p),
                                 days_to_recover=int(rec - tr) if rec is not None else None))
        rec_list = [e["days_to_recover"] for e in own_list if e["days_to_recover"] is not None]
        last = px.iloc[-1]
        last_date = px.index[-1]
        ma50 = px.rolling(50).mean().iloc[-1]
        ma200 = px.rolling(200).mean().iloc[-1] if len(px) >= 200 else np.nan
        hi52 = px.iloc[-252:].max()
        i_last = idx.loc[:last_date]
        def rs(n):
            if len(px) <= n:
                return np.nan
            return (px.iloc[-1] / px.iloc[-1 - n] - 1) - (i_last.iloc[-1] / i_last.iloc[-1 - n] - 1)
        def r(n):
            return px.iloc[-1] / px.iloc[-1 - n] - 1 if len(px) > n else np.nan
        eps_t = stock_ep[t]
        caps = [e["capture"] for e in eps_t]
        rss = [e["rs"] for e in eps_t]
        rbs = [e["rebound"] for e in eps_t if not np.isnan(e["rebound"])]
        held = sum(1 for e in eps_t if e["rs"] > 0)
        stocks.append(dict(
            t=t, name=names[t], sector=sectors.get(t), start=str(px.index[0].date()), last_date=str(last_date.date()),
            n_days=int(len(px)), beta=rnd(ball, 3), beta_down=rnd(bd, 3), beta_up=rnd(bu, 3),
            down_capture=rnd(dcap, 3), up_capture=rnd(ucap, 3), corr=rnd(corr, 3),
            vol=rnd(sr.std() * np.sqrt(252), 3),
            max_dd=rnd(dd.min()), cur_dd=rnd(dd.iloc[-1]), from_52w_high=rnd(last / hi52 - 1),
            above_50=bool(last > ma50) if not np.isnan(ma50) else None,
            above_200=bool(last > ma200) if not np.isnan(ma200) else None,
            r1=rnd(r(1)), r5=rnd(r(5)), r20=rnd(r(20)), rs20=rnd(rs(20)), rs60=rnd(rs(60)),
            own_dd_count=len(own_list), own_dd_median_recover=rnd(float(np.median(rec_list)), 1) if rec_list else None,
            own_dd=own_list,
            ep_capture_med=rnd(float(np.median(caps)), 3) if caps else None,
            ep_capture_q1=rnd(float(np.percentile(caps, 25)), 3) if caps else None,
            ep_capture_q3=rnd(float(np.percentile(caps, 75)), 3) if caps else None,
            ep_rs_med=rnd(float(np.median(rss))) if rss else None,
            ep_rebound_med=rnd(float(np.median(rbs))) if rbs else None,
            ep_held=held, ep_n=len(eps_t),
            ep=[dict(ep=e["ep"], ret=rnd(e["ret"]), rs=rnd(e["rs"]), rebound=rnd(e["rebound"]), rec_days=e["rec_days"]) for e in eps_t],
        ))

    sec_out = []
    for s in sorted(set(sectors.values())):
        mem = [t for t in df.columns if sectors.get(t) == s]
        if not mem:
            continue
        sret = rets[mem].mean(axis=1, skipna=True)
        sidx = (1 + sret.fillna(0)).cumprod() * 100
        sdd = sidx / sidx.cummax() - 1
        s_eps = episodes(sidx, 0.10)
        s_eps_out = [dict(peak=str(cal[p].date()), trough=str(cal[tr].date()),
                          depth=rnd(sidx.iloc[tr] / sidx.iloc[p] - 1),
                          index_same_window=rnd(idx.iloc[tr] / idx.iloc[p] - 1),
                          days_to_recover=int(rec - tr) if rec is not None else None) for (p, tr, rec) in s_eps]
        m = sret.notna() & ir.notna()
        md = m & down
        bd = np.cov(ir[md], sret[md])[0, 1] / np.var(ir[md], ddof=1)
        def avg_corr(mask):
            if len(mem) < 2:
                return None
            c = rets.loc[mask, mem].corr().values
            iu = np.triu_indices_from(c, 1)
            return rnd(float(np.nanmean(c[iu])), 3)
        worst_days = ir.nsmallest(int(len(ir) * 0.1)).index
        wd_mask = pd.Series(ir.index.isin(worst_days), index=ir.index)
        def rs(n):
            return (sidx.iloc[-1] / sidx.iloc[-1 - n] - 1) - (idx.iloc[-1] / idx.iloc[-1 - n] - 1)
        sec_out.append(dict(
            sector=s, n=len(mem), members=mem, beta_down=rnd(bd, 3),
            max_dd=rnd(sdd.min()), cur_dd=rnd(sdd.iloc[-1]),
            corr_all=avg_corr(m), corr_down=avg_corr(wd_mask & m),
            avg_on_worst_days=rnd(float(sret[wd_mask].mean())), index_on_worst_days=rnd(float(ir[wd_mask].mean())),
            rs20=rnd(rs(20)), rs60=rnd(rs(60)),
            r1=rnd(sidx.iloc[-1] / sidx.iloc[-2] - 1), r5=rnd(sidx.iloc[-1] / sidx.iloc[-6] - 1),
            index_r1=rnd(idx.iloc[-1] / idx.iloc[-2] - 1), index_r5=rnd(idx.iloc[-1] / idx.iloc[-6] - 1),
            episodes=s_eps_out,
            series=[rnd(v, 2) for v in sidx.values[::2]],
        ))

    idd = idx / idx.cummax() - 1
    ma50 = idx.rolling(50).mean()
    ma200 = idx.rolling(200).mean()
    above50 = [s for s in stocks if s["above_50"]]
    def breadth_at(lag):
        a = b = c = 0
        for t in df.columns:
            px = df[t].iloc[: len(df) - lag].dropna()
            if len(px) < 60:
                continue
            c += 1
            if px.iloc[-1] > px.iloc[-50:].mean():
                a += 1
            if px.iloc[-1] <= px.iloc[-252:].max() * 0.8:
                b += 1
        return (a / c, b / c) if c else (None, None)
    b5 = breadth_at(5)
    peak_v = float(idx.cummax().iloc[-1])
    state = dict(
        last_date=str(cal[-1].date()), index_last=rnd(idx.iloc[-1], 2), index_dd=rnd(idd.iloc[-1]),
        index_peak_date=str(idx.idxmax().date()), index_r5=rnd(idx.iloc[-1] / idx.iloc[-6] - 1),
        index_r20=rnd(idx.iloc[-1] / idx.iloc[-21] - 1),
        index_above_50=bool(idx.iloc[-1] > ma50.iloc[-1]), index_above_200=bool(idx.iloc[-1] > ma200.iloc[-1]),
        ew_dd=rnd((ew / ew.cummax() - 1).iloc[-1]),
        pct_above_50=rnd(len(above50) / len(stocks)),
        pct_above_200=rnd(sum(1 for s in stocks if s["above_200"]) / sum(1 for s in stocks if s["above_200"] is not None)),
        pct_10_off_high=rnd(sum(1 for s in stocks if s["from_52w_high"] is not None and s["from_52w_high"] <= -0.10) / len(stocks)),
        pct_20_off_high=rnd(sum(1 for s in stocks if s["from_52w_high"] is not None and s["from_52w_high"] <= -0.20) / len(stocks)),
        pct_above_50_5d=rnd(b5[0]), pct_20_off_high_5d=rnd(b5[1]),
        index_r1=rnd(idx.iloc[-1] / idx.iloc[-2] - 1),
        index_peak=rnd(peak_v, 2), lv10=rnd(peak_v * 0.9, 2), lv20=rnd(peak_v * 0.8, 2), lv25=rnd(peak_v * 0.75, 2),
        ma50=rnd(ma50.iloc[-1], 2), ma200=rnd(ma200.iloc[-1], 2),
        recent=[dict(date=str(cal[i].date()), index=rnd(idx.iloc[i], 2), dd=rnd(idd.iloc[i]), r1=rnd(idx.iloc[i] / idx.iloc[i - 1] - 1))
                for i in range(len(cal) - 10, len(cal))],
    )
    ser = dict(dates=[str(d.date()) for d in cal[::2]], index=[rnd(v, 2) for v in idx.values[::2]],
               ew=[rnd(v, 2) for v in ew.values[::2]], dd=[rnd(v) for v in idd.values[::2]])
    return dict(key=key, index_label=index_label, alt_label=alt_label, missing=missing,
                n=len(stocks), episodes=ep_list, stocks=stocks, sectors=sec_out, state=state, series=ser)


out = dict(
    generated=pd.Timestamp.now(tz="Asia/Bangkok").strftime("%Y-%m-%d %H:%M"),
    us=market("us", "NDX", "Nasdaq-100 (NDX)", "Equal-weight of the 49 stocks"),
    th=market("set", "TDEX.BK", "SET50 (TDEX fund)", "Equal-weight SET100"),
)
os.makedirs(f"{BASE}/out", exist_ok=True)
json.dump(out, open(f"{BASE}/out/analysis.json", "w"), separators=(",", ":"))
print("ok", out["us"]["n"], out["th"]["n"], out["us"]["missing"], out["th"]["missing"])
