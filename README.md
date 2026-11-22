# The Lookout

How the 50 largest Nasdaq-100 stocks and the SET100 behave when the market falls: index pullbacks of 5% or more, sector behaviour on down days, each stock's downside beta, a daily watch of levels and breadth, and a stress test.

Live site: https://kaorii-ako.github.io/the-lookout/

The data refreshes on weekdays at 17:20 Bangkok time (after the Thai close) and 05:20 Bangkok time (after the US close) through GitHub Actions. Everything describes the past. It is not a forecast and not financial advice.

## How it works

- `scripts/fetch.py` downloads two years of daily prices from Yahoo Finance with yfinance into `raw/`. `names.json` holds company names.
- `scripts/analyze.py` finds pullbacks, downside betas, sector behaviour and the current state.
- `scripts/export.py` writes the JSON files the page reads into `site/data/`, and appends one row per trading day to `site/data/history.json`.
- `site/index.html` is the page. It loads d3 from jsDelivr and draws everything in the browser.
- `.github/workflows/update.yml` runs the three scripts on a schedule, commits the new data and publishes `site/` to the `gh-pages` branch.

To stop the daily updates, disable the workflow under Actions, or delete the two `schedule` lines.

`universe.json` holds the stock lists and `sectors.json` the sector groups used in the study.
