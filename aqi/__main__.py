"""Command line: python -m aqi <command>

  fetch     [--cities delhi lahore ...]   download history into data/raw/
  evaluate  [--test-start 2025-11-01]     time-split evaluation → reports/
  train                                   fit final models on all data → models/
  impact                                  how much each occasion raises PM2.5 → reports/
  forecast  CITY|all                      AQI for today and the next 2 days
"""
from __future__ import annotations

import argparse

import pandas as pd

from .cities import CITIES


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m aqi", description="South Asia AQI forecaster")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--cities", nargs="*", default=list(CITIES))
    e = sub.add_parser("evaluate")
    e.add_argument("--test-start", default=None, help="first test day; default = 12 months before the last day")
    sub.add_parser("train")
    sub.add_parser("impact")
    fc = sub.add_parser("forecast")
    fc.add_argument("city", choices=list(CITIES) + ["all"])
    args = p.parse_args(argv)

    if args.cmd == "fetch":
        from .data import build_city_history, save_history
        for key in args.cities:
            df = build_city_history(key)
            print(f"{key}: {len(df)} days → {save_history(df, key)}")
        return

    from .data import load_history

    if args.cmd == "evaluate":
        from .model import evaluate, save_report
        data = load_history()
        start = args.test_start or str((data["date"].max() - pd.DateOffset(months=12)).date())
        ev = evaluate(data, start)
        save_report(ev)
        print(ev.to_markdown())
    elif args.cmd == "train":
        from .model import save_models, train_all
        print("saved", save_models(train_all(load_history())))
    elif args.cmd == "impact":
        from .impact import event_impact
        from .model import REPORT_DIR
        table = event_impact(load_history())
        REPORT_DIR.mkdir(exist_ok=True)
        table.to_csv(REPORT_DIR / "event_impact.csv", index=False)
        print(table.to_string(index=False))
    elif args.cmd == "forecast":
        from .forecast import forecast_city
        from .model import REPORT_DIR, load_models
        models = load_models()
        keys = list(CITIES) if args.city == "all" else [args.city]
        table = pd.concat([forecast_city(k, models) for k in keys], ignore_index=True)
        print(table.to_string(index=False))
        if args.city == "all":
            REPORT_DIR.mkdir(exist_ok=True)
            issued = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
            (REPORT_DIR / "latest_forecast.md").write_text(
                f"# Latest forecast\n\nIssued {issued}.\n\n" + table.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
