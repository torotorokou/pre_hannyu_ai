import argparse
import os
from typing import Optional, Dict, Any

import pandas as pd

from .stacking_core import run_stacking, PRESETS


def read_csv_safely(path: str, date_col: Optional[str]):
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def main():
    parser = argparse.ArgumentParser(description="Run stacking regression experiments with presets A-D")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--date-col", default=None, help="Optional date column name")
    parser.add_argument("--preset", choices=["A", "B", "C", "D", "ALL"], default="A")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ts-cv", action="store_true", help="Use time series split CV")
    parser.add_argument("--robust-meta", action="store_true", help="Use HuberRegressor as meta model")
    parser.add_argument("--output", default=None, help="Output directory to save artifacts")

    args = parser.parse_args()

    df = read_csv_safely(args.csv, args.date_col)
    out_dir = args.output
    if out_dir is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_dir = os.path.join("outputs", f"stack_{args.preset}_{base}")

    if args.preset == "ALL":
        summaries: Dict[str, Any] = {}
        best_key = None
        best_rmse = float("inf")
        for p in PRESETS.keys():
            out_p = os.path.join(out_dir, p)
            res = run_stacking(
                df=df,
                target=args.target,
                date_col=args.date_col,
                preset=p,
                n_splits=args.n_splits,
                time_series_cv=args.ts_cv,
                robust_meta=args.robust_meta,
                output_dir=out_p,
            )
            rmse = float(res["meta_metrics"]["rmse"]) if "meta_metrics" in res else float("inf")
            summaries[p] = {
                "meta": res.get("meta_metrics", {}),
                "base": res.get("base_metrics", {}),
                "out_dir": out_p,
            }
            if rmse < best_rmse:
                best_rmse, best_key = rmse, p

        os.makedirs(out_dir, exist_ok=True)
        pd.Series({k: v["meta"].get("rmse", None) for k, v in summaries.items()}).to_csv(
            os.path.join(out_dir, "rmse_summary.csv"), header=["rmse"], index_label="preset"
        )
        print("=== ALL presets finished ===")
        for p, info in summaries.items():
            print(f"Preset {p}: meta={info['meta']}, saved at {info['out_dir']}")
        print(f"BEST: {best_key} (rmse={best_rmse:.6f}) -> {os.path.join(out_dir, best_key)}")
    else:
        results = run_stacking(
            df=df,
            target=args.target,
            date_col=args.date_col,
            preset=args.preset,
            n_splits=args.n_splits,
            time_series_cv=args.ts_cv,
            robust_meta=args.robust_meta,
            output_dir=out_dir,
        )

        # Compact printout
        print("=== Stacking finished ===")
        print(f"Preset: {results['preset']}")
        print("Base metrics:")
        for k, v in results["base_metrics"].items():
            print(f"  - {k}: {v}")
        print("Meta metrics:")
        print(results["meta_metrics"])
        print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
