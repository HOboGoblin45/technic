from __future__ import annotations

"""
Nightly maintenance orchestration for Technic v5.

Steps:
 0) Parse CLI flags
 1) Refresh data cache (prices/fundamentals)
 2) Train alpha model (LGBM)
 3) Train TFT (optional)
 4) Export ONNX (optional)
 5) Update scoreboard metrics (optional)
 6) Log summary
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def _log(msg: str) -> None:
    timestamped = f"[{datetime.utcnow().isoformat()}] {msg}"
    print(timestamped)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "nightly.log", "a", encoding="utf-8") as fh:
        fh.write(timestamped + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Technic nightly maintenance")
    parser.add_argument("--no-alpha-train", action="store_true", help="Skip LGBM alpha training")
    parser.add_argument("--no-tft-train", action="store_true", help="Skip TFT training")
    parser.add_argument("--no-onnx-export", action="store_true", help="Skip ONNX export")
    parser.add_argument("--no-scoreboard", action="store_true", help="Skip scoreboard metrics")
    parser.add_argument("--universe-limit", type=int, default=None, help="Limit symbols for training")
    parser.add_argument("--strategy-profile", type=str, default=None, help="Optional strategy profile name to log/use")
    return parser.parse_args(argv)


def refresh_data(universe_limit: int | None = None) -> None:
    try:
        from technic_v4.data_layer import bulk_daily
        from technic_v4.universe_loader import load_universe

        universe = load_universe()
        symbols = [u.symbol for u in universe][:universe_limit] if universe_limit else [u.symbol for u in universe]
        if hasattr(bulk_daily, "refresh_bulk_prices"):
            bulk_daily.refresh_bulk_prices(symbols)
            _log(f"Data refresh complete for {len(symbols)} symbols.")
        else:
            _log("Data refresh skipped: bulk_daily.refresh_bulk_prices not available.")
    except Exception as exc:
        _log(f"Data refresh failed: {exc}")


def train_alpha() -> None:
    try:
        from technic_v4.engine.alpha_models import train_lgbm_alpha

        result = train_lgbm_alpha.main()
        if isinstance(result, dict):
            _log(f"Alpha training complete. Version={result.get('version')} Metrics={result.get('metrics')}")
        else:
            _log("Alpha training complete.")
    except BaseException as exc:
        _log(f"Alpha training failed: {exc}")


def train_meta_winprob() -> None:
    """
    Retrain the win_prob_10d meta model so nightly scans have fresh
    probabilities. Uses the dev backtest training module.
    """
    try:
        from technic_v4.dev.backtest import train_meta_model

        result = train_meta_model.main()
        if isinstance(result, dict):
            _log(
                "Meta win_prob_10d training complete. "
                f"Rows={result.get('train_rows')} Features={result.get('features_used')} "
                f"AUC={result.get('auc_test')}"
            )
        else:
            _log("Meta win_prob_10d training complete.")
    except BaseException as exc:
        _log(f"Meta win_prob_10d training failed: {exc}")


def train_tft(universe_limit: int | None = None) -> None:
    try:
        from technic_v4.engine.multihorizon import train_and_save_tft_model
        from technic_v4.universe_loader import load_universe

        universe = load_universe()
        symbols = [u.symbol for u in universe][:universe_limit] if universe_limit else [u.symbol for u in universe]
        train_and_save_tft_model(symbols)
        _log("TFT training complete.")
    except Exception as exc:
        _log(f"TFT training failed: {exc}")


def export_onnx() -> None:
    try:
        from technic_v4 import model_registry
        from technic_v4.engine.alpha_models import lgbm_alpha
        from technic_v4.engine import inference_engine

        reg_entry = model_registry.get_latest_model("alpha_lgbm_v1")
        pickle_path = reg_entry["path_pickle"] if reg_entry else "models/alpha/lgbm_v1.pkl"
        onnx_target = reg_entry.get("path_onnx") if reg_entry else None
        feature_names = reg_entry.get("feature_names", []) if reg_entry else []
        model = lgbm_alpha.LGBMAlphaModel.load(pickle_path)
        out_path = onnx_target or "models/alpha/lgbm_v1.onnx"
        inference_engine.export_lgbm_to_onnx(model.model, feature_names, out_path)
        if reg_entry:
            model_registry.register_model(
                model_name="alpha_lgbm_v1",
                version=reg_entry.get("version", "latest"),
                metrics=reg_entry.get("metrics", {}),
                path_pickle=pickle_path,
                path_onnx=out_path,
                feature_names=feature_names,
            )
        _log(f"ONNX export complete to {out_path}.")
    except Exception as exc:
        _log(f"ONNX export failed: {exc}")


def update_scoreboard() -> None:
    try:
        from technic_v4.evaluation import scoreboard

        metrics = scoreboard.compute_history_metrics()
        _log(f"Scoreboard metrics: {metrics}")
    except Exception as exc:
        _log(f"Scoreboard update failed: {exc}")


def main(argv=None):
    args = parse_args(argv)
    _log("Nightly maintenance started.")
    if args.strategy_profile:
        os.environ["TECHNIC_STRATEGY_PROFILE"] = args.strategy_profile
        _log(f"Using strategy profile hint: {args.strategy_profile}")

    refresh_data(args.universe_limit)

    if not args.no_alpha_train:
        train_alpha()
    else:
        _log("Skipping alpha training (flag).")

    # Refresh the meta win_prob_10d model on the latest replay/training data.
    train_meta_winprob()

    if not args.no_tft_train:
        train_tft(args.universe_limit)
    else:
        _log("Skipping TFT training (flag).")

    if not args.no_onnx_export:
        export_onnx()
    else:
        _log("Skipping ONNX export (flag).")

    if not args.no_scoreboard:
        update_scoreboard()
    else:
        _log("Skipping scoreboard (flag).")

    _log("Nightly maintenance finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
