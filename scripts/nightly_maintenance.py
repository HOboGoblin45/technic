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
import sys
from datetime import datetime
from pathlib import Path


def _log(msg: str) -> None:
    print(msg)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "nightly.log").write_text(
        f"[{datetime.utcnow().isoformat()}] {msg}\n", encoding="utf-8", append=True
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Technic nightly maintenance")
    parser.add_argument("--no-alpha-train", action="store_true", help="Skip LGBM alpha training")
    parser.add_argument("--no-tft-train", action="store_true", help="Skip TFT training")
    parser.add_argument("--no-onnx-export", action="store_true", help="Skip ONNX export")
    parser.add_argument("--no-scoreboard", action="store_true", help="Skip scoreboard metrics")
    parser.add_argument("--universe-limit", type=int, default=None, help="Limit symbols for training")
    return parser.parse_args(argv)


def refresh_data(universe_limit: int | None = None) -> None:
    try:
        from technic_v4.data_layer import bulk_daily
        from technic_v4.universe_loader import load_universe

        universe = load_universe()
        symbols = [u.symbol for u in universe][:universe_limit] if universe_limit else [u.symbol for u in universe]
        bulk_daily.refresh_bulk_prices(symbols)
        _log(f"Data refresh complete for {len(symbols)} symbols.")
    except Exception as exc:
        _log(f"Data refresh failed: {exc}")


def train_alpha() -> None:
    try:
        from technic_v4.engine.alpha_models import train_lgbm_alpha

        train_lgbm_alpha.main()
        _log("Alpha training complete.")
    except Exception as exc:
        _log(f"Alpha training failed: {exc}")


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
        from technic_v4.engine.alpha_models import lgbm_alpha
        from technic_v4.engine import inference_engine
        from technic_v4.engine.alpha_models import train_lgbm_alpha

        model = lgbm_alpha.LGBMAlphaModel.load("models/alpha/lgbm_v1.pkl")
        feature_names = []  # TODO: persist feature list alongside model
        inference_engine.export_lgbm_to_onnx(model.model, feature_names, "models/alpha/lgbm_v1.onnx")
        _log("ONNX export complete.")
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

    refresh_data(args.universe_limit)

    if not args.no_alpha_train:
        train_alpha()
    else:
        _log("Skipping alpha training (flag).")

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
