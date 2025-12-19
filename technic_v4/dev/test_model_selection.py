"""
Quick check of available alpha model bundles and selection order.

Run:
    python technic_v4/dev/test_model_selection.py
"""

from __future__ import annotations

from pathlib import Path

from technic_v4.engine import alpha_inference


def main() -> None:
    print("Default 5d path:", alpha_inference._XGB_MODEL_PATH_5D)  # type: ignore[attr-defined]
    print("Default 10d path:", alpha_inference._XGB_MODEL_PATH_10D)  # type: ignore[attr-defined]
    print("Regime-specific paths:")
    for k, v in alpha_inference._XGB_MODEL_PATH_5D_REGIME.items():  # type: ignore[attr-defined]
        print(f"  {k}: {v} (exists={Path(v).exists()})")
    print("Sector prefix:", alpha_inference._XGB_MODEL_PATH_5D_SECTOR_PREFIX)  # type: ignore[attr-defined]

    # Sample selections
    for reg in ["TRENDING_UP_LOW_VOL", "HIGH_VOL_UNKNOWN"]:
        path = alpha_inference.select_xgb_model_path(regime_label=reg, sector=None)
        print(f"Selected path for regime {reg}: {path}")
    for sec in ["TECHNOLOGY", "FINANCIALS"]:
        path = alpha_inference.select_xgb_model_path(regime_label=None, sector=sec)
        print(f"Selected path for sector {sec}: {path}")


if __name__ == "__main__":
    main()
