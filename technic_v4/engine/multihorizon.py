"""
Multi-horizon forecasting (Temporal Fusion Transformer) scaffolding.

Designed to produce multi-horizon forecasts (e.g., 1d/5d/20d) that can later be
used as features for alpha models or standalone signals. This module does not
run heavy training on import; all heavy work is behind functions.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from technic_v4.data_layer.price_layer import get_stock_history_df


class PriceForecastDataModule(pl.LightningDataModule):
    """
    Builds TimeSeriesDataSet for multi-symbol OHLCV-based forecasting.
    Expects df_prices with columns:
      - Symbol
      - time_idx (int, increasing across dates)
      - target (e.g., next-day log return or next close)
      - optional covariates (Close, Volume, etc.)
    """

    def __init__(
        self,
        df_prices: pd.DataFrame,
        max_encoder_length: int = 60,
        max_prediction_length: int = 5,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.df_prices = df_prices
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.training: Optional[TimeSeriesDataSet] = None
        self.validation: Optional[TimeSeriesDataSet] = None

    def setup(self, stage: Optional[str] = None) -> None:
        df = self.df_prices.copy()
        self.training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["Symbol"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_unknown_reals=["target", "Close", "Volume"],
            time_varying_known_reals=[],
            target_normalizer=GroupNormalizer(groups=["Symbol"]),
        )
        self.validation = self.training.split_before(0.8)

    def train_dataloader(self):
        return self.training.to_dataloader(train=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return self.validation.to_dataloader(train=False, batch_size=self.batch_size)


def build_price_dataset_from_market_cache(symbols: List[str], days: int = 260) -> pd.DataFrame:
    """
    Fetch daily history for symbols and build a panel with time_idx.
    """
    frames = []
    for sym in symbols:
        try:
            hist = get_stock_history_df(symbol=sym, days=days, use_intraday=False)
        except Exception:
            continue
        if hist is None or hist.empty or "Close" not in hist or "Volume" not in hist:
            continue
        df = hist.copy()
        df["Symbol"] = sym
        frames.append(df[["Symbol", "Close", "Volume"]])
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=0)
    panel = panel.reset_index().rename(columns={"index": "date"})
    panel = panel.sort_values(["date", "Symbol"])
    # global time_idx across all dates
    unique_dates = sorted(panel["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    panel["time_idx"] = panel["date"].map(date_to_idx)
    # target = next-day log return
    panel["target"] = panel.groupby("Symbol")["Close"].shift(-1) / panel["Close"]
    panel["target"] = np.log(panel["target"])
    panel = panel.dropna(subset=["target"])
    return panel


def train_tft(
    df_prices: pd.DataFrame,
    max_epochs: int = 10,
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu",
) -> TemporalFusionTransformer:
    """
    Train TFT on price panel; returns the trained model.
    """
    dm = PriceForecastDataModule(df_prices)
    dm.setup()
    tft = TemporalFusionTransformer.from_dataset(
        dm.training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=1,
        loss=QuantileLoss(),
        log_interval=50,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(tft, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    return tft


def predict_tft(model: TemporalFusionTransformer, dataset: TimeSeriesDataSet, n_future_steps: int = 5) -> pd.DataFrame:
    """
    Generate multi-horizon forecasts from a trained TFT.
    Returns DataFrame: Symbol, time_idx, horizon, forecast_return.
    """
    if model is None or dataset is None:
        return pd.DataFrame()
    loader = dataset.to_dataloader(train=False, batch_size=64)
    preds, index = model.predict(loader, return_index=True, mode="prediction")
    records = []
    symbols = index["Symbol"]
    time_idxs = index["time_idx"]
    for p, sym, tidx in zip(preds, symbols, time_idxs):
        for h in range(min(len(p), n_future_steps)):
            records.append(
                {"Symbol": sym, "time_idx": int(tidx), "horizon": h + 1, "forecast_return": float(p[h])}
            )
    return pd.DataFrame(records)


def train_and_save_tft_model(symbols: List[str], out_path: str = "models/tft_price_forecast.ckpt") -> None:
    """
    Build dataset from market cache, train TFT, and save checkpoint.
    """
    df = build_price_dataset_from_market_cache(symbols, days=260)
    if df.empty:
        raise SystemExit("No data to train TFT.")
    model = train_tft(df, max_epochs=5)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer()  # dummy trainer to save
    trainer.save_checkpoint(str(out_path), weights_only=True, model=model)


def load_tft_model(path: str = "models/tft_price_forecast.ckpt") -> Optional[TemporalFusionTransformer]:
    """
    Load TFT checkpoint if available.
    """
    from pytorch_lightning import LightningModule

    ckpt = Path(path)
    if not ckpt.exists():
        return None
    try:
        # Lightning will restore model class from checkpoint; ensure consistent import paths.
        model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
        return model
    except Exception:
        return None


def build_tft_features_for_symbols(symbols: List[str], n_future_steps: int = 3) -> pd.DataFrame:
    """
    Generate TFT forecast features for symbols. Returns DataFrame keyed by Symbol with columns:
      tft_forecast_h1, tft_forecast_h2, ..., up to n_future_steps.
    """
    model = load_tft_model()
    if model is None:
        return pd.DataFrame()
    df = build_price_dataset_from_market_cache(symbols, days=260)
    if df.empty:
        return pd.DataFrame()
    dm = PriceForecastDataModule(df)
    dm.setup()
    preds = predict_tft(model, dm.validation, n_future_steps=n_future_steps)
    if preds.empty:
        return pd.DataFrame()
    features = preds.pivot_table(index="Symbol", columns="horizon", values="forecast_return", aggfunc="mean")
    features = features.add_prefix("tft_forecast_h")
    features = features.reset_index().rename_axis(None, axis=1)
    return features


# TODO:
# - Wire TFT forecasts into scan outputs or feature_engine joins for alpha models.
# - Add richer covariates (regime flags, technicals) for better forecasts.
