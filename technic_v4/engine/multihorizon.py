"""
Skeleton for multi-horizon forecasting with PyTorch Forecasting (Temporal Fusion Transformer).

This is decoupled from scanner_core for now; intended as a blueprint to
produce multi-horizon alpha signals (e.g., 1d/5d/20d forecasts) later.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


class PriceTimeSeriesDataModule(pl.LightningDataModule):
    """
    Wraps construction of a TimeSeriesDataSet for multi-symbol OHLCV data.
    Expects input DataFrame with columns:
      - symbol
      - time_idx (int, increasing)
      - target (float; e.g., future return or price)
      - optional covariates (known/observed/static)

    Note: This is a skeleton; populate covariates and preprocessing as needed.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        max_encoder_length: int = 60,
        max_prediction_length: int = 5,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.data = data
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        df = self.data.copy()
        self.training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["symbol"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_unknown_reals=["target"],
            target_normalizer=GroupNormalizer(groups=["symbol"]),
        )
        self.validation = self.training.split_before(0.8)

    def train_dataloader(self):
        return self.training.to_dataloader(train=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return self.validation.to_dataloader(train=False, batch_size=self.batch_size)


def train_tft_model(
    data_module: PriceTimeSeriesDataModule,
    max_epochs: int = 10,
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu",
) -> TemporalFusionTransformer:
    """
    Train a Temporal Fusion Transformer on the provided data module.
    """
    data_module.setup()
    tft = TemporalFusionTransformer.from_dataset(
        data_module.training,
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
    trainer.fit(
        tft,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )
    return tft


def predict_tft(model: TemporalFusionTransformer, dataset: TimeSeriesDataSet, n_future_steps: int = 5) -> pd.DataFrame:
    """
    Run multi-horizon prediction for each symbol. Returns DataFrame with columns:
      symbol, time_idx, horizon, prediction
    """
    if model is None or dataset is None:
        return pd.DataFrame()
    new_raw = dataset.to_dataloader(train=False, batch_size=64)
    preds, index = model.predict(new_raw, return_index=True, mode="prediction")
    records = []
    for p, idx in zip(preds, index["time_idx"]):
        for h in range(min(len(p), n_future_steps)):
            records.append({"time_idx": idx, "horizon": h + 1, "prediction": float(p[h])})
    return pd.DataFrame(records)


# TODOs for future integration:
# - Hook data ingest from MarketCache/Polygon to build multi-symbol time_idx panel.
# - Engineer covariates: technical factors, regime flags, macro indicators.
# - Store TFT outputs as multi-horizon alpha features to blend into scan ranking.
