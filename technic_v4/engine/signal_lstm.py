"""Sequential pattern LSTM for signal forecasting."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False
    torch = None
    nn = None


if HAVE_TORCH:

    class SignalLSTM(nn.Module):
        """Simple LSTM head for sequence-to-one regression/classification."""

        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.linear(lstm_out[:, -1, :])
            return output

else:  # pragma: no cover

    class SignalLSTM:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to use SignalLSTM.")

        def forward(self, *args, **kwargs):
            raise ImportError("torch is required to use SignalLSTM.")


__all__ = ["SignalLSTM"]
