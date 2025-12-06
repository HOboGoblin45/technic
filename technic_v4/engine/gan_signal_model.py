"""Causal GAN signal generator (skeleton)."""

from __future__ import annotations

try:
    from torch import nn

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False
    nn = None


if HAVE_TORCH:

    class CausalGenerator(nn.Module):
        """Simple GAN generator skeleton for synthetic trading patterns."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(100, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.fc(x)

else:  # pragma: no cover

    class CausalGenerator:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to use CausalGenerator.")

        def forward(self, *args, **kwargs):
            raise ImportError("torch is required to use CausalGenerator.")


__all__ = ["CausalGenerator"]
