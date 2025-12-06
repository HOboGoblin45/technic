"""Signal blacklist/quarantine feature."""

from __future__ import annotations

quarantine_list = {"GME", "MEME", "TSLA_vol_model"}


def is_blacklisted(signal) -> bool:
    """Temporarily block known erroneous models or symbols."""
    return signal["symbol"] in quarantine_list or signal["model"] in quarantine_list


__all__ = ["is_blacklisted", "quarantine_list"]
