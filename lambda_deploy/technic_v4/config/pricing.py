from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    price_usd_per_month: float
    description: str
    features: List[str]


PLANS: List[Plan] = [
    Plan(
        id="free",
        name="Free (Beta)",
        price_usd_per_month=0.0,
        description="Limited scans for evaluation and learning.",
        features=[
            "End-of-day scans on a core US universe",
            "Limited number of results per scan",
            "No options overlays",
        ],
    ),
    Plan(
        id="pro",
        name="Pro",
        price_usd_per_month=39.0,
        description="For active traders who want high-conviction swing ideas.",
        features=[
            "Intraday & end-of-day scans",
            "Full TechRating + ML alpha engine",
            "Options overlays on top ideas",
            "Rationales & basic trade management suggestions",
        ],
    ),
    Plan(
        id="desk",
        name="Desk / API",
        price_usd_per_month=99.0,
        description="For desks and power users who want API access and higher limits.",
        features=[
            "All Pro features",
            "API access to /v1/scan",
            "Higher universes and rate limits",
            "Priority support & research collaboration",
        ],
    ),
]
