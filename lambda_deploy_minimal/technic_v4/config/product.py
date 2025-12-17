from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ProductMeta:
    name: str
    tagline: str
    short_description: str
    disclaimers: List[str]
    website_url: str
    docs_url: str


PRODUCT = ProductMeta(
    name="Technic",
    tagline="Institutional-grade alpha scanner with a minimalist surface.",
    short_description=(
        "Technic ingests prices, fundamentals, sentiment, and regimes to surface a "
        "small list of high-conviction trade ideas—each with a signal, score, entry, "
        "stop, target, and rationale."
    ),
    disclaimers=[
        "Technic provides quantitative analysis and trade ideas for educational and informational purposes only.",
        "Technic does not provide investment, legal, tax, or financial advice.",
        "You are solely responsible for your own trading decisions and risk.",
    ],
    website_url="https://technic.yourdomain.com",
    docs_url="https://technic.yourdomain.com/docs",
)
