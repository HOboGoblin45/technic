"""Fund-level explainability card export to PDF."""

from __future__ import annotations

try:
    from fpdf import FPDF  # type: ignore

    HAVE_FPDF = True
except ImportError:  # pragma: no cover
    HAVE_FPDF = False


def export_summary_pdf(portfolio_summary):
    """Export a simplified PDF summary of portfolio rationale and factor drivers."""
    if not HAVE_FPDF:
        raise ImportError("fpdf is required to export PDF summaries.")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for key, val in portfolio_summary.items():
        pdf.cell(200, 10, txt=f"{key}: {val}", ln=True)
    pdf.output("portfolio_summary.pdf")


__all__ = ["export_summary_pdf"]
