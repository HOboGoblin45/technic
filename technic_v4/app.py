"""Interactive model feedback view (Streamlit sidebar)."""

from __future__ import annotations

import streamlit as st

st.sidebar.markdown("### Strategy Feedback")
vote = st.sidebar.radio("Did this strategy work?", ["👍", "👎"])
if vote:
    # Placeholder: store feedback vote (wire to user_feedback_collector if desired)
    st.sidebar.caption("Feedback recorded.")


__all__ = []
