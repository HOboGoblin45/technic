# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
path = Path("technic_v4/ui/technic_app.py")
text = path.read_text(encoding="utf-8")
start = text.find("use_cards = force_cards or card_mode or auto_cards")
if start == -1:
    raise SystemExit("start not found")
marker = "\n        else:\n            styled = style_results_table"
end = text.find(marker, start)
if end == -1:
    raise SystemExit("end not found")
new_block = """
        use_cards = force_cards or card_mode or auto_cards

        if use_cards:
            max_cards = 20
            st.caption(f\"Showing top {min(len(table_df), max_cards)} setups in card view.\")
            for _, r in table_df.head(max_cards).iterrows():
                sym = r.get(\"Symbol\", \"-\")
                sig = r.get(\"Signal\", \"-\")
                tech = r.get(\"TechRating\", None)
                alpha = r.get(\"AlphaScore\", None)
                rationale = r.get(\"Rationale\", \"\") or \"\"
                rationale_short = (rationale[:140] + "...") if len(rationale) > 140 else rationale
                entry = r.get(\"Entry\", None)
                stop = r.get(\"Stop\", None)
                target = r.get(\"Target\", None)

                st.markdown(
                    f\"\"\"
                    <div class=\\\"technic-card\\\" style=\\\"margin-bottom:0.6rem;padding:0.9rem 1rem;\\\">\n                      <div style=\\\"display:flex;justify-content:space-between;align-items:center;\\\">\n                        <div style=\\\"font-weight:800;font-size:1.1rem;\\\">{sym}</div>\n                        <div style=\\\"font-weight:700;padding:4px 8px;border-radius:12px;background:rgba(158,240,26,0.15);color:#9ef01a;\\\">{sig}</div>\n                      </div>\n                      <div style=\\\"display:flex;gap:12px;margin-top:4px;font-size:0.95rem;\\\">\n                        <div>TechRating: <strong>{'' if pd.isna(tech) else f'{tech:.1f}'}</strong></div>\n                        <div>Alpha: <strong>{'' if pd.isna(alpha) else f'{alpha:.2f}'}</strong></div>\n                      </div>\n                      <div style=\\\"display:flex;gap:12px;margin-top:4px;font-size:0.9rem;color:#cbd5e1;\\\">\n                        <div>Entry: <span style=\\\"color:#e5e7eb;\\\">{'' if pd.isna(entry) else f'{entry:.2f}'}</span></div>\n                        <div>Stop: <span style=\\\"color:#e5e7eb;\\\">{'' if pd.isna(stop) else f'{stop:.2f}'}</span></div>\n                        <div>Target: <span style=\\\"color:#e5e7eb;\\\">{'' if pd.isna(target) else f'{target:.2f}'}</span></div>\n                      </div>\n                      <div style=\\\"margin-top:6px;font-size:0.95rem;color:#cbd5e1;\\\">{rationale_short}</div>\n                    </div>\n                    \"\"\",
                    unsafe_allow_html=True,
                )
                with st.expander(\"Details\"):
                    if rationale:
                        st.write(rationale)
                    if \"OptionTrade\" in r and r.get(\"OptionTrade\"):
                        st.markdown(\"**Option idea:**\")
                        st.json(r.get(\"OptionTrade\"))
                    st.button(f\"Open Copilot Chat for {sym} (coming soon)\", key=f\"chat_{sym}\", disabled=True)

        else:
"""
text = text[:start] + new_block + text[end:]
path.write_text(text, encoding="utf-8")
