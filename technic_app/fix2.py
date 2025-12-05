from pathlib import Path
lines = Path('lib/main.dart').read_text(encoding='utf-8', errors='ignore').splitlines()
# ensure index exists
if len(lines) > 506:
    lines[506] = "                .map((m) => _listRow(m.ticker, ' · ', m.isPositive))"
# fix bullets
repls = {
    'Summarize today': '• Summarize today’s scan highlights',
    'Explain risk on NVDA setup': '• Explain risk on NVDA setup',
    'Compare TSLA vs AAPL momentum': '• Compare TSLA vs AAPL momentum',
    'Mode: Swing / Long-term': '• Mode: Swing / Long-term',
    'Risk per trade: 1.0%': '• Risk per trade: 1.0%',
    'Universe: US Equities': '• Universe: US Equities',
    '?': '•',
}
lines = [line for line in lines]
for i,line in enumerate(lines):
    for k,v in repls.items():
        if k in line:
            lines[i] = line.replace(k, v)
text = '\n'.join(lines)
Path('lib/main.dart').write_text(text, encoding='utf-8')
