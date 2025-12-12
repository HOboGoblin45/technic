# -*- coding: utf-8 -*-
from pathlib import Path
p = Path('technic_v4/scanner_core.py')
lines = p.read_text().splitlines()
start = next(i for i,l in enumerate(lines) if '# Structured picks (for UI/API)' in l)
end = next(i for i,l in enumerate(lines[start+1:], start+1) if 'except Exception:' in l)
lines[start:end] = ['    ' + ln for ln in lines[start:end]]
p.write_text('\n'.join(lines))
