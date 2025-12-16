#!/usr/bin/env python3
"""Fix the malformed imports in scanner_core.py"""

import re

# Read the file
with open('technic_v4/scanner_core.py', 'r') as f:
    content = f.read()

# Find the imports section (everything before the first function/class definition)
lines = content.split('\n')
import_end = 0
for i, line in enumerate(lines):
    if line.startswith('logger = '):
        import_end = i
        break

# Extract the imports section
import_lines = lines[:import_end]

# Clean up the imports - remove duplicates and malformed lines
clean_imports = []
seen_imports = set()

for line in import_lines:
    # Skip malformed lines with multiple '======='
    if '=======' in line:
        continue
    # Skip empty lines in imports
    if not line.strip() and len(clean_imports) > 0 and clean_imports[-1].strip():
        clean_imports.append(line)
        continue
    # Add valid import lines
    if line.strip() and not line in seen_imports:
        clean_imports.append(line)
        seen_imports.add(line)

# Make sure we have the necessary imports
required_imports = [
    "from technic_v4.engine.meta_inference import score_win_prob_10d",
    "from technic_v4.engine.portfolio_optim import (",
    "    mean_variance_weights,",
    "    inverse_variance_weights,",
    "    hrp_weights,",
    ")",
    "from technic_v4.engine.recommendation import build_recommendation",
    "from technic_v4.engine.batch_processor import get_batch_processor",
    "import concurrent.futures"
]

# Check if imports are already there
for imp in required_imports:
    found = False
    for existing in clean_imports:
        if imp in existing:
            found = True
            break
    if not found and imp.strip():
        # Add before the last import line
        if imp.startswith("from technic_v4.engine"):
            # Find where to insert
            for i in range(len(clean_imports)-1, -1, -1):
                if clean_imports[i].startswith("from technic_v4"):
                    clean_imports.insert(i+1, imp)
                    break
        else:
            clean_imports.append(imp)

# Reconstruct the file
new_content = '\n'.join(clean_imports) + '\n' + '\n'.join(lines[import_end:])

# Write back
with open('technic_v4/scanner_core.py', 'w') as f:
    f.write(new_content)

print("Fixed scanner_core.py imports")
