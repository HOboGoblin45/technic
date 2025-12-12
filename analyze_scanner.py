#!/usr/bin/env python3
"""Analyze ScannerPage structure"""

with open('technic_app/lib/main.dart', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Analyze ScannerPage structure (lines 484-2106)
scanner_lines = lines[483:2106]

print('=== SCANNER PAGE STRUCTURE ===\n')
print(f'Total lines: {len(scanner_lines)}\n')

# Find key methods
methods = []
for i, line in enumerate(scanner_lines, 484):
    stripped = line.strip()
    
    # Track methods
    if ('void ' in line or 'Future' in line or 'Widget ' in line or 'bool get' in line) and not stripped.startswith('//'):
        if '(' in stripped or 'get ' in stripped:
            method_name = stripped.split('(')[0] if '(' in stripped else stripped
            methods.append((i, method_name[:80]))

print('Key Methods:')
for line_num, method in methods[:30]:  # First 30 methods
    print(f'  Line {line_num}: {method}')

print(f'\n... and {len(methods) - 30} more methods')
print(f'\nTotal methods found: {len(methods)}')
