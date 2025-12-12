#!/usr/bin/env python3
"""Extract IdeasPage from main.dart"""

with open('technic_app/lib/main.dart', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract IdeasPage (line 2108-2113) and _IdeasPageState (line 2115-2289)
ideas_class = lines[2107:2113]  # 0-indexed
state_class = lines[2114:2289]

print('IdeasPage + _IdeasPageState:')
print('='*80)
content = ''.join(ideas_class + state_class)
print(content[:3000])  # First 3000 chars
print('\n...\n')
print(f'Total length: {len(content)} characters')
print(f'Total lines: {len(ideas_class) + len(state_class)}')
