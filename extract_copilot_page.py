#!/usr/bin/env python3
"""Extract CopilotPage from main.dart"""

with open('technic_app/lib/main.dart', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find CopilotPage class (line 2291-2296)
copilot_start = 2290  # 0-indexed
copilot_end = 2296

# Find _CopilotPageState class (line 2298-2676)
state_start = 2297  # 0-indexed
state_end = 2676

# Extract both classes
copilot_class = lines[copilot_start:copilot_end]
state_class = lines[state_start:state_end]

print(f"CopilotPage: lines {copilot_start+1}-{copilot_end}")
print(f"_CopilotPageState: lines {state_start+1}-{state_end}")
print(f"Total lines: {len(copilot_class) + len(state_class)}")
print()
print("="*80)
print("".join(copilot_class + state_class))
