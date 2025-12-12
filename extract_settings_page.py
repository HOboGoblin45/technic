#!/usr/bin/env python3
"""Extract SettingsPage from main.dart"""

def extract_settings_page():
    with open('technic_app/lib/main.dart', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find start of SettingsPage class
    start_idx = None
    for i, line in enumerate(lines):
        if 'class SettingsPage extends StatelessWidget' in line:
            start_idx = i
            break
    
    if start_idx is None:
        print("SettingsPage not found!")
        return
    
    # Find end of SettingsPage class (next class definition or end of file)
    end_idx = len(lines)
    brace_count = 0
    in_class = False
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        
        # Count braces
        brace_count += line.count('{') - line.count('}')
        
        if '{' in line and not in_class:
            in_class = True
        
        # If we've closed all braces and we're in the class, we're done
        if in_class and brace_count == 0:
            end_idx = i + 1
            break
    
    # Extract the SettingsPage section
    settings_lines = lines[start_idx:end_idx]
    
    print(f"Found SettingsPage from line {start_idx + 1} to {end_idx}")
    print(f"Total lines: {len(settings_lines)}")
    print("\n" + "="*80)
    print("".join(settings_lines))

if __name__ == '__main__':
    extract_settings_page()
