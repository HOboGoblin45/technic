#!/usr/bin/env python3
"""
Fix all remaining compilation errors
"""

from pathlib import Path
import re

def fix_app_shell():
    """Fix all app_shell.dart errors"""
    path = Path('technic_app/lib/app_shell.dart')
    content = path.read_text(encoding='utf-8')
    
    # Remove unused _titles field
    content = re.sub(
        r'\s*final List<String> _titles = const \[[\s\S]*?\];',
        '',
        content
    )
    
    # Remove iconTheme line (not a valid parameter)
    content = re.sub(
        r'\s*// Force white icons and text\s*iconTheme: WidgetStateProperty\.all\(\s*const IconThemeData\(color: Colors\.white, size: 24\),\s*\),',
        '',
        content
    )
    
    # Remove duplicate indicatorColor
    lines = content.split('\n')
    new_lines = []
    seen_indicator = False
    for line in lines:
        if 'indicatorColor:' in line:
            if not seen_indicator:
                new_lines.append(line)
                seen_indicator = True
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)
    
    # Remove duplicate labelBehavior
    lines = content.split('\n')
    new_lines = []
    seen_label = False
    for line in lines:
        if 'labelBehavior:' in line:
            if not seen_label:
                new_lines.append(line)
                seen_label = True
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed app_shell.dart")

def fix_app_colors():
    """Fix app_colors.dart - add warningOrange properly"""
    path = Path('technic_app/lib/theme/app_colors.dart')
    content = path.read_text(encoding='utf-8')
    
    # Remove any incorrectly added warningOrange
    content = re.sub(
        r'\s*static const Color warningOrange = Color\(0xFFFF9800\);.*',
        '',
        content
    )
    
    # Add warningOrange in the right place (after other colors)
    if 'warningOrange' not in content:
        content = re.sub(
            r'(static const Color successGreen = Color\(0xFF10B981\);)',
            r'\1\n  static const Color warningOrange = Color(0xFFFF9800);',
            content
        )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed app_colors.dart")

def main():
    print("🔧 Fixing all remaining errors...\n")
    
    fix_app_shell()
    fix_app_colors()
    
    print("\n✨ All errors fixed!")
    print("\n📝 Run 'cd technic_app; flutter analyze' to verify")

if __name__ == '__main__':
    main()
