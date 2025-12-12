#!/usr/bin/env python3
"""
Replace all neon lime green (#B6FF3B) with professional success green.
"""

import re
from pathlib import Path

# Replacement mapping
REPLACEMENTS = {
    'Color(0xFFB6FF3B)': 'AppColors.successGreen',
    '0xFFB6FF3B': '0xFF10B981',  # Fallback for non-Color contexts
}

# Files to process
FILES = [
    'technic_app/lib/utils/mock_data.dart',
    'technic_app/lib/screens/settings/settings_page.dart',
    'technic_app/lib/screens/ideas/ideas_page.dart',
    'technic_app/lib/app_shell.dart',
]

def fix_file(filepath):
    """Replace neon colors in a file."""
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️  File not found: {filepath}")
        return False
    
    content = path.read_text(encoding='utf-8')
    original = content
    
    # Apply replacements
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
    
    if content != original:
        path.write_text(content, encoding='utf-8')
        print(f"✅ Fixed: {filepath}")
        return True
    else:
        print(f"ℹ️  No changes: {filepath}")
        return False

def main():
    print("🎨 Removing neon lime green colors...\n")
    
    fixed_count = 0
    for filepath in FILES:
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\n✨ Complete! Fixed {fixed_count} files.")
    print("\n📝 Note: You may need to add 'import theme/app_colors.dart' to some files.")

if __name__ == '__main__':
    main()
