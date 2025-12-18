#!/usr/bin/env python3
"""
Fix remaining Flutter/Dart warnings
"""

import re
from pathlib import Path

def fix_shadows():
    """Fix withOpacity in shadows.dart"""
    file_path = Path("technic_mobile/lib/theme/shadows.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Fix deprecated withOpacity
    content = re.sub(
        r'\.withOpacity\(([0-9.]+)\)',
        r'.withValues(alpha: \1)',
        content
    )
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed shadows.dart")

def fix_glass_container():
    """Fix withOpacity in glass_container.dart"""
    file_path = Path("technic_mobile/lib/widgets/glass_container.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Fix deprecated withOpacity
    content = re.sub(
        r'\.withOpacity\(([0-9.]+)\)',
        r'.withValues(alpha: \1)',
        content
    )
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed glass_container.dart")

def fix_mac_card():
    """Fix withOpacity in mac_card.dart"""
    file_path = Path("technic_mobile/lib/widgets/mac_card.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Fix deprecated withOpacity
    content = re.sub(
        r'\.withOpacity\(([0-9.]+)\)',
        r'.withValues(alpha: \1)',
        content
    )
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed mac_card.dart")

def delete_backup_folder():
    """Delete the lib_backup folder to remove those warnings"""
    import shutil
    backup_path = Path("technic_mobile/lib_backup")
    
    if backup_path.exists():
        shutil.rmtree(backup_path)
        print("✓ Deleted lib_backup folder (old backup files)")
    else:
        print("  lib_backup folder not found (already deleted)")

def main():
    """Run all fixes"""
    print("🔧 Fixing remaining Flutter/Dart warnings...\n")
    
    try:
        fix_shadows()
        fix_glass_container()
        fix_mac_card()
        delete_backup_folder()
        
        print("\n✅ All remaining warnings fixed!")
        print("\nFixed:")
        print("  • Replaced withOpacity with withValues in shadows.dart")
        print("  • Replaced withOpacity with withValues in glass_container.dart")
        print("  • Replaced withOpacity with withValues in mac_card.dart")
        print("  • Deleted lib_backup folder (old backup files)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
