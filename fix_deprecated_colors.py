"""
Fix all deprecated color references in the codebase
"""
import re
from pathlib import Path

def fix_file(file_path):
    """Fix deprecated color references in a single file"""
    path = Path(file_path)
    if not path.exists():
        return False
    
    content = path.read_text(encoding='utf-8')
    original = content
    
    # Replace deprecated color names
    replacements = {
        'AppColors.skyBlue': 'AppColors.primaryBlue',
        'AppColors.darkDeep': 'AppColors.darkBackground',
        'AppColors.darkBg': 'AppColors.darkCard',
        'AppColors.darkAccent': 'AppColors.darkBorder',
        'AppColors.pineGrove': 'AppColors.successGreen',  # Guess based on context
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    if content != original:
        path.write_text(content, encoding='utf-8')
        return True
    return False

# Files to fix
files_to_fix = [
    'technic_app/lib/app_shell.dart',
    'technic_app/lib/screens/copilot/copilot_page.dart',
    'technic_app/lib/screens/copilot/widgets/message_bubble.dart',
    'technic_app/lib/screens/ideas/ideas_page.dart',
    'technic_app/lib/screens/ideas/widgets/idea_card.dart',
    'technic_app/lib/screens/scanner/scanner_page.dart',
    'technic_app/lib/screens/scanner/widgets/filter_panel.dart',
    'technic_app/lib/screens/scanner/widgets/market_pulse_card.dart',
    'technic_app/lib/screens/scanner/widgets/onboarding_card.dart',
    'technic_app/lib/screens/scanner/widgets/preset_manager.dart',
    'technic_app/lib/screens/scanner/widgets/quick_actions.dart',
    'technic_app/lib/screens/scanner/widgets/scan_result_card.dart',
    'technic_app/lib/screens/scanner/widgets/scoreboard_card.dart',
    'technic_app/lib/screens/settings/settings_page.dart',
    'technic_app/lib/screens/symbol_detail/symbol_detail_page.dart',
    'technic_app/lib/widgets/info_card.dart',
]

print("Fixing deprecated color references...")
fixed_count = 0
for file_path in files_to_fix:
    if fix_file(file_path):
        print(f"✓ Fixed {file_path}")
        fixed_count += 1
    else:
        print(f"  Skipped {file_path} (no changes needed)")

print(f"\nFixed {fixed_count} files")
