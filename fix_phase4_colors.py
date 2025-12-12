"""
Phase 4: Fix color system conflicts and update theme
"""
import re
from pathlib import Path

def fix_tone_imports():
    """Remove tone import from app_colors, keep only in helpers"""
    
    # Files that import both - we'll hide tone from app_colors
    files_to_fix = [
        'technic_app/lib/app_shell.dart',
        'technic_app/lib/screens/scanner/scanner_page.dart',
        'technic_app/lib/screens/settings/settings_page.dart',
        'technic_app/lib/screens/copilot/copilot_page.dart',
        'technic_app/lib/screens/copilot/widgets/message_bubble.dart',
        'technic_app/lib/screens/ideas/ideas_page.dart',
        'technic_app/lib/screens/ideas/widgets/idea_card.dart',
        'technic_app/lib/screens/scanner/widgets/filter_panel.dart',
        'technic_app/lib/screens/scanner/widgets/market_pulse_card.dart',
        'technic_app/lib/screens/scanner/widgets/onboarding_card.dart',
        'technic_app/lib/screens/scanner/widgets/preset_manager.dart',
        'technic_app/lib/screens/scanner/widgets/quick_actions.dart',
        'technic_app/lib/screens/scanner/widgets/scan_result_card.dart',
        'technic_app/lib/screens/scanner/widgets/scoreboard_card.dart',
        'technic_app/lib/screens/symbol_detail/symbol_detail_page.dart',
    ]
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            print(f"Skipping {file_path} - not found")
            continue
            
        content = path.read_text(encoding='utf-8')
        
        # Add hide clause to app_colors import if not already there
        if "import '../../theme/app_colors.dart'" in content or "import '../theme/app_colors.dart'" in content:
            # Find the app_colors import line
            content = re.sub(
                r"(import ['\"].*?app_colors\.dart['\"];)",
                r"\1 // Using tone from helpers.dart",
                content
            )
        
        path.write_text(content, encoding='utf-8')
        print(f"Fixed {file_path}")

def update_app_colors():
    """Remove tone function from app_colors.dart to avoid conflict"""
    path = Path('technic_app/lib/theme/app_colors.dart')
    content = path.read_text(encoding='utf-8')
    
    # Remove the deprecated tone function and its comment
    content = re.sub(
        r'/// Helper function for backward compatibility\n@Deprecated.*?\nColor tone\(Color base, double opacity\) \{\n  return AppColors\.withOpacity\(base, opacity\);\n\}',
        '',
        content,
        flags=re.DOTALL
    )
    
    path.write_text(content, encoding='utf-8')
    print("Removed tone() from app_colors.dart")

if __name__ == '__main__':
    print("Phase 4: Fixing color system conflicts...")
    update_app_colors()
    fix_tone_imports()
    print("Done! Now updating app_theme.dart...")
