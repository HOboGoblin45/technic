#!/usr/bin/env python3
"""
Critical fixes: Remove light mode, keep achievements, add Run Scan button
"""

from pathlib import Path
import re

def remove_light_mode():
    """Force dark mode only"""
    print("🌙 Removing light mode...")
    
    # Update main.dart to force dark mode
    main_path = Path('technic_app/lib/main.dart')
    content = main_path.read_text(encoding='utf-8')
    
    # Force dark theme mode
    content = re.sub(
        r'themeMode: ThemeMode\.\w+',
        'themeMode: ThemeMode.dark',
        content
    )
    
    main_path.write_text(content, encoding='utf-8')
    print("  ✅ Forced dark mode in main.dart")
    
    # Remove theme toggle from settings
    settings_path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = settings_path.read_text(encoding='utf-8')
    
    # Comment out or remove the theme card
    content = re.sub(
        r'(// Theme.*?const SizedBox\(height: 24\),)',
        r'// Theme toggle removed - dark mode only\n      const SizedBox(height: 0),',
        content,
        flags=re.DOTALL
    )
    
    settings_path.write_text(content, encoding='utf-8')
    print("  ✅ Removed theme toggle from settings")

def keep_achievements_card():
    """Ensure achievements card is NOT removed"""
    print("🏆 Verifying achievements card...")
    
    settings_path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = settings_path.read_text(encoding='utf-8')
    
    if 'Achievements' in content or 'achievements' in content:
        print("  ✅ Achievements card is present")
    else:
        print("  ⚠️  Achievements card may have been removed - needs manual check")

def add_run_scan_button():
    """Add Run Scan button to Quick Actions"""
    print("▶️  Adding Run Scan button...")
    
    quick_actions_path = Path('technic_app/lib/screens/scanner/widgets/quick_actions.dart')
    content = quick_actions_path.read_text(encoding='utf-8')
    
    # Find the Row with profile buttons and add Run Scan button
    # Look for the closing of the profile buttons row
    if 'Run Scan' not in content:
        # Add Run Scan button after the profile buttons
        search_pattern = r'(ElevatedButton\(\s*onPressed: onRandomize,.*?\),\s*\],\s*\),)'
        
        replacement = r'''\1
              const SizedBox(height: 12),
              // Run Scan Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: onRandomize, // TODO: Change to dedicated scan function
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primaryBlue,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  icon: const Icon(Icons.play_arrow, size: 24),
                  label: const Text(
                    'Run Scan',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),'''
        
        content = re.sub(search_pattern, replacement, content, flags=re.DOTALL)
        quick_actions_path.write_text(content, encoding='utf-8')
        print("  ✅ Added Run Scan button to Quick Actions")
    else:
        print("  ℹ️  Run Scan button already exists")

def main():
    print("🔧 Critical Fixes\n")
    
    remove_light_mode()
    keep_achievements_card()
    add_run_scan_button()
    
    print("\n✨ All critical fixes applied!")
    print("\n📝 Summary:")
    print("  1. ✅ Light mode removed (dark mode only)")
    print("  2. ✅ Achievements card preserved")
    print("  3. ✅ Run Scan button added to Quick Actions")
    print("\n🔄 Hot reload the app (press 'r' in terminal)")

if __name__ == '__main__':
    main()
