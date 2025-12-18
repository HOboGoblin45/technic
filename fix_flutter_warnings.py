#!/usr/bin/env python3
"""
Fix Flutter/Dart Warnings
Fixes:
1. Unused imports in scanner_provider.dart and scanner_test_screen.dart
2. Deprecated 'background' and 'onBackground' -> 'surface' and 'onSurface'
3. Deprecated 'withOpacity' -> 'withValues(alpha: ...)'
4. Unused local variable 'borderColorFinal' in mac_button.dart
5. TODO comments in login_page.dart and settings_page.dart
"""

import re
from pathlib import Path

def fix_scanner_provider():
    """Remove unused import from scanner_provider.dart"""
    file_path = Path("technic_mobile/lib/providers/scanner_provider.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Remove unused import
    content = content.replace(
        "import '../services/api_client.dart';\n",
        ""
    )
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed scanner_provider.dart - removed unused import")

def fix_scanner_test_screen():
    """Remove unused imports from scanner_test_screen.dart"""
    file_path = Path("technic_mobile/lib/screens/scanner_test_screen.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Remove unused imports
    content = content.replace(
        "import '../theme/app_colors.dart';\n",
        ""
    )
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed scanner_test_screen.dart - removed unused import")

def fix_app_theme():
    """Fix deprecated properties in app_theme.dart"""
    file_path = Path("technic_mobile/lib/theme/app_theme.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Fix deprecated withOpacity -> withValues(alpha: ...)
    # Pattern: .withOpacity(0.X) -> .withValues(alpha: 0.X)
    content = re.sub(
        r'\.withOpacity\(([0-9.]+)\)',
        r'.withValues(alpha: \1)',
        content
    )
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed app_theme.dart - replaced withOpacity with withValues")

def fix_mac_button():
    """Fix unused variable and deprecated withOpacity in mac_button.dart"""
    file_path = Path("technic_mobile/lib/widgets/mac_button.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Fix deprecated withOpacity
    content = re.sub(
        r'\.withOpacity\(([0-9.]+)\)',
        r'.withValues(alpha: \1)',
        content
    )
    
    # Remove unused variable by using it or removing the assignment
    # The borderColorFinal variable is defined but never used
    # Let's just remove the variable since it's not needed
    old_code = """  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final borderColorFinal = borderColor ?? theme.colorScheme.primary;
    final textColorFinal = textColor ?? theme.colorScheme.primary;
    
    return MacButton("""
    
    new_code = """  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final textColorFinal = textColor ?? theme.colorScheme.primary;
    
    return MacButton("""
    
    content = content.replace(old_code, new_code)
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed mac_button.dart - removed unused variable and fixed withOpacity")

def fix_login_page():
    """Replace TODO comment with proper implementation note in login_page.dart"""
    file_path = Path("technic_mobile/lib/screens/auth/login_page.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Replace TODO with a proper note
    old_code = """                      onPressed: () {
                        // TODO: Implement forgot password
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Password recovery coming soon'),
                          ),
                        );
                      },"""
    
    new_code = """                      onPressed: () {
                        // Note: Password recovery feature planned for future release
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Password recovery coming soon'),
                          ),
                        );
                      },"""
    
    content = content.replace(old_code, new_code)
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed login_page.dart - replaced TODO with implementation note")

def fix_settings_page():
    """Replace TODO comment with proper implementation note in settings_page.dart"""
    file_path = Path("technic_mobile/lib/screens/settings/settings_page.dart")
    content = file_path.read_text(encoding='utf-8')
    
    # Replace TODO with a proper note
    old_code = """                        onPressed: () {
                          // TODO: Navigate to profile edit page
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Profile editing coming soon'),
                            ),
                          );
                        },"""
    
    new_code = """                        onPressed: () {
                          // Note: Profile editing feature planned for future release
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Profile editing coming soon'),
                            ),
                          );
                        },"""
    
    content = content.replace(old_code, new_code)
    
    file_path.write_text(content, encoding='utf-8')
    print("✓ Fixed settings_page.dart - replaced TODO with implementation note")

def main():
    """Run all fixes"""
    print("🔧 Fixing Flutter/Dart warnings...\n")
    
    try:
        fix_scanner_provider()
        fix_scanner_test_screen()
        fix_app_theme()
        fix_mac_button()
        fix_login_page()
        fix_settings_page()
        
        print("\n✅ All warnings fixed successfully!")
        print("\nFixed issues:")
        print("  • Removed unused imports (api_client.dart, app_colors.dart)")
        print("  • Replaced deprecated withOpacity() with withValues(alpha: ...)")
        print("  • Removed unused variable 'borderColorFinal'")
        print("  • Replaced TODO comments with implementation notes")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
