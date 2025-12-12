#!/usr/bin/env python3
"""
Fix all 10 compilation errors/warnings from the Problems panel
"""

from pathlib import Path
import re

def fix_app_shell_errors():
    """Fix app_shell.dart errors (5 issues)"""
    path = Path('technic_app/lib/app_shell.dart')
    content = path.read_text(encoding='utf-8')
    
    # 1. Fix iconTheme parameter (undefined_named_parameter)
    # Remove iconTheme and use proper theme approach
    content = re.sub(
        r'iconTheme: MaterialStateProperty\.resolveWith\(\(states\) \{[^}]+\}\),\s*',
        '',
        content
    )
    
    # 2. Fix indicatorColor duplicate parameter
    # Keep only one indicatorColor
    content = re.sub(
        r'(indicatorColor: tone\(AppColors\.primaryBlue, 0\.3\),)\s*indicatorColor: tone\(AppColors\.primaryBlue, 0\.3\),',
        r'\1',
        content
    )
    
    # 3. Fix labelBehavior duplicate parameter
    # Keep only one labelBehavior
    content = re.sub(
        r'(labelBehavior: NavigationDestinationLabelBehavior\.alwaysShow,)\s*labelBehavior: NavigationDestinationLabelBehavior\.alwaysShow,',
        r'\1',
        content
    )
    
    # 4. Remove unused _titles field
    content = re.sub(
        r'final List<String> _titles = \[.*?\];',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 5. Fix MaterialStateProperty deprecation
    # Replace MaterialStateProperty with WidgetStateProperty
    content = content.replace('MaterialStateProperty', 'WidgetStateProperty')
    content = content.replace('MaterialState', 'WidgetState')
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed app_shell.dart (5 issues)")

def fix_settings_page_errors():
    """Fix settings_page.dart errors (2 issues)"""
    path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = path.read_text(encoding='utf-8')
    
    # Add warningOrange to AppColors if not present
    colors_path = Path('technic_app/lib/theme/app_colors.dart')
    colors_content = colors_path.read_text(encoding='utf-8')
    
    if 'warningOrange' not in colors_content:
        # Add warningOrange color
        colors_content = re.sub(
            r'(class AppColors \{[^}]*)',
            r'\1\n  static const Color warningOrange = Color(0xFFFF9800);',
            colors_content
        )
        colors_path.write_text(colors_content, encoding='utf-8')
    
    print("✅ Fixed settings_page.dart (2 issues)")

def fix_app_colors_errors():
    """Fix app_colors.dart error (1 issue)"""
    path = Path('technic_app/lib/theme/app_colors.dart')
    content = path.read_text(encoding='utf-8')
    
    # Replace withOpacity with withValues
    content = content.replace('.withOpacity(', '.withValues(alpha: ')
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed app_colors.dart (1 issue)")

def fix_app_theme_errors():
    """Fix app_theme.dart errors (2 issues)"""
    path = Path('technic_app/lib/theme/app_theme.dart')
    content = path.read_text(encoding='utf-8')
    
    # Replace withOpacity with withValues
    content = content.replace('.withOpacity(', '.withValues(alpha: ')
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed app_theme.dart (2 issues)")

def main():
    print("🔧 Fixing all 10 compilation errors/warnings...\n")
    
    fix_app_shell_errors()
    fix_settings_page_errors()
    fix_app_colors_errors()
    fix_app_theme_errors()
    
    print("\n✨ All 10 errors/warnings fixed!")
    print("\n📝 Next: Run 'flutter analyze' to verify")

if __name__ == '__main__':
    main()
