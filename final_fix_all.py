#!/usr/bin/env python3
"""
Final comprehensive fix for all remaining errors
"""

from pathlib import Path
import re

def fix_app_shell_navigation_bar():
    """Fix NavigationBar parameters in app_shell.dart"""
    path = Path('technic_app/lib/app_shell.dart')
    content = path.read_text(encoding='utf-8')
    
    # Find and replace the NavigationBar section
    # Remove iconTheme, keep only one indicatorColor and one labelBehavior
    nav_bar_pattern = r'child: NavigationBar\([^)]+\),'
    
    # Simple approach: replace the entire NavigationBar configuration
    old_nav = re.search(
        r'(child: NavigationBar\(\s*selectedIndex: _index,.*?onDestinationSelected:.*?\},\s*destinations:)',
        content,
        re.DOTALL
    )
    
    if old_nav:
        new_nav = '''child: NavigationBar(
          selectedIndex: _index,
          backgroundColor: Colors.transparent,
          labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
          indicatorColor: tone(AppColors.primaryBlue, 0.18),
          height: 70,
          onDestinationSelected: (value) {
            setState(() => _index = value);
            LocalStore.saveLastTab(value);
          },
          destinations:'''
        
        content = content[:old_nav.start()] + new_nav + content[old_nav.end():]
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed app_shell.dart NavigationBar")

def fix_settings_withopacity():
    """Fix withOpacity deprecation in settings_page.dart"""
    path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = path.read_text(encoding='utf-8')
    
    # Replace .withOpacity( with .withValues(alpha:
    content = content.replace('.withOpacity(', '.withValues(alpha: ')
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed settings_page.dart withOpacity")

def main():
    print("🔧 Final comprehensive fix...\n")
    
    fix_app_shell_navigation_bar()
    fix_settings_withopacity()
    
    print("\n✨ All fixes applied!")
    print("\n📝 Run 'cd technic_app; flutter analyze' to verify")

if __name__ == '__main__':
    main()
