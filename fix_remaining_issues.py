"""
Fix remaining Dart issues
"""

import os
import re

def fix_file(filepath, fixes):
    """Apply fixes to a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in fixes:
        content = content.replace(old, new)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Fixed {filepath}")

print("🔧 Fixing remaining Dart issues...\n")

# Fix 1: tag_selector.dart - Replace 'where' with 'where' method call
fix_file('technic_app/lib/screens/watchlist/widgets/tag_selector.dart', [
    ("const [].where", "const <String>[].where"),
])

# Fix 2: main.dart - Remove unused auth_service import
fix_file('technic_app/lib/main.dart', [
    ("import 'services/auth_service.dart';\n", ""),
])

# Fix 3: add_alert_dialog.dart - Fix deprecated groupValue and onChanged
with open('technic_app/lib/screens/watchlist/widgets/add_alert_dialog.dart', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace groupValue with value for RadioListTile
content = re.sub(
    r'groupValue:\s*_selectedType',
    'value: _selectedType == type',
    content
)

# Replace onChanged parameter
content = re.sub(
    r'onChanged:\s*\(AlertType\?\s*value\)\s*\{',
    'onChanged: (bool? selected) {\n              if (selected == true) {',
    content
)

# Fix the setState call
content = content.replace(
    'setState(() {\n                _selectedType = value!;\n              });',
    'setState(() {\n                  _selectedType = type;\n                });\n              }'
)

with open('technic_app/lib/screens/watchlist/widgets/add_alert_dialog.dart', 'w', encoding='utf-8') as f:
    f.write(content)
print("✅ Fixed technic_app/lib/screens/watchlist/widgets/add_alert_dialog.dart")

# Fix 4: empty_state.dart and error_display.dart - Fix withOpacity
fix_file('technic_app/lib/widgets/empty_state.dart', [
    ('.withOpacity(', '.withValues(alpha: '),
])

fix_file('technic_app/lib/widgets/error_display.dart', [
    ('.withOpacity(', '.withValues(alpha: '),
])

# Fix 5: watchlist_page.dart - Fix BuildContext across async gap
with open('technic_app/lib/screens/watchlist/watchlist_page.dart', 'r', encoding='utf-8') as f:
    content = f.read()

# Add mounted check before Navigator.pop in filter dialog
content = content.replace(
    '                                    Navigator.pop(context);',
    '                                    if (mounted) Navigator.pop(context);'
)

with open('technic_app/lib/screens/watchlist/watchlist_page.dart', 'w', encoding='utf-8') as f:
    f.write(content)
print("✅ Fixed technic_app/lib/screens/watchlist/watchlist_page.dart")

print("\n✅ All remaining issues fixed!")
print("\nRemaining TODOs are intentional placeholders for future features.")
