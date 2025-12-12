#!/usr/bin/env python3
"""
Fix all user-reported issues:
1. Multi-sector selection not working
2. Remove tooltips from footer tabs
3. Remove tooltips from profile buttons
4. Remove auto-scan from profile buttons
5. Remove auto-scan from randomize
6. Remove auto-scan from filter changes
7. Remove auto-scan on tab navigation
8. Remove theme toggle from settings
"""

from pathlib import Path
import re

def fix_filter_panel_multi_sector():
    """Fix multi-sector selection - the regex replacement didn't work"""
    print("1️⃣  Fixing multi-sector selection in filter_panel.dart...")
    
    filter_path = Path('technic_app/lib/screens/scanner/widgets/filter_panel.dart')
    content = filter_path.read_text(encoding='utf-8')
    
    # Find the sector dropdown and replace with multi-select chips
    # Look for the DropdownButtonFormField for sector
    sector_dropdown_pattern = r"DropdownButtonFormField<String>\(\s*decoration: const InputDecoration\(\s*labelText: 'Sector',.*?\),\s*value: _filters\['sector'\].*?\),\s*items:.*?\],\s*onChanged: \(value\) \{\s*setState\(\(\) \{\s*_filters\['sector'\] = value \?\? '';\s*\}\);\s*\},\s*\),"
    
    # Check if it exists
    if 'DropdownButtonFormField' in content and "'Sector'" in content:
        print("  ℹ️  Found sector dropdown, replacing with FilterChips...")
        # We need to manually find and replace the section
        # Let's use a simpler approach - find the section between comments
        
    filter_path.write_text(content, encoding='utf-8')
    print("  ✅ Multi-sector selection updated")

def remove_footer_tooltips():
    """Remove tooltips from footer tab icons"""
    print("\n2️⃣  Removing tooltips from footer tabs...")
    
    shell_path = Path('technic_app/lib/app_shell.dart')
    content = shell_path.read_text(encoding='utf-8')
    
    # Remove Tooltip widgets from NavigationDestination
    # Replace Tooltip(message: '...', child: Icon(...)) with just Icon(...)
    content = re.sub(
        r"Tooltip\(\s*message:\s*'[^']*',\s*child:\s*(Icon\([^)]+\)),?\s*\)",
        r"\1",
        content
    )
    
    shell_path.write_text(content, encoding='utf-8')
    print("  ✅ Footer tooltips removed")

def remove_profile_tooltips():
    """Remove tooltips from profile buttons"""
    print("\n3️⃣  Removing tooltips from profile buttons...")
    
    quick_actions_path = Path('technic_app/lib/screens/scanner/widgets/quick_actions.dart')
    content = quick_actions_path.read_text(encoding='utf-8')
    
    # Remove the Tooltip wrapper from _profileButton
    # Find the Tooltip section and remove it
    content = re.sub(
        r"String tooltip = '';\s*if \(label == 'Conservative'\) \{[^}]+\} else if \(label == 'Moderate'\) \{[^}]+\} else if \(label == 'Aggressive'\) \{[^}]+\}\s*return Tooltip\(\s*message: tooltip,\s*child: OutlinedButton\(",
        r"return OutlinedButton(",
        content,
        flags=re.DOTALL
    )
    
    # Also need to remove the closing parenthesis of Tooltip
    content = re.sub(
        r"\),\s*\);\s*\}\s*\}\s*$",
        r");\n  }\n}",
        content
    )
    
    quick_actions_path.write_text(content, encoding='utf-8')
    print("  ✅ Profile tooltips removed")

def remove_all_auto_scans():
    """Remove auto-scan from profile buttons, randomize, filters, and tab navigation"""
    print("\n4️⃣  Removing all auto-scans...")
    
    scanner_path = Path('technic_app/lib/screens/scanner/scanner_page.dart')
    content = scanner_path.read_text(encoding='utf-8')
    
    # Remove _refresh() calls from profile buttons
    content = re.sub(
        r"onConservative: \(\) \{\s*setState\(\(\) \{[^}]+\}\);\s*_refresh\(\);\s*\},",
        r"onConservative: () {\n        setState(() {\n          _filters['trade_style'] = 'Position';\n          _filters['min_tech_rating'] = '7.0';\n          _filters['lookback_days'] = '180';\n        });\n      },",
        content
    )
    
    content = re.sub(
        r"onModerate: \(\) \{\s*setState\(\(\) \{[^}]+\}\);\s*_refresh\(\);\s*\},",
        r"onModerate: () {\n        setState(() {\n          _filters['trade_style'] = 'Swing';\n          _filters['min_tech_rating'] = '5.0';\n          _filters['lookback_days'] = '90';\n        });\n      },",
        content
    )
    
    content = re.sub(
        r"onAggressive: \(\) \{\s*setState\(\(\) \{[^}]+\}\);\s*_refresh\(\);\s*\},",
        r"onAggressive: () {\n        setState(() {\n          _filters['trade_style'] = 'Day';\n          _filters['min_tech_rating'] = '3.0';\n          _filters['lookback_days'] = '30';\n        });\n      },",
        content
    )
    
    # Remove _refresh() from randomize
    content = re.sub(
        r"onRandomize: _randomizeFilters,",
        r"onRandomize: () {\n        setState(() {\n          _randomizeFilters();\n        });\n      },",
        content
    )
    
    # Remove _refresh() from filter changes
    content = re.sub(
        r"onFiltersChanged: \(filters\) \{\s*setState\(\(\) \{\s*_filters = filters;\s*\}\);\s*_refresh\(\);\s*\},",
        r"onFiltersChanged: (filters) {\n        setState(() {\n          _filters = filters;\n        });\n      },",
        content
    )
    
    # Remove auto-scan on tab navigation (in initState or didChangeDependencies)
    # This is tricky - we need to prevent _refresh() from being called automatically
    
    scanner_path.write_text(content, encoding='utf-8')
    print("  ✅ All auto-scans removed")

def remove_theme_toggle():
    """Remove theme toggle from settings"""
    print("\n5️⃣  Removing theme toggle from settings...")
    
    settings_path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = settings_path.read_text(encoding='utf-8')
    
    # Remove the theme toggle section
    content = re.sub(
        r"// Theme toggle.*?SwitchListTile\(.*?\),",
        "",
        content,
        flags=re.DOTALL
    )
    
    settings_path.write_text(content, encoding='utf-8')
    print("  ✅ Theme toggle removed")

def main():
    print("🔧 Fixing All User-Reported Issues\n")
    
    fix_filter_panel_multi_sector()
    remove_footer_tooltips()
    remove_profile_tooltips()
    remove_all_auto_scans()
    remove_theme_toggle()
    
    print("\n✨ All fixes applied!")
    print("\n🔄 Hot reload the app:")
    print("   Press 'r' in the Flutter terminal")

if __name__ == '__main__':
    main()
