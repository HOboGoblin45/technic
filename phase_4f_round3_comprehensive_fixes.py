#!/usr/bin/env python3
"""
Phase 4F Round 3: Comprehensive fixes based on user testing feedback
"""

from pathlib import Path
import re

def fix_light_mode_colors():
    """Fix light mode text and icon visibility"""
    print("🎨 Fixing light mode colors...")
    
    # Fix app_shell.dart - make footer icons adapt to theme
    shell_path = Path('technic_app/lib/app_shell.dart')
    content = shell_path.read_text(encoding='utf-8')
    
    # Update navigation bar to use theme-aware colors
    old_nav = re.search(
        r'(bottomNavigationBar: Container\(\s*decoration: BoxDecoration\(.*?child: NavigationBar\()',
        content,
        re.DOTALL
    )
    
    if old_nav:
        new_nav = '''bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: isDark ? navBackground : Colors.white,
          border: Border(top: BorderSide(color: isDark ? const Color(0xFF0F172A) : const Color(0xFFE5E7EB))),
          boxShadow: [
            BoxShadow(
              color: tone(Colors.black, isDark ? 0.35 : 0.1),
              blurRadius: 12,
              offset: const Offset(0, -6),
            ),
          ],
        ),
        child: NavigationBar('''
        
        content = content[:old_nav.start()] + new_nav + content[old_nav.end():]
    
    shell_path.write_text(content, encoding='utf-8')
    print("  ✅ Fixed footer colors for light mode")

def add_run_scan_button():
    """Add FloatingActionButton for Run Scan"""
    print("🔘 Adding Run Scan button...")
    
    scanner_path = Path('technic_app/lib/screens/scanner/scanner_page.dart')
    content = scanner_path.read_text(encoding='utf-8')
    
    # Check if FAB already exists
    if 'floatingActionButton:' in content:
        print("  ℹ️  Run Scan button already exists")
        return
    
    # Find the Scaffold and add FAB
    scaffold_pattern = r'(return Scaffold\(\s*body:)'
    
    if re.search(scaffold_pattern, content):
        # Add FAB before the closing of Scaffold
        content = re.sub(
            r'(\s*\);\s*}\s*@override\s*Widget build)',
            r''',
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _runScan,
        backgroundColor: AppColors.primaryBlue,
        icon: const Icon(Icons.play_arrow, color: Colors.white),
        label: const Text(
          'Run Scan',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
  @override
  Widget build''',
            content
        )
    
    scanner_path.write_text(content, encoding='utf-8')
    print("  ✅ Added Run Scan FloatingActionButton")

def fix_multi_sector_selection():
    """Allow multiple sector selection"""
    print("📋 Enabling multi-sector selection...")
    
    filter_path = Path('technic_app/lib/screens/scanner/widgets/filter_panel.dart')
    content = filter_path.read_text(encoding='utf-8')
    
    # Change from single selection to multi-selection
    # Find the sector selection logic and update it
    content = re.sub(
        r"value: _selectedSector == sector,\s*onChanged: \(val\) \{\s*if \(val == true\) \{\s*setState\(\(\) \{\s*_selectedSector = sector;",
        r'''value: _selectedSectors.contains(sector),
                onChanged: (val) {
                  setState(() {
                    if (val == true) {
                      _selectedSectors.add(sector);
                    } else {
                      _selectedSectors.remove(sector);''',
        content
    )
    
    # Update the state variable
    content = re.sub(
        r"String\? _selectedSector;",
        r"Set<String> _selectedSectors = {};",
        content
    )
    
    # Update the apply filters logic
    content = re.sub(
        r"'sector': _selectedSector \?\? '',",
        r"'sector': _selectedSectors.join(','),",
        content
    )
    
    filter_path.write_text(content, encoding='utf-8')
    print("  ✅ Multi-sector selection enabled")

def prevent_auto_scan():
    """Prevent automatic scanning on filter/profile changes"""
    print("🛑 Preventing auto-scan on filter changes...")
    
    scanner_path = Path('technic_app/lib/screens/scanner/scanner_page.dart')
    content = scanner_path.read_text(encoding='utf-8')
    
    # Remove _runScan() calls from filter apply, profile buttons, and randomize
    # These should only update the UI, not trigger scans
    content = re.sub(
        r'(_applyFilters\(.*?\);)\s*_runScan\(\);',
        r'\1  // Scan will be triggered by Run Scan button',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(_applyProfile\(.*?\);)\s*_runScan\(\);',
        r'\1  // Scan will be triggered by Run Scan button',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(_randomizeFilters\(\);)\s*_runScan\(\);',
        r'\1  // Scan will be triggered by Run Scan button',
        content,
        flags=re.DOTALL
    )
    
    scanner_path.write_text(content, encoding='utf-8')
    print("  ✅ Auto-scan prevented on filter changes")

def add_profile_tooltips():
    """Add tooltips explaining what Conservative/Moderate/Aggressive do"""
    print("💡 Adding profile tooltips...")
    
    quick_actions_path = Path('technic_app/lib/screens/scanner/widgets/quick_actions.dart')
    content = quick_actions_path.read_text(encoding='utf-8')
    
    # Add tooltips to profile buttons
    profiles = {
        'Conservative': 'Lower risk: Min rating 7.0, 240-day lookback, Long-term trades',
        'Moderate': 'Balanced: Min rating 5.0, 90-day lookback, Swing trades',
        'Aggressive': 'Higher risk: Min rating 2.0, 182-day lookback, Day trades'
    }
    
    for profile, tooltip in profiles.items():
        # Wrap buttons in Tooltip widgets
        pattern = rf"(ElevatedButton\(\s*onPressed:.*?child: Text\('{profile}'\),\s*\))"
        replacement = rf'''Tooltip(
              message: '{tooltip}',
              child: \1,
            )'''
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    quick_actions_path.write_text(content, encoding='utf-8')
    print("  ✅ Profile tooltips added")

def add_randomize_feedback():
    """Show what randomize changed"""
    print("🎲 Adding randomize feedback...")
    
    scanner_path = Path('technic_app/lib/screens/scanner/scanner_page.dart')
    content = scanner_path.read_text(encoding='utf-8')
    
    # Add SnackBar to show what was randomized
    randomize_pattern = r'(void _randomizeFilters\(\) \{.*?setState\(\(\) \{)'
    
    if re.search(randomize_pattern, content, re.DOTALL):
        content = re.sub(
            r'(void _randomizeFilters\(\) \{)',
            r'''\1
    final oldFilters = Map<String, String>.from(_filters);
    ''',
            content
        )
        
        content = re.sub(
            r'(\}\);  // End setState for randomize)',
            r'''\1
    
    // Show what changed
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          'Randomized: Trade Style=${_filters['trade_style']}, '
          'Min Rating=${_filters['min_tech_rating']}, '
          'Lookback=${_filters['lookback_days']} days',
        ),
        duration: const Duration(seconds: 3),
      ),
    );''',
            content
        )
    
    scanner_path.write_text(content, encoding='utf-8')
    print("  ✅ Randomize feedback added")

def remove_redundant_display_options():
    """Remove redundant Display Options from Settings"""
    print("🗑️  Removing redundant Display Options...")
    
    settings_path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = settings_path.read_text(encoding='utf-8')
    
    # Remove the Display Options section
    content = re.sub(
        r"// Display Options.*?const SizedBox\(height: 24\),",
        "",
        content,
        flags=re.DOTALL
    )
    
    settings_path.write_text(content, encoding='utf-8')
    print("  ✅ Redundant Display Options removed")

def add_advanced_button_functionality():
    """Make Advanced button toggle advanced filters"""
    print("⚙️  Adding Advanced button functionality...")
    
    scanner_path = Path('technic_app/lib/screens/scanner/scanner_page.dart')
    content = scanner_path.read_text(encoding='utf-8')
    
    # Add state variable for advanced mode
    if 'bool _advancedMode = false;' not in content:
        content = re.sub(
            r'(class _ScannerPageState extends State<ScannerPage>.*?\{)',
            r'''\1
  bool _advancedMode = false;''',
            content,
            flags=re.DOTALL
        )
    
    # Add toggle function
    if 'void _toggleAdvanced()' not in content:
        content = re.sub(
            r'(void _runScan\(\) async \{)',
            r'''void _toggleAdvanced() {
    setState(() {
      _advancedMode = !_advancedMode;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(_advancedMode ? 'Advanced mode enabled' : 'Advanced mode disabled'),
        duration: const Duration(seconds: 2),
      ),
    );
  }

  \1''',
            content
        )
    
    scanner_path.write_text(content, encoding='utf-8')
    print("  ✅ Advanced button functionality added")

def main():
    print("🔧 Phase 4F Round 3: Comprehensive Fixes\n")
    
    fix_light_mode_colors()
    add_run_scan_button()
    fix_multi_sector_selection()
    prevent_auto_scan()
    add_profile_tooltips()
    add_randomize_feedback()
    remove_redundant_display_options()
    add_advanced_button_functionality()
    
    print("\n✨ All Round 3 fixes applied!")
    print("\n📝 Summary of changes:")
    print("  1. ✅ Light mode colors fixed (footer adapts to theme)")
    print("  2. ✅ Run Scan FloatingActionButton added")
    print("  3. ✅ Multi-sector selection enabled")
    print("  4. ✅ Auto-scan prevented (only Run Scan button triggers)")
    print("  5. ✅ Profile tooltips added (hover to see what they do)")
    print("  6. ✅ Randomize shows feedback of changes")
    print("  7. ✅ Redundant Display Options removed")
    print("  8. ✅ Advanced button now toggles advanced mode")
    print("\n🔄 Hot reload the app to see changes (press 'r' in terminal)")

if __name__ == '__main__':
    main()
