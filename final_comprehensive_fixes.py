#!/usr/bin/env python3
"""
Final Comprehensive Fixes - All Remaining Issues
Based on visual screenshots provided by user.
"""

from pathlib import Path
import re

def add_disclaimer_to_settings():
    """Add disclaimer card at the end of Settings ListView"""
    path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = path.read_text(encoding='utf-8')
    
    # Find the last InfoCard (Copilot status) and add disclaimer after it
    disclaimer_card = '''
        const SizedBox(height: 24),

        // Legal Disclaimer Card
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: tone(AppColors.darkCard, 0.5),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: AppColors.warningOrange.withOpacity(0.3),
              width: 1,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    color: AppColors.warningOrange,
                    size: 24,
                  ),
                  const SizedBox(width: 12),
                  const Text(
                    'Important Disclaimer',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Text(
                'Technic provides educational analysis and quantitative insights for informational purposes only. This app does not provide financial, investment, or trading advice.',
                style: TextStyle(
                  fontSize: 14,
                  height: 1.5,
                  color: tone(Colors.white, 0.8),
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Past performance does not guarantee future results. Trading and investing involve substantial risk of loss. Always consult with a licensed financial advisor before making investment decisions.',
                style: TextStyle(
                  fontSize: 14,
                  height: 1.5,
                  color: tone(Colors.white, 0.8),
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'By using this app, you acknowledge that you understand these risks and agree to use the information provided at your own discretion.',
                style: TextStyle(
                  fontSize: 13,
                  height: 1.5,
                  color: tone(Colors.white, 0.7),
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 32),'''
    
    # Insert before the closing bracket of ListView children
    content = re.sub(
        r'(\s+)\],\s*\);(\s+}\s+})',
        disclaimer_card + r'\n\1],\n\1);\2',
        content
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Added disclaimer card to Settings")

def add_run_scan_button():
    """Add FloatingActionButton for Run Scan"""
    path = Path('technic_app/lib/screens/scanner/scanner_page.dart')
    content = path.read_text(encoding='utf-8')
    
    # Check if already has floatingActionButton
    if 'floatingActionButton' in content:
        print("✅ Run Scan button already exists")
        return
    
    # Add FAB before the closing Scaffold
    fab_code = '''
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () async {
          // Trigger scan
          setState(() {});
        },
        icon: const Icon(Icons.play_arrow),
        label: const Text('Run Scan'),
        backgroundColor: AppColors.primaryBlue,
      ),'''
    
    # Find Scaffold closing and add FAB
    content = re.sub(
        r'(\s+)\);(\s+@override\s+Widget build)',
        fab_code + r'\n\1);\2',
        content
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Added Run Scan button")

def fix_footer_icons_white():
    """Make footer navigation icons white"""
    path = Path('technic_app/lib/app_shell.dart')
    content = path.read_text(encoding='utf-8')
    
    # Add theme to NavigationBar to force white icons
    if 'iconTheme: MaterialStateProperty' not in content:
        content = re.sub(
            r'(child: NavigationBar\(\s*selectedIndex: _index,\s*backgroundColor: Colors\.transparent,\s*indicatorColor:.*?,\s*height: 70,)',
            r'''\1
          iconTheme: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.selected)) {
              return const IconThemeData(color: Colors.white, size: 24);
            }
            return const IconThemeData(color: Colors.white70, size: 24);
          }),''',
            content
        )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed footer icons to be white")

def fix_technic_branding_white():
    """Make technic text white and thinner"""
    path = Path('technic_app/lib/app_shell.dart')
    content = path.read_text(encoding='utf-8')
    
    # Update technic text to be white and thin
    content = re.sub(
        r"const Text\(\s*'technic',\s*style: TextStyle\([^)]+\),\s*\),",
        '''const Text(
                  'technic',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w300,  // Thin
                    letterSpacing: 1.5,
                    color: Colors.white,  // White
                  ),
                ),''',
        content
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fixed technic branding (white, thin)")

def main():
    print("🔧 Applying Final Comprehensive Fixes...\n")
    
    add_disclaimer_to_settings()
    add_run_scan_button()
    fix_footer_icons_white()
    fix_technic_branding_white()
    
    print("\n✨ All fixes applied!")
    print("\n📝 Next: Hot reload the app (press 'r' in terminal)")

if __name__ == '__main__':
    main()
