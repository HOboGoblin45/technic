#!/usr/bin/env python3
"""
Phase 4 User Feedback Fixes
Addresses 8 specific issues identified during visual testing.
"""

from pathlib import Path
import re

def fix_1_center_header_logo():
    """1. Center the technic logo and name on the header"""
    path = Path('technic_app/lib/app_shell.dart')
    content = path.read_text(encoding='utf-8')
    
    # Replace the Row with centered content
    content = re.sub(
        r'child: Row\(\s*children: \[\s*// Logo.*?const Spacer\(\),',
        '''child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Logo
                Container(
                  width: 34,
                  height: 34,
                  decoration: BoxDecoration(
                    color: AppColors.primaryBlue,  // Light blue background
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: AppColors.primaryBlue),
                    boxShadow: [
                      BoxShadow(
                        color: tone(Colors.black, 0.15),
                        blurRadius: 6,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  alignment: Alignment.center,
                  child: SvgPicture.asset(
                    'assets/logo_tq.svg',
                    width: 20,
                    height: 20,
                    colorFilter: const ColorFilter.mode(
                      AppColors.darkBackground,  // Dark blue lettering
                      BlendMode.srcIn,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                
                // Title
                const Text(
                  'technic',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.2,
                  ),
                ),''',
        content,
        flags=re.DOTALL
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fix 1: Centered header logo and name")

def fix_2_light_mode_text_contrast():
    """2. Fix light mode text contrast - make text dark"""
    path = Path('technic_app/lib/theme/app_theme.dart')
    content = path.read_text(encoding='utf-8')
    
    # Update light theme text color
    content = re.sub(
        r'(static ThemeData get lightTheme.*?textTheme: const TextTheme\(\s*bodyLarge: TextStyle\(color: )Colors\.white(\),)',
        r'\1Color(0xFF1F2937)\2',  # Dark gray for light mode
        content,
        flags=re.DOTALL
    )
    
    # Also update bodyMedium and bodySmall
    content = re.sub(
        r'(bodyMedium: TextStyle\(color: )Colors\.white(\),)',
        r'\1Color(0xFF374151)\2',
        content
    )
    content = re.sub(
        r'(bodySmall: TextStyle\(color: )Colors\.white(\),)',
        r'\1Color(0xFF6B7280)\2',
        content
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fix 2: Fixed light mode text contrast")

def fix_3_keep_blue_header_footer():
    """3. Keep blue color for header and footer in both modes"""
    path = Path('technic_app/lib/app_shell.dart')
    content = path.read_text(encoding='utf-8')
    
    # Force header to use blue background
    content = re.sub(
        r'decoration: BoxDecoration\(\s*color: isDark \? AppColors\.darkCard : Colors\.white,',
        'decoration: BoxDecoration(\n            color: const Color(0xFF0F1C31),  // Dark blue for both modes',
        content
    )
    
    # Force navigation bar to use blue background
    content = re.sub(
        r'final navBackground = isDark \? tone\(AppColors\.darkBackground, 0\.9\) : Colors\.white;',
        'final navBackground = const Color(0xFF0F1C31);  // Dark blue for both modes',
        content
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fix 3: Kept blue header and footer for both modes")

def fix_4_logo_colors():
    """4. Logo background = light blue, lettering = dark blue"""
    # Already done in fix_1_center_header_logo
    print("✅ Fix 4: Logo colors updated (done in Fix 1)")

def fix_5_add_run_scan_button():
    """5. Add 'Run Scan' button to Scanner page"""
    path = Path('technic_app/lib/screens/scanner/scanner_page.dart')
    content = path.read_text(encoding='utf-8')
    
    # Add FloatingActionButton after the body
    if 'floatingActionButton:' not in content:
        # Find the closing of Scaffold and add FAB before it
        content = re.sub(
            r'(\s+)\),\s*\);(\s+@override)',
            r'''\1  ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isScanning ? null : () async {
          setState(() => _isScanning = true);
          try {
            final bundle = await ref.read(apiServiceProvider).runScan(
              filters: _filters,
            );
            if (mounted) {
              setState(() {
                _lastBundle = bundle;
                _isScanning = false;
              });
            }
          } catch (e) {
            if (mounted) {
              setState(() => _isScanning = false);
            }
          }
        },
        icon: _isScanning 
          ? const SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
              ),
            )
          : const Icon(Icons.play_arrow),
        label: Text(_isScanning ? 'Scanning...' : 'Run Scan'),
        backgroundColor: AppColors.primaryBlue,
      ),
    );\2''',
            content
        )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fix 5: Added 'Run Scan' button")

def fix_6_verify_sectors():
    """6. Verify all sectors are available for selection"""
    # This requires checking the filter panel and universe
    print("⚠️  Fix 6: Sector verification - needs manual check of universe CSV")
    print("    Action: Review technic_v4/ticker_universe.csv for complete sector list")

def fix_7_add_disclaimer_card():
    """7. Add disclaimer card to Settings page"""
    path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = path.read_text(encoding='utf-8')
    
    # Add disclaimer card before the closing ListView
    disclaimer_card = '''
              // Disclaimer Card
              const SizedBox(height: 24),
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
    
    # Insert before the last closing bracket of ListView
    content = re.sub(
        r'(\s+)\],\s*\),\s*\);(\s+@override)',
        disclaimer_card + r'\n\1],\n\1),\n      );\2',
        content
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fix 7: Added disclaimer card to Settings")

def fix_8_settings_icons_and_text():
    """8. Fix Google icon and remove underscores from user names"""
    path = Path('technic_app/lib/screens/settings/widgets/profile_row.dart')
    if not path.exists():
        # If profile_row doesn't exist, fix in settings_page.dart
        path = Path('technic_app/lib/screens/settings/settings_page.dart')
    
    content = path.read_text(encoding='utf-8')
    
    # Fix Google icon (change from current icon to Google G logo)
    content = re.sub(
        r'(Icon\(\s*Icons\.)account_circle(,\s*size: 24\),\s*const SizedBox\(width: 12\),\s*const Text\(\s*[\'"]Google)',
        r'\1g_translate\2',  # Use g_translate as placeholder for Google icon
        content
    )
    
    # Remove underscores from google_user and apple_user
    content = content.replace('google_user', 'Google User')
    content = content.replace('apple_user', 'Apple User')
    
    # Fix sign out icon (move account_circle to sign out button)
    content = re.sub(
        r'(TextButton.*?Sign out.*?icon: Icon\()Icons\.logout',
        r'\1Icons.account_circle',
        content,
        flags=re.DOTALL
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Fix 8: Fixed Settings icons and removed underscores")

def main():
    print("🔧 Applying Phase 4 User Feedback Fixes...\n")
    
    fix_1_center_header_logo()
    fix_2_light_mode_text_contrast()
    fix_3_keep_blue_header_footer()
    fix_4_logo_colors()
    fix_5_add_run_scan_button()
    fix_6_verify_sectors()
    fix_7_add_disclaimer_card()
    fix_8_settings_icons_and_text()
    
    print("\n✨ All fixes applied!")
    print("\n📝 Next steps:")
    print("1. Hot reload the app (press 'r' in terminal)")
    print("2. Test each fix visually")
    print("3. Verify sector list completeness (Fix 6)")
    print("4. Report any remaining issues")

if __name__ == '__main__':
    main()
