#!/usr/bin/env python3
"""
Remove gradients and flatten hero banners for institutional design.
Keep sparkline gradient (data visualization is acceptable).
"""

from pathlib import Path
import re

def flatten_app_shell():
    """Flatten app_shell.dart - remove header and body gradients."""
    path = Path('technic_app/lib/app_shell.dart')
    content = path.read_text(encoding='utf-8')
    
    # Replace header gradient with solid color
    content = re.sub(
        r'decoration: BoxDecoration\(\s*gradient: LinearGradient\(\s*colors: headerGradient,.*?\),\s*\),',
        'decoration: BoxDecoration(\n            color: isDark ? AppColors.darkCard : Colors.white,\n            border: Border(bottom: BorderSide(color: isDark ? AppColors.darkBorder : const Color(0xFFE5E7EB))),\n          ),',
        content,
        flags=re.DOTALL
    )
    
    # Replace body gradient with solid color
    content = re.sub(
        r'body: Container\(\s*decoration: BoxDecoration\(\s*gradient: LinearGradient\(\s*colors: bodyGradient,.*?\),\s*\),',
        'body: Container(\n        color: isDark ? AppColors.darkBackground : const Color(0xFFF9FAFB),',
        content,
        flags=re.DOTALL
    )
    
    # Remove unused gradient variables
    content = re.sub(
        r'final headerGradient = isDark.*?\];',
        '',
        content,
        flags=re.DOTALL
    )
    content = re.sub(
        r'final bodyGradient = isDark.*?\];',
        '',
        content,
        flags=re.DOTALL
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Flattened: app_shell.dart")

def flatten_hero_banners():
    """Flatten hero banners in settings, ideas, and copilot pages."""
    files = [
        'technic_app/lib/screens/settings/settings_page.dart',
        'technic_app/lib/screens/ideas/ideas_page.dart',
        'technic_app/lib/screens/copilot/copilot_page.dart',
    ]
    
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  Not found: {filepath}")
            continue
            
        content = path.read_text(encoding='utf-8')
        
        # Replace gradient in _buildHeroBanner with solid color + border
        content = re.sub(
            r'decoration: BoxDecoration\(\s*gradient: LinearGradient\(\s*colors: \[.*?\],.*?\),\s*borderRadius:',
            'decoration: BoxDecoration(\n        color: tone(AppColors.darkCard, 0.5),\n        borderRadius:',
            content,
            flags=re.DOTALL
        )
        
        # Reduce shadow blur from 18 to 6
        content = content.replace('blurRadius: 18,', 'blurRadius: 6,')
        
        # Reduce shadow opacity
        content = content.replace('tone(Colors.black, 0.35)', 'tone(Colors.black, 0.15)')
        
        path.write_text(content, encoding='utf-8')
        print(f"✅ Flattened: {filepath}")

def flatten_card_widgets():
    """Flatten gradient cards in scanner widgets."""
    files = [
        'technic_app/lib/screens/scanner/widgets/scoreboard_card.dart',
        'technic_app/lib/screens/scanner/widgets/onboarding_card.dart',
        'technic_app/lib/screens/scanner/widgets/market_pulse_card.dart',
    ]
    
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  Not found: {filepath}")
            continue
            
        content = path.read_text(encoding='utf-8')
        
        # Replace gradient with solid color
        content = re.sub(
            r'decoration: BoxDecoration\(\s*gradient: LinearGradient\(\s*colors: \[.*?\],.*?\),',
            'decoration: BoxDecoration(\n        color: tone(AppColors.darkCard, 0.5),',
            content,
            flags=re.DOTALL
        )
        
        # Reduce shadows
        content = content.replace('blurRadius: 18,', 'blurRadius: 6,')
        content = content.replace('blurRadius: 12,', 'blurRadius: 6,')
        content = content.replace('tone(Colors.black, 0.35)', 'tone(Colors.black, 0.15)')
        content = content.replace('tone(Colors.black, 0.25)', 'tone(Colors.black, 0.1)')
        
        path.write_text(content, encoding='utf-8')
        print(f"✅ Flattened: {filepath}")

def remove_settings_gradient():
    """Remove the theme gradient in settings page."""
    path = Path('technic_app/lib/screens/settings/settings_page.dart')
    content = path.read_text(encoding='utf-8')
    
    # Replace the theme preview gradient with solid color
    content = re.sub(
        r'decoration: BoxDecoration\(\s*borderRadius: BorderRadius\.circular\(12\),\s*gradient: const LinearGradient\(\s*colors: \[AppColors\.successGreen, Color\(0xFF5EEAD4\)\],.*?\),\s*\),',
        'decoration: BoxDecoration(\n                  borderRadius: BorderRadius.circular(12),\n                  color: AppColors.primaryBlue,\n                ),',
        content,
        flags=re.DOTALL
    )
    
    path.write_text(content, encoding='utf-8')
    print("✅ Removed theme gradient: settings_page.dart")

def main():
    print("🎨 Flattening gradients for institutional design...\n")
    
    flatten_app_shell()
    flatten_hero_banners()
    flatten_card_widgets()
    remove_settings_gradient()
    
    print("\n✨ Complete! All gradients flattened (except sparkline data viz).")
    print("📝 Shadows reduced from 18px to 6px blur.")
    print("🎯 Hero banners now use solid colors with subtle borders.")

if __name__ == '__main__':
    main()
