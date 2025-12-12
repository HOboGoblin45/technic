#!/usr/bin/env python3
"""Fix f-string syntax error in technic_app.py"""

from pathlib import Path
import re

def fix_fstring_syntax():
    """Fix the f-string syntax error at line 5001"""
    print("🔧 Fixing Streamlit f-string syntax error...")
    
    app_path = Path('technic_v4/ui/technic_app.py')
    content = app_path.read_text(encoding='utf-8')
    
    # Find and fix the problematic f-string
    # The issue is: f"{bt["win_rate"]:.1f}%"
    # Should be: f"{bt['win_rate']:.1f}%"
    
    # Replace all instances of bt["..."] in f-strings with bt['...']
    content = re.sub(
        r'f"([^"]*)\{bt\["([^"]+)"\]([^}]*)\}',
        r"f'\1{bt['\2']\3}",
        content
    )
    
    app_path.write_text(content, encoding='utf-8')
    print("✅ Fixed f-string syntax errors")
    print("🔄 Streamlit should auto-reload")

if __name__ == '__main__':
    fix_fstring_syntax()
