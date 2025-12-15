"""Fix indentation error in scanner_core.py line 1810"""

from pathlib import Path

def fix_indentation():
    scanner_path = Path("technic_v4/scanner_core.py")
    
    if not scanner_path.exists():
        print(f"❌ Error: {scanner_path} not found")
        return False
    
    content = scanner_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    # Find and fix the indentation issue around line 1810
    for i, line in enumerate(lines):
        if 'results_df = results_df[results_df["DollarVolume"] >= MIN_DOLLAR_VOL]' in line:
            # Check if it's incorrectly indented
            if line.startswith('        results_df'):
                # Fix: should have 8 spaces (2 levels of indentation)
                lines[i] = '        ' + line.lstrip()
                print(f"✅ Fixed indentation at line {i+1}")
            elif line.startswith('    results_df'):
                # Already correct
                print(f"✅ Line {i+1} already has correct indentation")
            else:
                # Ensure it has proper indentation (8 spaces)
                lines[i] = '        results_df = results_df[results_df["DollarVolume"] >= MIN_DOLLAR_VOL]'
                print(f"✅ Fixed indentation at line {i+1}")
    
    # Write back
    scanner_path.write_text('\n'.join(lines), encoding='utf-8')
    print("✅ File updated successfully")
    return True

if __name__ == "__main__":
    fix_indentation()
