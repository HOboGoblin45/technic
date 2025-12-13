"""
Fix compilation errors in copilot_page.dart
"""

def fix_copilot_page():
    file_path = "technic_app/lib/screens/copilot/copilot_page.dart"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # The errors are likely from leftover code fragments
    # Let's check lines around 228, 241, 242
    print(f"Line 228: {lines[227] if len(lines) > 227 else 'N/A'}")
    print(f"Line 241: {lines[240] if len(lines) > 240 else 'N/A'}")
    print(f"Line 242: {lines[241] if len(lines) > 241 else 'N/A'}")
    
    # Read the whole file to check for issues
    content = ''.join(lines)
    
    # Check if there are any stray references to 's.' or 'scan.'
    if 's.' in content or 'scan.' in content:
        print("\nFound references to 's.' or 'scan.' - these need to be removed")
        print("\nSearching for problematic lines...")
        for i, line in enumerate(lines, 1):
            if ('s.' in line or 'scan.' in line) and i not in [50, 51, 52, 53, 54]:  # Exclude valid uses
                print(f"Line {i}: {line.strip()}")
    
    return True

if __name__ == "__main__":
    fix_copilot_page()
