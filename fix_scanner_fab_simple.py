#!/usr/bin/env python3
"""
Simple fix to add FloatingActionButton to scanner_page.dart
"""

from pathlib import Path

def add_fab_to_scanner():
    """Add FloatingActionButton to Scanner page"""
    path = Path('technic_app/lib/screens/scanner/scanner_page.dart')
    content = path.read_text(encoding='utf-8')
    
    # Find the closing of the Scaffold's body and add FAB before the final closing
    # Look for the pattern: closing of CustomScrollView, then closing of RefreshIndicator, then closing of Scaffold
    
    # Find the last occurrence of the Scaffold closing
    lines = content.split('\n')
    
    # Find where to insert the FAB (before the last closing brace of Scaffold)
    insert_index = -1
    brace_count = 0
    scaffold_found = False
    
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line == '}' and not scaffold_found:
            # This might be the closing of the build method
            continue
        if line == ');' and 'Scaffold' in ''.join(lines[max(0, i-50):i]):
            insert_index = i
            break
    
    if insert_index > 0:
        # Insert the FAB before the Scaffold closing
        fab_code = '''      floatingActionButton: FloatingActionButton.extended(
        onPressed: _refresh,
        backgroundColor: AppColors.primaryBlue,
        icon: const Icon(Icons.play_arrow, color: Colors.white),
        label: const Text(
          'Run Scan',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),'''
        
        lines.insert(insert_index, fab_code)
        content = '\n'.join(lines)
        path.write_text(content, encoding='utf-8')
        print("✅ Added FloatingActionButton to Scanner page")
    else:
        print("❌ Could not find insertion point")

if __name__ == '__main__':
    add_fab_to_scanner()
