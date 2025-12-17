"""
Script to copy technic_app to technic_mobile systematically
This automates the file copying process for the mobile app migration
"""

import os
import shutil
from pathlib import Path

# Define source and destination
SOURCE = Path("technic_app/lib")
DEST = Path("technic_mobile/lib")

# Directories to copy
DIRS_TO_COPY = [
    "models",
    "services", 
    "providers",
    "utils",
    "widgets",
    "screens/scanner",
    "screens/watchlist",
    "screens/settings",
    "screens/symbol_detail",
    "screens/auth",
    "screens/history",
    "screens/copilot",
    "screens/ideas",
    "screens/my_ideas",
    "screens/onboarding",
    "screens/splash",
]

# Files to skip (already exist or will be manually merged)
SKIP_FILES = [
    "main.dart",  # Already customized
    "theme/app_theme.dart",  # Already corrected
]

def copy_directory(src_dir, dest_dir, skip_files=None):
    """Copy directory contents, creating destination if needed"""
    skip_files = skip_files or []
    
    src_path = SOURCE / src_dir
    dest_path = DEST / src_dir
    
    if not src_path.exists():
        print(f"⚠️  Source not found: {src_path}")
        return 0
    
    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    
    files_copied = 0
    
    # Copy all files in directory
    for item in src_path.rglob("*"):
        if item.is_file():
            # Get relative path
            rel_path = item.relative_to(SOURCE)
            
            # Check if should skip
            if any(str(rel_path).endswith(skip) for skip in skip_files):
                print(f"⏭️  Skipping: {rel_path}")
                continue
            
            # Destination file
            dest_file = DEST / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(item, dest_file)
            print(f"✅ Copied: {rel_path}")
            files_copied += 1
    
    return files_copied

def main():
    print("="*60)
    print("COPYING TECHNIC_APP TO TECHNIC_MOBILE")
    print("="*60)
    print()
    
    total_files = 0
    
    for dir_name in DIRS_TO_COPY:
        print(f"\n📁 Copying {dir_name}...")
        print("-"*60)
        count = copy_directory(dir_name, SKIP_FILES)
        total_files += count
        print(f"   {count} files copied from {dir_name}")
    
    print()
    print("="*60)
    print(f"✅ COMPLETE: {total_files} files copied")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Run: cd technic_mobile && flutter pub get")
    print("2. Run: cd technic_mobile && flutter analyze")
    print("3. Fix any compilation errors")
    print("4. Test the app")

if __name__ == "__main__":
    main()
