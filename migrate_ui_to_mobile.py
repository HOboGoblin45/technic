"""
Migrate complete UI from technic_app to technic_mobile
Copies all screens, widgets, models, services, and theme files
"""

import os
import shutil
from pathlib import Path

# Source and destination
SOURCE = Path("technic_app/lib")
DEST = Path("technic_mobile/lib")

# Backup current technic_mobile/lib
BACKUP = Path("technic_mobile/lib_backup")

def backup_current():
    """Backup current technic_mobile/lib"""
    if DEST.exists():
        print(f"📦 Backing up current lib to {BACKUP}")
        if BACKUP.exists():
            shutil.rmtree(BACKUP)
        shutil.copytree(DEST, BACKUP)
        print("✓ Backup complete")

def copy_directory(src, dst, skip_files=None):
    """Copy directory recursively, skipping specified files"""
    skip_files = skip_files or []
    
    if not dst.exists():
        dst.mkdir(parents=True)
    
    for item in src.iterdir():
        src_path = src / item.name
        dst_path = dst / item.name
        
        # Skip files we want to keep from technic_mobile
        if item.name in skip_files:
            print(f"⏭️  Skipping {item.name} (keeping technic_mobile version)")
            continue
        
        if src_path.is_dir():
            copy_directory(src_path, dst_path, skip_files)
        else:
            shutil.copy2(src_path, dst_path)
            print(f"✓ Copied {src_path.relative_to(SOURCE)}")

def main():
    print("="*60)
    print("TECHNIC UI MIGRATION")
    print("="*60)
    print(f"Source: {SOURCE}")
    print(f"Destination: {DEST}")
    print()
    
    # Backup
    backup_current()
    print()
    
    # Files to keep from technic_mobile (backend integration)
    keep_files = [
        'services/api_config.dart',  # Keep our API config
        'services/api_client.dart',  # Keep our HTTP client
        'services/scanner_service.dart',  # Keep our scanner service
        'providers/scanner_provider.dart',  # Keep our scanner provider
        'screens/scanner_test_screen.dart',  # Keep test screen
    ]
    
    # Copy all files
    print("📁 Copying files...")
    print()
    
    # Copy main files
    for file in ['main.dart', 'app_shell.dart', 'user_profile.dart', 'watchlist_store.dart']:
        src_file = SOURCE / file
        dst_file = DEST / file
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"✓ Copied {file}")
    
    # Copy directories
    directories = ['models', 'providers', 'screens', 'services', 'theme', 'utils', 'widgets']
    
    for dir_name in directories:
        src_dir = SOURCE / dir_name
        dst_dir = DEST / dir_name
        
        if src_dir.exists():
            print(f"\n📂 Copying {dir_name}/")
            copy_directory(src_dir, dst_dir, keep_files)
    
    print()
    print("="*60)
    print("✅ MIGRATION COMPLETE!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Update pubspec.yaml with dependencies from technic_app")
    print("2. Run: flutter pub get")
    print("3. Run: flutter run -d chrome")
    print()
    print(f"Backup saved to: {BACKUP}")

if __name__ == "__main__":
    main()
