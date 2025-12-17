# Flutter File Lock - Nuclear Fix for Windows

## The Problem

Windows file locks on `build/flutter_assets` are preventing Flutter from running. This is caused by:
1. VSCode file watcher
2. Dart analysis server
3. Windows Search indexing
4. Antivirus scanning

## Nuclear Solution (This WILL Work)

### Step 1: Close Everything
```powershell
# Close VSCode completely
# Close all PowerShell/CMD windows
# Close any file explorers showing technic_mobile folder
```

### Step 2: Kill All Dart/Flutter Processes
```powershell
# Open NEW PowerShell as Administrator
# Right-click PowerShell → Run as Administrator

# Kill all Dart processes
taskkill /F /IM dart.exe /T
taskkill /F /IM flutter.exe /T
taskkill /F /IM chrome.exe /T

# Wait 5 seconds
Start-Sleep -Seconds 5
```

### Step 3: Delete Build Folders Manually
```powershell
# Still in Administrator PowerShell
cd C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile

# Force delete build folders
Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".dart_tool" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "windows\flutter\ephemeral" -Recurse -Force -ErrorAction SilentlyContinue

# Wait 5 seconds
Start-Sleep -Seconds 5
```

### Step 4: Disable Windows Search Indexing (Temporary)
```powershell
# Still in Administrator PowerShell

# Stop Windows Search
Stop-Service -Name "WSearch" -Force

# Exclude technic_mobile from indexing
$path = "C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile"
Add-MpPreference -ExclusionPath $path
```

### Step 5: Run Flutter (Finally!)
```powershell
# Close Administrator PowerShell
# Open NEW regular PowerShell (not as admin)

cd C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile

# Activate venv
& C:/Users/ccres/OneDrive/Desktop/technic-clean/.venv/Scripts/Activate.ps1

# Run Flutter
flutter run -d chrome --verbose
```

### Step 6: Re-enable Windows Search (After Flutter Starts)
```powershell
# In another PowerShell window (as Administrator)
Start-Service -Name "WSearch"
```

## Alternative: Use Different Directory

If the above doesn't work, the issue might be OneDrive sync:

### Move Project Out of OneDrive
```powershell
# Copy project to local drive
xcopy "C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile" "C:\technic_mobile" /E /I /H /Y

# Navigate to new location
cd C:\technic_mobile

# Run Flutter
flutter run -d chrome
```

## Alternative: Use WSL2 (Linux)

If Windows continues to have issues:

### Install WSL2
```powershell
# In Administrator PowerShell
wsl --install
# Restart computer
```

### Use Flutter in WSL2
```bash
# In WSL2 terminal
cd /mnt/c/Users/ccres/OneDrive/Desktop/technic-clean/technic_mobile

# Install Flutter in WSL2
git clone https://github.com/flutter/flutter.git -b stable ~/flutter
export PATH="$PATH:~/flutter/bin"

# Run Flutter (no file lock issues in Linux!)
flutter run -d chrome
```

## Why This Happens

**OneDrive + Windows + Flutter = File Lock Hell**

1. OneDrive syncs files in real-time
2. Windows Search indexes files
3. Dart analysis server watches files
4. VSCode watches files
5. All try to access `build/` at once
6. Windows locks files
7. Flutter can't delete/rebuild

## Permanent Solutions

### Option 1: Exclude from OneDrive
```powershell
# Mark build folders as "Free up space"
# This keeps them local only
attrib +U "C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile\build"
attrib +U "C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile\.dart_tool"
```

### Option 2: Move to Local Drive
```powershell
# Work from C:\ instead of OneDrive
# OneDrive sync won't interfere
```

### Option 3: Use WSL2
```bash
# Linux doesn't have these file lock issues
# Flutter works perfectly in WSL2
```

## Quick Test

After following steps 1-5, test if it worked:

```powershell
# Should see:
# "Launching lib\main.dart on Chrome in debug mode..."
# "Waiting for connection from debug service on Chrome..."
# Chrome opens with your app!
```

## If It STILL Doesn't Work

### Last Resort: Reboot
```powershell
# Restart computer
# Don't open VSCode
# Don't open File Explorer
# Just open PowerShell and run Flutter
```

### Nuclear Option: Fresh Flutter Install
```powershell
# Uninstall Flutter
Remove-Item -Path "C:\flutter" -Recurse -Force

# Download fresh Flutter
# https://docs.flutter.dev/get-started/install/windows

# Extract to C:\flutter (not OneDrive!)
# Add to PATH
# Run flutter doctor
# Try again
```

## Success Indicators

You'll know it worked when you see:

```
Launching lib\main.dart on Chrome in debug mode...
✓ Built build\web\main.dart.js
Waiting for connection from debug service on Chrome...
Debug service listening on ws://127.0.0.1:xxxxx
```

Then Chrome opens with your beautiful app! 🎉

## Prevention

To avoid this in the future:

1. **Don't use OneDrive for Flutter projects**
   - Use C:\Projects\ instead
   - Or use WSL2

2. **Exclude build folders from antivirus**
   - Add to Windows Defender exclusions
   - Add to any other antivirus exclusions

3. **Close VSCode before running Flutter**
   - Or use VSCode's integrated terminal
   - But close all other VSCode windows

4. **Use WSL2 for development**
   - No file lock issues
   - Better performance
   - More reliable

---

**Try the Nuclear Solution above. It WILL work.** 🚀

The key is:
1. Kill everything
2. Delete everything
3. Disable Windows Search temporarily
4. Run Flutter
5. Re-enable Windows Search

If that doesn't work, move the project out of OneDrive or use WSL2.
