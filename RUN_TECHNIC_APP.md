# 🚀 How to Run Technic App

## Quick Start Commands

### Option 1: Run on Windows Desktop
```powershell
cd technic_app
flutter run -d windows
```

### Option 2: Run on Android Emulator
```powershell
cd technic_app
flutter run -d emulator
```

### Option 3: Run on Chrome (Web)
```powershell
cd technic_app
flutter run -d chrome
```

### Option 4: Auto-select Device
```powershell
cd technic_app
flutter run
```

---

## Step-by-Step Instructions

### 1. Check Available Devices
```powershell
cd technic_app
flutter devices
```

This will show you all available devices:
- Windows (windows)
- Chrome (chrome)
- Android emulator (if running)
- Connected Android device (if connected)

### 2. Run on Specific Device
```powershell
# Replace <device-id> with the device ID from step 1
flutter run -d <device-id>
```

### 3. Hot Reload During Development
Once the app is running:
- Press `r` to hot reload
- Press `R` to hot restart
- Press `q` to quit

---

## What You'll See

### 1. Splash Screen (First 2.5 seconds)
✅ Logo with light blue background  
✅ "technic" text in thin font  
✅ "Quantitative Trading Companion" tagline  
✅ Loading animation  
✅ Smooth fade-in effect  

### 2. Main App
✅ Scanner page loads  
✅ Bottom navigation bar  
✅ All features accessible  

### 3. During Scan
✅ Progress overlay appears  
✅ Real-time ETA updates every second  
✅ Universe count displays (e.g., "Universe: 3,085 tickers")  
✅ Progress percentage with decimal (e.g., "75.3%")  

---

## Troubleshooting

### Issue: "No devices found"
**Solution:** Start an emulator or connect a device
```powershell
# List available emulators
flutter emulators

# Launch an emulator
flutter emulators --launch <emulator-id>
```

### Issue: Build errors
**Solution:** Clean and rebuild
```powershell
cd technic_app
flutter clean
flutter pub get
flutter run
```

### Issue: Splash screen doesn't show
**Solution:** Make sure you're running the latest code
```powershell
git pull origin main
cd technic_app
flutter run
```

---

## Testing Checklist

### Splash Screen:
- [ ] Appears on app startup
- [ ] Logo displays correctly (light blue bg, dark blue lettering)
- [ ] "technic" text is thin and properly spaced
- [ ] Animations are smooth
- [ ] Auto-dismisses after 2.5 seconds
- [ ] Transitions smoothly to main app

### Scan Progress Overlay:
- [ ] Shows during scan
- [ ] ETA updates every second
- [ ] Universe count displays correctly
- [ ] Progress percentage shows decimal (e.g., "75.3%")
- [ ] Time formatting is correct (sec, m/s, h/m)
- [ ] Symbols scanned updates in real-time

---

## Quick Command Reference

```powershell
# Navigate to app directory
cd technic_app

# Check Flutter setup
flutter doctor

# List available devices
flutter devices

# Run on Windows
flutter run -d windows

# Run on Chrome
flutter run -d chrome

# Run on Android
flutter run -d emulator

# Clean build
flutter clean

# Get dependencies
flutter pub get

# Analyze code
flutter analyze

# Run tests
flutter test
```

---

## 🎉 Ready to Test!

Run this command to start the app:
```powershell
cd technic_app
flutter run -d windows
```

Then watch for:
1. **Splash screen** (2.5 seconds) - Professional branded loading
2. **Main app** - Scanner page with all features
3. **Scan progress** - Real-time ETA and universe count during scans

Enjoy your improved Technic app! 🚀
