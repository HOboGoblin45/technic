# Fix File Lock and Launch Mobile App

## The Problem
VSCode's file watcher is locking the Flutter build directories, preventing the app from launching.

## The Solution (Step-by-Step)

### Step 1: Close Everything
1. **Close this VSCode window completely** (File → Exit or Alt+F4)
2. **Close all PowerShell/Terminal windows**
3. Wait 5 seconds for processes to fully terminate

### Step 2: Open Fresh PowerShell
1. Press `Windows Key`
2. Type "PowerShell"
3. Click "Windows PowerShell" (regular, not admin needed)

### Step 3: Navigate and Launch
Copy and paste these commands one at a time:

```powershell
# Navigate to project
cd C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile

# Launch the app in Chrome
flutter run -d chrome
```

### Step 4: Wait for Launch
The app will:
1. Build (30-60 seconds first time)
2. Open Chrome automatically
3. Show the splash screen
4. Load the main Scanner page

## What You'll See

### 1. Splash Screen (2-3 seconds)
- Technic logo (blue rounded square with TQ symbol)
- "technic" text
- "Quantitative Trading Companion" subtitle
- Loading progress bar

### 2. Scanner Page (Main Screen)
- **Top**: "Scanner" title with notification bell and bookmark icons
- **Welcome Card**: "Welcome to Technic!" with 3 features:
  - Quantitative Scanner
  - AI Copilot
  - Custom Profiles
- **Quick Profiles**: 3 buttons (Conservative, Moderate, Aggressive)
- **Randomize button** with shuffle icon
- **Advanced toggle** switch
- **Run Scan button** (large blue button)
- **Scan Results**: "0 opportunities" with empty state

### 3. Bottom Navigation (5 tabs)
- 📊 **Scan** (active/blue)
- 💡 **Ideas**
- 💬 **Copilot**
- ⭐ **Watchlist**
- ⚙️ **Settings**

## Testing Checklist

Once the app launches, test these:

### ✅ Basic Navigation
- [ ] Click each of the 5 bottom nav tabs
- [ ] Verify each screen loads
- [ ] Return to Scanner tab

### ✅ Scanner Page
- [ ] Click "Conservative" profile button (should highlight green)
- [ ] Click "Moderate" profile button (should highlight blue)
- [ ] Click "Aggressive" profile button (should highlight orange)
- [ ] Click "Randomize" button (should shuffle selection)
- [ ] Toggle "Advanced" switch (should show/hide advanced options)
- [ ] Click "Run Scan" button (should show progress overlay)

### ✅ Ideas Page
- [ ] Navigate to Ideas tab
- [ ] Verify "No ideas yet" empty state shows
- [ ] Check "Filter ideas" button is visible

### ✅ Copilot Page
- [ ] Navigate to Copilot tab
- [ ] Verify "Voice" and "Notes" tabs are visible
- [ ] Check quick action buttons show
- [ ] Verify text input field is present

### ✅ Watchlist Page
- [ ] Navigate to Watchlist tab
- [ ] Verify "Sign In Required" message shows
- [ ] Check "Sign In" button is visible

### ✅ Settings Page
- [ ] Navigate to Settings tab
- [ ] Verify "Sign in to unlock all features" card shows
- [ ] Check "Profile and preferences" section
- [ ] Find the dark mode toggle
- [ ] Toggle dark mode ON
- [ ] Verify colors change to dark theme
- [ ] Toggle dark mode OFF
- [ ] Verify colors return to light theme

### ✅ Theme System
- [ ] Dark mode toggle works smoothly
- [ ] Colors match the screenshots:
  - Background: Deep navy (#0A0E27)
  - Cards: Slate gray
  - Accents: Blue (#4A9EFF)
  - Text: White/light gray

### ✅ Responsive Design
- [ ] Resize Chrome window smaller
- [ ] Verify layout adapts
- [ ] Resize window larger
- [ ] Verify layout scales properly

## If It Still Doesn't Work

### Option 1: Restart Computer
1. Restart your computer
2. Open PowerShell (don't open VSCode)
3. Run the commands from Step 3 above

### Option 2: Try Web Server
```powershell
cd C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile
flutter run -d web-server
```
Then open the URL it provides in your browser.

### Option 3: Build and Run Separately
```powershell
cd C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile
flutter build web
cd build\web
python -m http.server 8080
```
Then open: http://localhost:8080

## Success Criteria

✅ App launches without errors
✅ Splash screen appears
✅ Scanner page loads with all UI elements
✅ All 5 navigation tabs work
✅ Dark mode toggle works
✅ UI matches the screenshots exactly

## After Testing

Once you've tested the app and confirmed it works:

1. Take screenshots of each screen
2. Note any issues or differences from the original
3. Come back to VSCode
4. Let me know the results

I'll be ready to fix any issues you find!

## Quick Reference

**Project Location**: `C:\Users\ccres\OneDrive\Desktop\technic-clean\technic_mobile`

**Launch Command**: `flutter run -d chrome`

**Expected Build Time**: 30-60 seconds (first time)

**Expected Result**: Chrome opens with Technic app running

---

**Remember**: Close VSCode first! The file locks won't clear until VSCode is completely closed.
