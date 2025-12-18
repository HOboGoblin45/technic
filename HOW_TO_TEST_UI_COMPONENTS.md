# How to Test UI Components

## Quick Access to Test Screen

The UI test screen has been created at `technic_mobile/lib/screens/ui_test_screen.dart` and showcases all Phase 1 premium components.

### Option 1: Temporary Navigation (Quickest)

Add this temporary button to any existing screen to navigate to the test screen:

```dart
// Add this import at the top
import 'package:technic_mobile/screens/ui_test_screen.dart';

// Add this button anywhere in your UI
FloatingActionButton(
  onPressed: () {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const UITestScreen()),
    );
  },
  child: const Icon(Icons.science),
)
```

### Option 2: Add to Settings Page

1. Open `technic_mobile/lib/screens/settings/settings_page.dart`
2. Add a "UI Components Test" option in the settings list
3. Navigate to `UITestScreen()` when tapped

### Option 3: Replace Home Screen Temporarily

In `technic_mobile/lib/main.dart`, temporarily change the home to the test screen:

```dart
// Change this line:
home: _showSplash
    ? SplashScreen(onComplete: _onSplashComplete)
    : const TechnicShell(),

// To this:
home: const UITestScreen(),
```

## What to Test

### 1. Premium Buttons ✅
- **Animations**: Press any button and observe the smooth scale animation (0.95x)
- **Haptic Feedback**: Feel the light vibration on tap (on physical devices)
- **Variants**: Test all 6 variants (Primary, Secondary, Outline, Text, Success, Danger)
- **Sizes**: Verify all 3 sizes render correctly (Small, Medium, Large)
- **Loading State**: Tap "Loading State" button and watch the spinner
- **Disabled State**: Verify disabled button is grayed out and non-interactive
- **Icons**: Check that icons render correctly with text

### 2. Premium Cards ✅
- **Glass Morphism**: Verify the frosted glass effect with backdrop blur
- **Press Animation**: Tap cards and observe subtle scale effect (0.98x)
- **Variants**: Test all 4 variants (Glass, Elevated, Gradient, Outline)
- **Shadows**: Check that elevated cards have proper shadows
- **Gradients**: Verify gradient card shows smooth color transition

### 3. Stock Result Cards ✅
- **Layout**: Verify symbol, company name, price, and change display correctly
- **Color Coding**: Green for positive changes, red for negative
- **Rating Pills**: Check Tech Rating and Merit Score pills render with colors
- **Interaction**: Tap cards and verify snackbar appears

### 4. Metric Cards ✅
- **Grid Layout**: Verify 2-column grid displays correctly
- **Icons**: Check colored icon backgrounds
- **Values**: Large numbers should be prominent
- **Subtitles**: Verify subtitle text appears below values

### 5. Color Palette ✅
- **Vibrant Colors**: Verify neon green (#00FF88) and bright red (#FF3B5C) are vivid
- **Technic Blue**: Confirm brand color (#4A9EFF) is prominent
- **Glow Effects**: Check that color swatches have subtle glow shadows
- **Contrast**: Ensure all colors are readable against dark background

### 6. Gradients ✅
- **Smooth Transitions**: Verify all 5 gradients show smooth color blending
- **Primary Gradient**: Technic blue gradient
- **Success Gradient**: Neon green gradient
- **Danger Gradient**: Bright red gradient
- **Card Gradient**: Subtle depth gradient
- **Premium Gradient**: Purple gradient

### 7. Floating Action Button ✅
- **Animation**: Tap FAB and observe scale animation
- **Haptic**: Feel vibration on tap
- **Counter**: Verify counter increments and snackbar shows

## Performance Checks

### Frame Rate
- Scroll through the entire screen
- All animations should be smooth at 60fps
- No stuttering or jank

### Memory
- Open and close the screen multiple times
- No memory leaks from animations
- App should remain responsive

### Responsiveness
- Test on different screen sizes if possible
- All components should scale appropriately
- Touch targets should be at least 48x48px

## Expected Results

✅ **Buttons**:
- Smooth scale animation on press
- Haptic feedback (on physical devices)
- Gradient backgrounds with glow shadows
- Loading spinner animates smoothly
- Disabled state is clearly visible

✅ **Cards**:
- Glass morphism shows frosted effect
- Press animation is subtle and smooth
- Shadows render correctly
- Gradients are smooth

✅ **Colors**:
- Neon green is vibrant and eye-catching
- Bright red is attention-grabbing
- Technic blue is prominent throughout
- All colors have good contrast

✅ **Performance**:
- 60fps throughout
- No lag or stuttering
- Smooth scrolling
- Quick response to taps

## Common Issues & Solutions

### Issue: Animations are choppy
**Solution**: Run in Release mode (`flutter run --release`) for best performance

### Issue: Haptic feedback not working
**Solution**: Haptic only works on physical devices, not simulators

### Issue: Colors look different
**Solution**: Ensure dark mode is enabled (app defaults to dark theme)

### Issue: Glass morphism not visible
**Solution**: Backdrop blur requires proper rendering context, may not show in some emulators

## Next Steps After Testing

Once you've verified all components work correctly:

1. **Document any issues found**
2. **Take screenshots of the components**
3. **Proceed to Phase 2**: Integrate these components into actual app screens
4. **Remove test screen**: Delete `ui_test_screen.dart` after Phase 2 is complete

## Quick Test Checklist

- [ ] All button variants render correctly
- [ ] Button press animations are smooth
- [ ] Haptic feedback works (on device)
- [ ] Loading state shows spinner
- [ ] Disabled state is visible
- [ ] Glass cards show frosted effect
- [ ] Card press animations work
- [ ] Stock cards display all information
- [ ] Metric cards show in grid
- [ ] Color palette shows vibrant colors
- [ ] All 5 gradients render smoothly
- [ ] FAB animation works
- [ ] No performance issues
- [ ] No visual glitches
- [ ] Scrolling is smooth

---

**Estimated Testing Time**: 15-20 minutes
**Status**: Ready to test
**Next**: Phase 2 - Core Screens Enhancement
