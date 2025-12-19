# Testing Quick Reference Guide
**For**: Technic Mobile App Comprehensive Testing
**Date**: December 19, 2024

---

## 🚀 Quick Start Commands

### Check Devices
```bash
cd technic_mobile
flutter devices
```

### Launch Emulator
```bash
flutter emulators --launch Medium_Phone_API_36.1
```

### Install & Run App
```bash
# Option 1: Install and run
flutter run

# Option 2: Install APK only
flutter install

# Option 3: Run with specific device
flutter run -d emulator-5554
```

### Performance Profiling
```bash
# Run in profile mode
flutter run --profile

# Open DevTools
flutter pub global run devtools
```

---

## 📱 Test Checklist (Quick Version)

### Phase 1: Installation (2 min)
- [ ] Install APK
- [ ] Launch app
- [ ] Check splash screen

### Phase 2: Critical Path (10 min)
- [ ] Login/Authentication
- [ ] Main navigation
- [ ] Data loading
- [ ] Key features work

### Phase 3: UI Components (30 min)
- [ ] Test 10 most important components
- [ ] Check animations
- [ ] Verify interactions

### Phase 4: Performance (10 min)
- [ ] Check FPS
- [ ] Monitor memory
- [ ] Test scrolling

### Phase 5: Edge Cases (10 min)
- [ ] Network errors
- [ ] Empty states
- [ ] Error handling

---

## 🎯 Priority Components to Test

### Must Test (Critical)
1. ✅ App launch
2. ✅ Authentication
3. ✅ Main navigation
4. ✅ Data fetching
5. ✅ Charts/visualizations
6. ✅ Search functionality
7. ✅ List scrolling
8. ✅ Button interactions
9. ✅ Form inputs
10. ✅ Error states

### Should Test (Important)
11. Loading states
12. Pull-to-refresh
13. Infinite scroll
14. Modals/dialogs
15. Bottom sheets
16. Theme switching
17. Filters
18. Swipe actions
19. Badges
20. Tooltips

### Nice to Test (Optional)
21-50. All other components

---

## 📊 Performance Targets

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Startup Time | <2s | <3s | >3s |
| Screen Transition | <200ms | <300ms | >300ms |
| Frame Rate | 60fps | 55fps | <50fps |
| Memory Usage | <150MB | <200MB | >200MB |
| List Scroll | Smooth | Occasional jank | Frequent jank |

---

## 🐛 Bug Severity Levels

### Critical (P0)
- App crashes
- Data loss
- Security issues
- Cannot complete core tasks

### Major (P1)
- Feature doesn't work
- Significant UX issues
- Performance problems
- Visual glitches

### Minor (P2)
- Small UI issues
- Minor inconsistencies
- Edge case bugs
- Enhancement opportunities

### Trivial (P3)
- Cosmetic issues
- Nice-to-have improvements
- Documentation updates

---

## 📝 Bug Report Template

```markdown
### Bug Title
Brief description

**Severity**: Critical/Major/Minor/Trivial
**Component**: Which component/screen
**Steps to Reproduce**:
1. Step 1
2. Step 2
3. Step 3

**Expected**: What should happen
**Actual**: What actually happens
**Screenshot**: (if applicable)
**Logs**: (if applicable)
```

---

## ✅ Test Result Codes

- ✅ **PASS**: Works as expected
- ❌ **FAIL**: Doesn't work, bug found
- ⚠️ **PARTIAL**: Works but has issues
- ⏭️ **SKIP**: Not tested
- 🚫 **BLOCKED**: Cannot test (dependency)
- 📝 **NOTE**: Observation/comment

---

## 🔧 Troubleshooting

### Emulator Won't Start
```bash
# Kill and restart
taskkill /F /IM qemu-system-x86_64.exe
flutter emulators --launch Medium_Phone_API_36.1
```

### App Won't Install
```bash
# Clean and rebuild
flutter clean
flutter pub get
flutter build apk --debug
flutter install
```

### Performance Issues
```bash
# Run in profile mode
flutter run --profile

# Enable performance overlay
# In app: Press 'P' key
```

### Logs Not Showing
```bash
# View logs
flutter logs

# Clear logs
flutter logs --clear
```

---

## 📱 Emulator Controls

### Keyboard Shortcuts
- **Power**: Power button
- **Volume Up**: Volume up
- **Volume Down**: Volume down
- **Back**: ESC or Back button
- **Home**: Home button
- **Menu**: Menu button
- **Rotate**: Ctrl+F11/F12

### Gestures
- **Tap**: Click
- **Long Press**: Click and hold
- **Swipe**: Click and drag
- **Pinch**: Ctrl + scroll

---

## 📈 Progress Tracking

### Quick Status
```
Total Tests: 200+
Completed: __/__
Passed: __/__
Failed: __/__
Time Spent: __ hours
Time Remaining: __ hours
```

### Phase Completion
```
Phase 1 (Install): [ ] 0%
Phase 2 (Auth): [ ] 0%
Phase 3 (Nav): [ ] 0%
Phase 4-18 (UI): [ ] 0%
Phase 19 (Core): [ ] 0%
Phase 20 (Perf): [ ] 0%
Phase 21 (Edge): [ ] 0%
Phase 22 (A11y): [ ] 0%
Phase 23 (Final): [ ] 0%
```

---

## 🎯 Success Criteria

### Minimum (Must Have)
- [x] APK builds
- [x] Static analysis passes
- [ ] App installs
- [ ] App launches
- [ ] No critical bugs
- [ ] Core features work

### Target (Should Have)
- [ ] All components work
- [ ] Good performance
- [ ] No major bugs
- [ ] Smooth animations

### Ideal (Nice to Have)
- [ ] Perfect performance
- [ ] No bugs at all
- [ ] Excellent UX
- [ ] Full accessibility

---

## 📞 Quick Links

- [Full Test Plan](COMPREHENSIVE_TESTING_EXECUTION_PLAN.md)
- [Build Documentation](BUILD_FIX_DOCUMENTATION.md)
- [UI Summary](UI_ENHANCEMENT_FINAL_SUMMARY.md)
- [Build Success](BUILD_SUCCESS_SUMMARY.md)

---

**Remember**: Quality over quantity. Better to test 10 things thoroughly than 50 things superficially.

---

*Quick Reference - December 19, 2024*
