# Features 10 & 11: Implementation Plan

**Features:** Watchlist Alerts & Onboarding Flow  
**Estimated Time:** 3 hours total  
**Status:** Planning Phase

---

## 🎯 FEATURE 10: WATCHLIST ALERTS

**Estimated Time:** 2 hours  
**Priority:** High  
**Complexity:** Medium

### **Overview:**
Allow users to set price alerts on watchlist items. Get notified when a symbol reaches a target price.

### **Requirements:**
1. Add alert configuration to watchlist items
2. Alert types: Price Above, Price Below, Percent Change
3. Alert notification system
4. Alert management UI
5. Persistence of alerts
6. Alert status tracking (active/triggered)

### **Implementation Plan:**

#### **Step 1: Data Model (15 min)**
**File:** `technic_app/lib/models/price_alert.dart`
```dart
class PriceAlert {
  final String id;
  final String ticker;
  final AlertType type;
  final double targetValue;
  final bool isActive;
  final DateTime createdAt;
  final DateTime? triggeredAt;
  
  // Methods: toJson, fromJson, copyWith
}

enum AlertType {
  priceAbove,
  priceBelow,
  percentChange,
}
```

#### **Step 2: Alert Provider (30 min)**
**File:** `technic_app/lib/providers/alert_provider.dart`
- CRUD operations for alerts
- Check alert conditions
- Trigger notifications
- Persistence through StorageService

#### **Step 3: Storage Integration (15 min)**
**File:** `technic_app/lib/services/storage_service.dart`
- Add `loadAlerts()` method
- Add `saveAlerts()` method

#### **Step 4: Alert Dialog UI (30 min)**
**File:** `technic_app/lib/screens/watchlist/widgets/add_alert_dialog.dart`
- Alert type selector
- Target value input
- Validation
- Save/Cancel buttons

#### **Step 5: Watchlist Integration (20 min)**
**File:** `technic_app/lib/screens/watchlist/watchlist_page.dart`
- Add "Set Alert" button to watchlist items
- Show alert indicators on cards
- Alert management options

#### **Step 6: Alert Checking Service (10 min)**
**File:** `technic_app/lib/services/alert_service.dart`
- Background alert checking
- Notification triggering
- Alert status updates

---

## 🎯 FEATURE 11: ONBOARDING FLOW

**Estimated Time:** 1 hour  
**Priority:** Medium  
**Complexity:** Low

### **Overview:**
Welcome new users with a guided onboarding experience explaining key features.

### **Requirements:**
1. Welcome screen
2. Feature highlights (3-4 screens)
3. Skip option
4. "Get Started" button
5. Show only on first launch
6. Persistence of onboarding status

### **Implementation Plan:**

#### **Step 1: Onboarding Model (10 min)**
**File:** `technic_app/lib/models/onboarding_page.dart`
```dart
class OnboardingPage {
  final String title;
  final String description;
  final IconData icon;
  final Color color;
}
```

#### **Step 2: Onboarding Screen (30 min)**
**File:** `technic_app/lib/screens/onboarding/onboarding_screen.dart`
- PageView with 4 pages
- Skip button
- Next/Get Started buttons
- Smooth animations
- Beautiful design

#### **Step 3: Storage Integration (10 min)**
**File:** `technic_app/lib/services/storage_service.dart`
- Add `hasCompletedOnboarding()` method
- Add `setOnboardingComplete()` method

#### **Step 4: App Integration (10 min)**
**File:** `technic_app/lib/main.dart`
- Check onboarding status on launch
- Show onboarding if first time
- Navigate to main app after completion

---

## 📋 IMPLEMENTATION ORDER

### **Phase 1: Feature 10 - Watchlist Alerts (2 hours)**
1. ✅ Create data model (15 min)
2. ✅ Create alert provider (30 min)
3. ✅ Add storage methods (15 min)
4. ✅ Create alert dialog UI (30 min)
5. ✅ Integrate with watchlist (20 min)
6. ✅ Create alert service (10 min)

### **Phase 2: Feature 11 - Onboarding (1 hour)**
1. ✅ Create onboarding model (10 min)
2. ✅ Create onboarding screen (30 min)
3. ✅ Add storage methods (10 min)
4. ✅ Integrate with main app (10 min)

---

## 🎨 DESIGN SPECIFICATIONS

### **Feature 10: Alert Dialog**
```
┌─────────────────────────────────┐
│ Set Price Alert                 │
├─────────────────────────────────┤
│ Symbol: AAPL                    │
│                                 │
│ Alert Type:                     │
│ ○ Price Above                   │
│ ○ Price Below                   │
│ ○ Percent Change                │
│                                 │
│ Target Value:                   │
│ [___________]                   │
│                                 │
│ [Cancel]  [Set Alert]           │
└─────────────────────────────────┘
```

### **Feature 11: Onboarding Pages**

**Page 1: Welcome**
- Icon: 🎯
- Title: "Welcome to Technic"
- Description: "Your AI-powered trading companion"

**Page 2: Scanner**
- Icon: 🔍
- Title: "Smart Scanner"
- Description: "Find high-quality stocks with MERIT scoring"

**Page 3: Watchlist**
- Icon: ⭐
- Title: "Organize Your Ideas"
- Description: "Track symbols with notes, tags, and alerts"

**Page 4: Get Started**
- Icon: 🚀
- Title: "Ready to Begin"
- Description: "Start scanning and building your watchlist"

---

## 🔧 TECHNICAL DETAILS

### **Alert Checking Strategy:**
- Check alerts when app is active
- Use background task for periodic checks
- Trigger local notifications
- Update alert status in real-time

### **Onboarding Storage:**
- Use SharedPreferences
- Key: `has_completed_onboarding`
- Value: boolean

### **Dependencies:**
- No new dependencies required
- Use existing Flutter/Riverpod
- Use existing storage service

---

## ✅ SUCCESS CRITERIA

### **Feature 10:**
- [ ] Users can create price alerts
- [ ] Alerts trigger correctly
- [ ] Notifications work
- [ ] Alert management UI functional
- [ ] Persistence working
- [ ] No performance issues

### **Feature 11:**
- [ ] Onboarding shows on first launch
- [ ] All pages display correctly
- [ ] Skip button works
- [ ] Get Started navigates to app
- [ ] Onboarding doesn't show again
- [ ] Beautiful animations

---

## 📊 PROGRESS TRACKING

**Feature 10: Watchlist Alerts**
- [ ] Step 1: Data Model
- [ ] Step 2: Alert Provider
- [ ] Step 3: Storage Integration
- [ ] Step 4: Alert Dialog UI
- [ ] Step 5: Watchlist Integration
- [ ] Step 6: Alert Service

**Feature 11: Onboarding Flow**
- [ ] Step 1: Onboarding Model
- [ ] Step 2: Onboarding Screen
- [ ] Step 3: Storage Integration
- [ ] Step 4: App Integration

---

## 🚀 LET'S BEGIN!

Starting with Feature 10: Watchlist Alerts...
