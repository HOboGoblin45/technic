# API Authentication Strategy & Flutter Integration

## 🔐 Is Removing API Key Permanent?

**No, it's not permanent - you have full control!**

### **Three Deployment Modes:**

#### **1. Dev Mode (No Auth) - Current Recommendation**
- **Setup:** Remove `TECHNIC_API_KEY` from Render
- **Use Case:** Development, testing, internal use
- **Security:** Anyone with the URL can access
- **Best For:** Testing your Flutter app during development

#### **2. Production Mode (With Auth)**
- **Setup:** Set `TECHNIC_API_KEY` in Render environment
- **Use Case:** Public release, paid users
- **Security:** Only requests with valid API key can access
- **Best For:** When you launch to real users

#### **3. Hybrid Mode (Selective Auth)**
- **Setup:** Modify code to require auth only for certain endpoints
- **Use Case:** Free tier + paid tier
- **Security:** Some endpoints public, others require auth

---

## 🎯 Recommended Strategy

### **Phase 1: Development (Now)**
```
Remove TECHNIC_API_KEY → Dev Mode
```
- ✅ Easy testing
- ✅ Flutter app works immediately
- ✅ No auth complexity
- ⚠️ Not secure for public use

### **Phase 2: Beta Testing**
```
Keep Dev Mode OR Add simple API key
```
- Share with trusted users
- Monitor usage
- Gather feedback

### **Phase 3: Production Launch**
```
Add TECHNIC_API_KEY → Production Mode
```
- Secure API
- Control access
- Track usage per user
- Enable paid tiers

---

## 📱 How to Test Render with Flutter UI

### **Option 1: Point Flutter to Render API (Recommended)**

#### **1. Find Your API Base URL Config in Flutter**

Look for a file like:
- `lib/config/api_config.dart`
- `lib/services/api_service.dart`
- `lib/constants.dart`

#### **2. Update the Base URL**

```dart
// Before (local)
const String API_BASE_URL = 'http://localhost:8502';

// After (Render)
const String API_BASE_URL = 'https://technic-m5vn.onrender.com';
```

#### **3. Remove API Key Requirement (if present)**

If your Flutter app is sending an API key:

```dart
// Before
final headers = {
  'Content-Type': 'application/json',
  'X-API-Key': 'some-key',  // ← Remove this line
};

// After
final headers = {
  'Content-Type': 'application/json',
};
```

#### **4. Test Your Flutter App**

```bash
# Run Flutter app
cd technic_app
flutter run
```

Your Flutter app will now call the Render API!

---

### **Option 2: Environment-Based Configuration (Best Practice)**

Create different configs for dev/prod:

#### **1. Create `lib/config/environment.dart`**

```dart
enum Environment { development, production }

class EnvironmentConfig {
  static Environment _environment = Environment.development;
  
  static String get apiBaseUrl {
    switch (_environment) {
      case Environment.development:
        return 'http://localhost:8502';
      case Environment.production:
        return 'https://technic-m5vn.onrender.com';
    }
  }
  
  static void setEnvironment(Environment env) {
    _environment = env;
  }
}
```

#### **2. Use in Your API Service**

```dart
import 'package:technic_app/config/environment.dart';

class ApiService {
  final String baseUrl = EnvironmentConfig.apiBaseUrl;
  
  Future<ScanResponse> runScan(ScanRequest request) async {
    final response = await http.post(
      Uri.parse('$baseUrl/v1/scan'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(request.toJson()),
    );
    // ...
  }
}
```

#### **3. Switch Environments**

```dart
// In main.dart
void main() {
  // For testing with Render
  EnvironmentConfig.setEnvironment(Environment.production);
  
  runApp(MyApp());
}
```

---

### **Option 3: Use Flutter Build Flavors (Advanced)**

#### **1. Define Flavors**

```dart
// lib/main_dev.dart
void main() {
  EnvironmentConfig.setEnvironment(Environment.development);
  runApp(MyApp());
}

// lib/main_prod.dart
void main() {
  EnvironmentConfig.setEnvironment(Environment.production);
  runApp(MyApp());
}
```

#### **2. Run Specific Flavor**

```bash
# Test with local backend
flutter run -t lib/main_dev.dart

# Test with Render backend
flutter run -t lib/main_prod.dart
```

---

## 🔄 When to Add API Key Back

### **Add API Key When:**

1. **Going Public**
   - Launching to app stores
   - Opening to real users
   - Need usage tracking

2. **Implementing Paid Tiers**
   - Free vs Pro users
   - Rate limiting
   - Feature gating

3. **Security Concerns**
   - Preventing abuse
   - Controlling costs
   - Monitoring usage

### **How to Add API Key to Flutter:**

#### **1. Store API Key Securely**

```dart
// lib/config/api_config.dart
class ApiConfig {
  // DON'T hardcode in production!
  // Use flutter_dotenv or secure storage
  static const String apiKey = 'your-api-key-here';
}
```

#### **2. Add to Requests**

```dart
Future<ScanResponse> runScan(ScanRequest request) async {
  final response = await http.post(
    Uri.parse('$baseUrl/v1/scan'),
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': ApiConfig.apiKey,  // ← Add this
    },
    body: jsonEncode(request.toJson()),
  );
  // ...
}
```

#### **3. Handle 401 Errors**

```dart
if (response.statusCode == 401) {
  throw Exception('Invalid API key - please update your app');
}
```

---

## 📊 Testing Workflow

### **Current Setup (Dev Mode):**

```
Flutter App → Render API (no auth) → Works! ✅
```

### **Steps to Test:**

1. **Remove `TECHNIC_API_KEY` from Render** (if not done)
2. **Update Flutter API base URL** to `https://technic-m5vn.onrender.com`
3. **Remove any API key headers** from Flutter code
4. **Run Flutter app:** `flutter run`
5. **Test scanner:** Click scan button in app
6. **Verify:** Scanner calls Render API and returns results

---

## 🎯 Quick Start Guide

### **To Test Flutter with Render Right Now:**

1. **Remove API Key from Render:**
   - Go to https://dashboard.render.com
   - Click "technic" service
   - Environment tab
   - Delete `TECHNIC_API_KEY`
   - Save

2. **Update Flutter Config:**
   ```dart
   // Find your API config file and change:
   const String API_BASE_URL = 'https://technic-m5vn.onrender.com';
   ```

3. **Remove API Key from Flutter:**
   ```dart
   // Remove any 'X-API-Key' headers from your HTTP requests
   ```

4. **Run Flutter:**
   ```bash
   cd technic_app
   flutter run
   ```

5. **Test Scanner:**
   - Open app
   - Click scan button
   - Should see results from Render! 🎉

---

## 🔐 Security Best Practices

### **For Development (Now):**
- ✅ No API key (easy testing)
- ✅ Render URL not public yet
- ✅ Monitor Render logs for abuse

### **For Production (Later):**
- ✅ Add API key
- ✅ Use environment variables
- ✅ Never commit keys to git
- ✅ Use different keys for dev/prod
- ✅ Implement rate limiting
- ✅ Add user authentication

---

## 📝 Summary

**Question 1: Is removing API key permanent?**
- No! You can add it back anytime
- Recommended: Keep it off during development
- Add it back when launching to users

**Question 2: How to test Render with Flutter UI?**
1. Remove `TECHNIC_API_KEY` from Render
2. Update Flutter API base URL to Render URL
3. Remove API key headers from Flutter code
4. Run `flutter run`
5. Test scanner in app - it calls Render!

**Current Recommendation:**
- Keep API key removed for now
- Test Flutter app with Render
- Add authentication later when launching

Your Flutter app will work perfectly with Render once you update the base URL! 🚀
