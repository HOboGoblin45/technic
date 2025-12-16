# Fix Scanner Timeout Issue

## 🔍 Problem Identified

The scanner is stuck at "Initializing scan..." because:

1. **Long-running request**: Full scan takes 75-90 seconds
2. **No progress updates**: HTTP request doesn't stream progress
3. **Possible timeout**: Default HTTP timeout might be too short
4. **No error handling**: If request fails, UI stays stuck

## 🔧 Solution

### **Option 1: Increase HTTP Timeout (Quick Fix)**

Update `api_service.dart` to increase timeout:

```dart
class ApiService {
  ApiService({http.Client? client, ApiConfig? config})
      : _client = client ?? http.Client(),
        _config = config ?? ApiConfig.fromEnv();

  final http.Client _client;
  final ApiConfig _config;
  
  // Add timeout constant
  static const Duration _scanTimeout = Duration(minutes: 3); // 3 minutes for full scan

  Future<ScannerBundle> fetchScannerBundle({
    Map<String, String>? params,
  }) async {
    try {
      final body = <String, dynamic>{
        'max_symbols': int.tryParse(params?['max_symbols'] ?? '6000') ?? 6000,
        'trade_style': params?['trade_style'] ?? 'Short-term swing',
        'min_tech_rating': double.tryParse(params?['min_tech_rating'] ?? '0.0') ?? 0.0,
      };
      
      // ... rest of body building ...
      
      final res = await _client.post(
        _config.scanUri(),
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: jsonEncode(body),
      ).timeout(_scanTimeout); // ← Add timeout here
      
      // ... rest of method ...
    } catch (e) {
      debugPrint('API error: $e');
      rethrow;
    }
  }
}
```

### **Option 2: Start with Smaller Scan (Recommended)**

Modify the default scan to use fewer symbols initially:

```dart
// In scanner_page.dart _fetchBundle()
Future<ScannerBundle> _fetchBundle() async {
  setState(() {
    _isScanning = true;
    _scanStartTime = DateTime.now();
    // Start with smaller number for testing
    final maxSymbols = int.tryParse(_filters['max_symbols'] ?? '100') ?? 100;
    _symbolsScanned = 0;
    _totalSymbols = maxSymbols;
    _scanProgress = 'Initializing scan...';
  });

  try {
    final apiService = ref.read(apiServiceProvider);
    
    // Override max_symbols if not set
    final scanParams = Map<String, String>.from(_filters);
    if (!scanParams.containsKey('max_symbols')) {
      scanParams['max_symbols'] = '100'; // Start small
    }
    
    final bundle = await apiService.fetchScannerBundle(
      params: scanParams.isNotEmpty ? scanParams : null,
    );
    
    // ... rest of method ...
  }
}
```

### **Option 3: Add Better Error Handling**

Add timeout error handling in scanner_page.dart:

```dart
Future<ScannerBundle> _fetchBundle() async {
  setState(() {
    _isScanning = true;
    _scanStartTime = DateTime.now();
    _symbolsScanned = 0;
    _totalSymbols = int.tryParse(_filters['max_symbols'] ?? '6000') ?? 6000;
    _scanProgress = 'Initializing scan...';
  });

  try {
    final apiService = ref.read(apiServiceProvider);
    final bundle = await apiService.fetchScannerBundle(
      params: _filters.isNotEmpty ? _filters : null,
    ).timeout(
      const Duration(minutes: 3),
      onTimeout: () {
        throw TimeoutException('Scan took too long. Try reducing max_symbols.');
      },
    );

    setState(() {
      _scanCount++;
      final now = DateTime.now();
      if (_lastScan != null &&
          now.difference(_lastScan!).inHours < 24 &&
          now.day != _lastScan!.day) {
        _streakDays++;
      } else if (_lastScan == null ||
          now.difference(_lastScan!).inHours >= 48) {
        _streakDays = 1;
      }
      _lastScan = now;
      _isScanning = false;
    });
    _saveState();

    await LocalStore.saveLastBundle(bundle);
    return bundle;
    
  } on TimeoutException catch (e) {
    setState(() {
      _isScanning = false;
    });
    
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(e.message ?? 'Request timed out'),
          action: SnackBarAction(
            label: 'Retry with 100 symbols',
            onPressed: () {
              setState(() {
                _filters['max_symbols'] = '100';
              });
              _refresh();
            },
          ),
        ),
      );
    }
    
    // Try to load cached data
    final state = await LocalStore.loadScannerState();
    final scansList = state?['last_scans'] as List?;
    final lastScans = scansList
        ?.map((e) => ScanResult.fromJson(e as Map<String, dynamic>))
        .toList() ?? [];
    
    if (lastScans.isNotEmpty) {
      return ScannerBundle(
        scanResults: lastScans,
        movers: [],
        scoreboard: [],
        progress: 'Loaded from cache (scan timed out)',
      );
    }
    
    rethrow;
    
  } catch (e) {
    setState(() {
      _isScanning = false;
    });
    
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Scan failed: ${e.toString()}'),
          action: SnackBarAction(
            label: 'Retry',
            onPressed: _refresh,
          ),
        ),
      );
    }
    
    // Try to load cached data
    final state = await LocalStore.loadScannerState();
    final scansList = state?['last_scans'] as List?;
    final lastScans = scansList
        ?.map((e) => ScanResult.fromJson(e as Map<String, dynamic>))
        .toList() ?? [];
    
    final moversList = state?['last_movers'] as List?;
    final lastMovers = moversList
        ?.map((e) => MarketMover.fromJson(e as Map<String, dynamic>))
        .toList() ?? [];

    if (lastScans.isNotEmpty) {
      return ScannerBundle(
        scanResults: lastScans,
        movers: lastMovers,
        scoreboard: [],
        progress: 'Loaded from cache (offline mode)',
      );
    }

    rethrow;
  }
}
```

## 🎯 Recommended Quick Fix

**Test with 100 symbols first:**

1. In the Flutter app, before clicking "Run Scan"
2. Open filters (tune icon)
3. Set `max_symbols` to `100`
4. Click "Run Scan"
5. Should complete in ~2-3 seconds

**If that works, gradually increase:**
- 100 symbols: ~2-3 seconds
- 500 symbols: ~10-15 seconds
- 1000 symbols: ~20-30 seconds
- 6000 symbols: ~75-90 seconds

## 🔍 Debug Steps

1. **Check Render logs** to see if API is receiving the request
2. **Check Flutter console** for any error messages
3. **Test with curl** to verify API works:
   ```powershell
   Invoke-RestMethod -Uri "https://technic-m5vn.onrender.com/v1/scan" -Method Post -Body '{"max_symbols":10}' -ContentType "application/json"
   ```

## 📝 Files to Update

1. `technic_app/lib/services/api_service.dart` - Add timeout
2. `technic_app/lib/screens/scanner/scanner_page.dart` - Add error handling
3. Test with small `max_symbols` first

Would you like me to implement these fixes?
