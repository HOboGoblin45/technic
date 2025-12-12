/// Storage Service
/// 
/// Handles local data persistence using SharedPreferences.
/// Provides clean interface for saving/loading app state.

import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/scan_result.dart';
import '../models/market_mover.dart';
import '../models/scanner_bundle.dart';

/// Saved screen preset
class SavedScreen {
  final String name;
  final String description;
  final String horizon;
  final bool isActive;
  final Map<String, String>? params;

  const SavedScreen(
    this.name,
    this.description,
    this.horizon,
    this.isActive, {
    this.params,
  });

  Map<String, dynamic> toJson() => {
        'name': name,
        'description': description,
        'horizon': horizon,
        'isActive': isActive,
        'params': params,
      };

  factory SavedScreen.fromJson(Map<String, dynamic> json) => SavedScreen(
        json['name']?.toString() ?? '',
        json['description']?.toString() ?? '',
        json['horizon']?.toString() ?? '',
        json['isActive'] == true,
        params: (json['params'] as Map?)?.map(
              (k, v) => MapEntry(k.toString(), v.toString()),
            ) ??
            const {},
      );

  SavedScreen copyWith({
    String? name,
    String? description,
    String? horizon,
    bool? isActive,
    Map<String, String>? params,
  }) {
    return SavedScreen(
      name ?? this.name,
      description ?? this.description,
      horizon ?? this.horizon,
      isActive ?? this.isActive,
      params: params ?? this.params,
    );
  }
}

/// Scanner state data
class ScannerState {
  final Map<String, String> filters;
  final List<SavedScreen> savedScreens;
  final int scanCount;
  final int streakDays;
  final DateTime? lastScan;
  final bool advancedMode;
  final bool showOnboarding;
  final List<ScanResult> lastScans;
  final List<MarketMover> lastMovers;

  const ScannerState({
    required this.filters,
    required this.savedScreens,
    required this.scanCount,
    required this.streakDays,
    this.lastScan,
    required this.advancedMode,
    required this.showOnboarding,
    required this.lastScans,
    required this.lastMovers,
  });
}

/// Local Storage Service
class StorageService {
  StorageService._();
  
  static final StorageService instance = StorageService._();
  
  SharedPreferences? _prefs;

  /// Initialize storage
  Future<void> init() async {
    _prefs ??= await SharedPreferences.getInstance();
  }

  Future<SharedPreferences> get _prefsInstance async {
    if (_prefs == null) await init();
    return _prefs!;
  }

  // ============================================================================
  // USER MANAGEMENT
  // ============================================================================

  /// Load user ID
  Future<String?> loadUser() async {
    final p = await _prefsInstance;
    return p.getString('user_id');
  }

  /// Save user ID
  Future<void> saveUser(String userId) async {
    final p = await _prefsInstance;
    await p.setString('user_id', userId);
  }

  /// Clear user ID (sign out)
  Future<void> clearUser() async {
    final p = await _prefsInstance;
    await p.remove('user_id');
  }

  // ============================================================================
  // TAB NAVIGATION
  // ============================================================================

  /// Load last active tab index
  Future<int?> loadLastTab() async {
    final p = await _prefsInstance;
    return p.getInt('last_tab');
  }

  /// Save last active tab index
  Future<void> saveLastTab(int index) async {
    final p = await _prefsInstance;
    await p.setInt('last_tab', index);
  }

  // ============================================================================
  // SCANNER STATE
  // ============================================================================

  /// Load complete scanner state
  Future<ScannerState?> loadScannerState() async {
    final p = await _prefsInstance;
    final filtersJson = p.getString('filters');
    
    if (filtersJson == null) return null;

    final savedJson = p.getString('saved_screens');
    final lastScansJson = p.getString('last_scans');
    final lastMoversJson = p.getString('last_movers');
    final lastScanStr = p.getString('last_scan');

    return ScannerState(
      filters: Map<String, String>.from(jsonDecode(filtersJson) as Map),
      savedScreens: savedJson != null
          ? (jsonDecode(savedJson) as List)
              .map((e) => SavedScreen.fromJson(Map<String, dynamic>.from(e as Map)))
              .toList()
          : [],
      scanCount: p.getInt('scan_count') ?? 0,
      streakDays: p.getInt('streak_days') ?? 0,
      lastScan: lastScanStr != null ? DateTime.tryParse(lastScanStr) : null,
      advancedMode: p.getBool('advanced_mode') ?? false,
      showOnboarding: p.getBool('show_onboarding') ?? true,
      lastScans: lastScansJson != null
          ? (jsonDecode(lastScansJson) as List)
              .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e as Map)))
              .toList()
          : [],
      lastMovers: lastMoversJson != null
          ? (jsonDecode(lastMoversJson) as List)
              .map((e) => MarketMover.fromJson(Map<String, dynamic>.from(e as Map)))
              .toList()
          : [],
    );
  }

  /// Save complete scanner state
  Future<void> saveScannerState({
    required Map<String, String> filters,
    required List<SavedScreen> savedScreens,
    required int scanCount,
    required int streakDays,
    DateTime? lastScan,
    required bool advancedMode,
    required bool showOnboarding,
    List<ScanResult>? lastScans,
    List<MarketMover>? lastMovers,
  }) async {
    final p = await _prefsInstance;
    
    await p.setString('filters', jsonEncode(filters));
    await p.setString(
      'saved_screens',
      jsonEncode(savedScreens.map((e) => e.toJson()).toList()),
    );
    await p.setInt('scan_count', scanCount);
    await p.setInt('streak_days', streakDays);
    
    if (lastScan != null) {
      await p.setString('last_scan', lastScan.toIso8601String());
    }
    
    await p.setBool('advanced_mode', advancedMode);
    await p.setBool('show_onboarding', showOnboarding);
    
    if (lastScans != null) {
      await p.setString(
        'last_scans',
        jsonEncode(lastScans.map((e) => e.toJson()).toList()),
      );
    }
    
    if (lastMovers != null) {
      await p.setString(
        'last_movers',
        jsonEncode(lastMovers.map((e) => e.toJson()).toList()),
      );
    }
  }

  /// Save last scanner bundle (quick cache)
  Future<void> saveLastBundle(ScannerBundle bundle) async {
    final p = await _prefsInstance;
    
    await p.setString(
      'last_scans',
      jsonEncode(bundle.scanResults.map((e) => e.toJson()).toList()),
    );
    await p.setString(
      'last_movers',
      jsonEncode(bundle.movers.map((e) => e.toJson()).toList()),
    );
  }

  // ============================================================================
  // THEME & PREFERENCES
  // ============================================================================

  /// Load theme mode
  Future<String?> loadThemeMode() async {
    final p = await _prefsInstance;
    return p.getString('theme_mode');
  }

  /// Save theme mode
  Future<void> saveThemeMode(String mode) async {
    final p = await _prefsInstance;
    await p.setString('theme_mode', mode);
  }

  /// Load options mode
  Future<String?> loadOptionsMode() async {
    final p = await _prefsInstance;
    return p.getString('options_mode');
  }

  /// Save options mode
  Future<void> saveOptionsMode(String mode) async {
    final p = await _prefsInstance;
    await p.setString('options_mode', mode);
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /// Clear all stored data
  Future<void> clearAll() async {
    final p = await _prefsInstance;
    await p.clear();
  }

  /// Check if key exists
  Future<bool> containsKey(String key) async {
    final p = await _prefsInstance;
    return p.containsKey(key);
  }

  /// Remove specific key
  Future<void> remove(String key) async {
    final p = await _prefsInstance;
    await p.remove(key);
  }

  /// Get all keys
  Future<Set<String>> getAllKeys() async {
    final p = await _prefsInstance;
    return p.getKeys();
  }
}
