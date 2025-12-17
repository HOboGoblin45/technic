/// Local Store Service
/// 
/// Handles all local storage operations using SharedPreferences.
library;

import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/scan_result.dart';
import '../models/market_mover.dart';
import '../models/scanner_bundle.dart';
import '../models/saved_screen.dart';

/// Local storage service for persisting app state
class LocalStore {
  static Future<SharedPreferences> _prefs() => SharedPreferences.getInstance();

  /// Load user ID from storage
  static Future<String?> loadUser() async {
    final p = await _prefs();
    return p.getString('user_id');
  }

  /// Save user ID to storage
  static Future<void> saveUser(String user) async {
    final p = await _prefs();
    await p.setString('user_id', user);
  }

  /// Load last active tab index
  static Future<int?> loadLastTab() async {
    final p = await _prefs();
    return p.getInt('last_tab');
  }

  /// Save last active tab index
  static Future<void> saveLastTab(int index) async {
    final p = await _prefs();
    await p.setInt('last_tab', index);
  }

  /// Load scanner state including filters, presets, and cached results
  static Future<Map<String, dynamic>?> loadScannerState() async {
    final p = await _prefs();
    final filters = p.getString('filters');
    if (filters == null) return null;
    
    final saved = p.getString('saved_screens');
    final lastScans = p.getString('last_scans');
    final lastMovers = p.getString('last_movers');
    
    return {
      'filters': Map<String, String>.from(jsonDecode(filters)),
      'saved_screens': saved != null
          ? (jsonDecode(saved) as List)
              .map((e) => SavedScreen.fromJson(Map<String, dynamic>.from(e)))
              .toList()
          : <SavedScreen>[],
      'scanCount': p.getInt('scan_count') ?? 0,
      'streakDays': p.getInt('streak_days') ?? 0,
      'lastScan': p.getString('last_scan'),
      'advancedMode': p.getBool('advanced_mode') ?? true,
      'showOnboarding': p.getBool('show_onboarding') ?? true,
      'last_scans': lastScans != null
          ? (jsonDecode(lastScans) as List)
              .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e)))
              .toList()
          : <ScanResult>[],
      'last_movers': lastMovers != null
          ? (jsonDecode(lastMovers) as List)
              .map((e) => MarketMover.fromJson(Map<String, dynamic>.from(e)))
              .toList()
          : <MarketMover>[],
    };
  }

  /// Save scanner state including filters, presets, and statistics
  static Future<void> saveScannerState({
    required Map<String, String> filters,
    required List<SavedScreen> savedScreens,
    required int scanCount,
    required int streakDays,
    required DateTime? lastScan,
    required bool advancedMode,
    required bool showOnboarding,
    List<ScanResult>? lastScans,
    List<MarketMover>? lastMovers,
  }) async {
    final p = await _prefs();
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

  /// Save last scanner bundle (scan results and movers)
  static Future<void> saveLastBundle(ScannerBundle bundle) async {
    final p = await _prefs();
    await p.setString(
      'last_scans',
      jsonEncode(bundle.scanResults.map((e) => e.toJson()).toList()),
    );
    await p.setString(
      'last_movers',
      jsonEncode(bundle.movers.map((e) => e.toJson()).toList()),
    );
  }
}
