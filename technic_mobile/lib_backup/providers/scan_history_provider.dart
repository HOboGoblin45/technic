/// Scan History Provider
/// 
/// Manages scan history state and persistence
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/scan_history_item.dart';
import '../models/scan_result.dart';
import '../services/storage_service.dart';

/// Scan History Provider
final scanHistoryProvider = StateNotifierProvider<ScanHistoryNotifier, List<ScanHistoryItem>>((ref) {
  return ScanHistoryNotifier(ref.read(storageServiceProvider));
});

class ScanHistoryNotifier extends StateNotifier<List<ScanHistoryItem>> {
  ScanHistoryNotifier(this._storage) : super([]) {
    _loadHistory();
  }

  final StorageService _storage;
  static const int maxHistoryItems = 10;

  Future<void> _loadHistory() async {
    // Load scan history from storage
    final historyJson = await _storage.loadScanHistory();
    final history = historyJson
        .map((e) => ScanHistoryItem.fromJson(Map<String, dynamic>.from(e as Map)))
        .toList();
    state = history;
  }

  /// Save a new scan to history
  Future<void> saveScan({
    required List<ScanResult> results,
    required Map<String, dynamic> scanParams,
  }) async {
    // Calculate average MERIT
    double? avgMerit;
    if (results.isNotEmpty) {
      final merits = results
          .where((r) => r.meritScore != null)
          .map((r) => r.meritScore!)
          .toList();
      if (merits.isNotEmpty) {
        avgMerit = merits.reduce((a, b) => a + b) / merits.length;
      }
    }

    // Create new history item
    final item = ScanHistoryItem(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      timestamp: DateTime.now(),
      results: results,
      scanParams: scanParams,
      resultCount: results.length,
      averageMerit: avgMerit,
    );

    // Add to beginning of list
    final newHistory = [item, ...state];

    // Keep only last 10 items
    if (newHistory.length > maxHistoryItems) {
      state = newHistory.sublist(0, maxHistoryItems);
    } else {
      state = newHistory;
    }

    await _saveHistory();
  }

  /// Delete a scan from history
  Future<void> deleteScan(String id) async {
    state = state.where((item) => item.id != id).toList();
    await _saveHistory();
  }

  /// Clear all history
  Future<void> clearHistory() async {
    state = [];
    await _saveHistory();
  }

  /// Get scan by ID
  ScanHistoryItem? getScan(String id) {
    try {
      return state.firstWhere((item) => item.id == id);
    } catch (e) {
      return null;
    }
  }

  Future<void> _saveHistory() async {
    await _storage.saveScanHistory(state);
  }
}

/// Storage Service Provider (from app_providers.dart)
final storageServiceProvider = Provider<StorageService>((ref) {
  return StorageService.instance;
});
