/// Alert Provider
///
/// State management for price alerts.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/price_alert.dart';
import '../services/storage_service.dart';
import 'app_providers.dart'; // Import centralized storageServiceProvider

/// Alert Notifier
class AlertNotifier extends StateNotifier<List<PriceAlert>> {
  AlertNotifier(this._storage) : super([]) {
    _loadAlerts();
  }

  final StorageService _storage;

  /// Load alerts from storage
  Future<void> _loadAlerts() async {
    try {
      final alertsJson = await _storage.loadAlerts();
      state = alertsJson
          .map((json) => PriceAlert.fromJson(Map<String, dynamic>.from(json as Map)))
          .toList();
    } catch (e) {
      // If loading fails, start with empty list
      state = [];
    }
  }

  /// Add a new alert
  Future<void> addAlert(PriceAlert alert) async {
    state = [...state, alert];
    await _saveAlerts();
  }

  /// Remove an alert by ID
  Future<void> removeAlert(String id) async {
    state = state.where((a) => a.id != id).toList();
    await _saveAlerts();
  }

  /// Toggle alert active status
  Future<void> toggleAlert(String id) async {
    state = state.map((a) {
      if (a.id == id) {
        return a.copyWith(isActive: !a.isActive);
      }
      return a;
    }).toList();
    await _saveAlerts();
  }

  /// Trigger an alert (mark as triggered and inactive)
  Future<void> triggerAlert(String id) async {
    state = state.map((a) {
      if (a.id == id) {
        return a.copyWith(
          triggeredAt: DateTime.now(),
          isActive: false,
        );
      }
      return a;
    }).toList();
    await _saveAlerts();
  }

  /// Update an existing alert
  Future<void> updateAlert(PriceAlert alert) async {
    state = state.map((a) {
      if (a.id == alert.id) {
        return alert;
      }
      return a;
    }).toList();
    await _saveAlerts();
  }

  /// Get alerts for a specific ticker
  List<PriceAlert> getAlertsForTicker(String ticker) {
    return state.where((a) => a.ticker == ticker).toList();
  }

  /// Get all active alerts
  List<PriceAlert> getActiveAlerts() {
    return state.where((a) => a.isActive && !a.isTriggered).toList();
  }

  /// Get all triggered alerts
  List<PriceAlert> getTriggeredAlerts() {
    return state.where((a) => a.isTriggered).toList();
  }

  /// Check if ticker has any active alerts
  bool hasActiveAlerts(String ticker) {
    return state.any((a) => a.ticker == ticker && a.isActive && !a.isTriggered);
  }

  /// Get count of active alerts for ticker
  int getActiveAlertCount(String ticker) {
    return state.where((a) => a.ticker == ticker && a.isActive && !a.isTriggered).length;
  }

  /// Clear all triggered alerts
  Future<void> clearTriggeredAlerts() async {
    state = state.where((a) => !a.isTriggered).toList();
    await _saveAlerts();
  }

  /// Save alerts to storage
  Future<void> _saveAlerts() async {
    try {
      await _storage.saveAlerts(state.map((a) => a.toJson()).toList());
    } catch (e) {
      // Handle save error silently
      // In production, use proper logging framework
    }
  }

  /// Refresh alerts from storage
  Future<void> refresh() async {
    await _loadAlerts();
  }
}

/// Alert Provider
final alertProvider = StateNotifierProvider<AlertNotifier, List<PriceAlert>>((ref) {
  return AlertNotifier(ref.read(storageServiceProvider));
});
