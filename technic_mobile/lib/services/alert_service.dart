/// Alert Service
///
/// Background service to check price alerts and trigger notifications.
library;

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/price_alert.dart';
import '../providers/alert_provider.dart';
import 'api_service.dart';
import 'notification_service.dart';

/// Alert Service
///
/// Checks active alerts periodically and triggers them when conditions are met.
/// Integrates with NotificationService for user notifications and ApiService
/// for fetching current prices.
class AlertService {
  AlertService(this._ref);

  final Ref _ref;
  Timer? _timer;
  bool _isRunning = false;
  bool _isChecking = false;

  // Cache to store last known prices for percent change calculations
  final Map<String, double> _lastPrices = {};

  /// Start checking alerts periodically
  ///
  /// [interval] - How often to check alerts (default: 5 minutes)
  void start({Duration interval = const Duration(minutes: 5)}) {
    if (_isRunning) return;

    _isRunning = true;
    _timer = Timer.periodic(interval, (_) => checkAlerts());

    // Check immediately on start
    checkAlerts();

    debugPrint('[AlertService] Started with ${interval.inMinutes} minute interval');
  }

  /// Stop checking alerts
  void stop() {
    _timer?.cancel();
    _timer = null;
    _isRunning = false;
    debugPrint('[AlertService] Stopped');
  }

  /// Check all active alerts
  ///
  /// This is the main alert checking logic that:
  /// 1. Gets all active alerts
  /// 2. Fetches current prices for the symbols
  /// 3. Compares prices against alert conditions
  /// 4. Triggers notifications for matched alerts
  Future<void> checkAlerts() async {
    // Prevent concurrent checks
    if (_isChecking) {
      debugPrint('[AlertService] Already checking, skipping');
      return;
    }

    _isChecking = true;

    try {
      final alertNotifier = _ref.read(alertProvider.notifier);
      final activeAlerts = alertNotifier.getActiveAlerts();

      if (activeAlerts.isEmpty) {
        debugPrint('[AlertService] No active alerts to check');
        _isChecking = false;
        return;
      }

      debugPrint('[AlertService] Checking ${activeAlerts.length} active alerts');

      // Get unique tickers from alerts
      final tickers = activeAlerts.map((a) => a.ticker).toSet().toList();

      // Fetch current prices
      final apiService = ApiService();
      final prices = await apiService.getCurrentPrices(tickers);

      debugPrint('[AlertService] Fetched prices for ${prices.length} symbols');

      // Check each alert
      for (final alert in activeAlerts) {
        await _checkAlert(alert, alertNotifier, prices);
      }

      // Update last known prices for percent change calculations
      _lastPrices.addAll(prices);
    } catch (e) {
      debugPrint('[AlertService] Error checking alerts: $e');
    } finally {
      _isChecking = false;
    }
  }

  /// Check a single alert against current price
  Future<void> _checkAlert(
    PriceAlert alert,
    AlertNotifier alertNotifier,
    Map<String, double> prices,
  ) async {
    final currentPrice = prices[alert.ticker];

    if (currentPrice == null) {
      debugPrint('[AlertService] No price for ${alert.ticker}, skipping');
      return;
    }

    bool triggered = false;
    String? notificationBody;

    switch (alert.type) {
      case AlertType.priceAbove:
        if (currentPrice >= alert.targetValue) {
          triggered = true;
          notificationBody =
              '${alert.ticker} is now \$${currentPrice.toStringAsFixed(2)} '
              '(above \$${alert.targetValue.toStringAsFixed(2)})';
        }

      case AlertType.priceBelow:
        if (currentPrice <= alert.targetValue) {
          triggered = true;
          notificationBody =
              '${alert.ticker} is now \$${currentPrice.toStringAsFixed(2)} '
              '(below \$${alert.targetValue.toStringAsFixed(2)})';
        }

      case AlertType.percentChange:
        // Get last known price for comparison
        final lastPrice = _lastPrices[alert.ticker];
        if (lastPrice != null && lastPrice > 0) {
          final percentChange = ((currentPrice - lastPrice) / lastPrice) * 100;
          final absChange = percentChange.abs();

          if (absChange >= alert.targetValue.abs()) {
            triggered = true;
            final direction = percentChange >= 0 ? 'up' : 'down';
            notificationBody =
                '${alert.ticker} moved $direction ${absChange.toStringAsFixed(2)}% '
                'to \$${currentPrice.toStringAsFixed(2)}';
          }
        }
    }

    if (triggered) {
      debugPrint('[AlertService] Alert triggered: ${alert.description}');

      // Show notification
      final notificationService = NotificationService();
      await notificationService.showAlertNotification(
        ticker: alert.ticker,
        title: '${alert.ticker} Alert',
        body: notificationBody ?? alert.description,
        payload: alert.id,
      );

      // Mark alert as triggered
      await alertNotifier.triggerAlert(alert.id);
    }
  }

  /// Manually trigger a check (for testing or user-initiated refresh)
  Future<void> forceCheck() async {
    debugPrint('[AlertService] Force check initiated');
    await checkAlerts();
  }

  /// Get whether the service is running
  bool get isRunning => _isRunning;

  /// Get count of pending checks
  Future<int> getPendingAlertCount() async {
    final alertNotifier = _ref.read(alertProvider.notifier);
    return alertNotifier.getActiveAlerts().length;
  }

  /// Clear the price cache
  void clearPriceCache() {
    _lastPrices.clear();
    debugPrint('[AlertService] Price cache cleared');
  }

  /// Dispose resources
  void dispose() {
    stop();
    _lastPrices.clear();
  }
}

/// Alert Service Provider
final alertServiceProvider = Provider<AlertService>((ref) {
  final service = AlertService(ref);
  ref.onDispose(() => service.dispose());
  return service;
});
