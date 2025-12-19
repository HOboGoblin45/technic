/// Alert Service
/// 
/// Background service to check price alerts and trigger notifications.
library;

import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/price_alert.dart';
import '../providers/alert_provider.dart';

/// Alert Service
/// 
/// Checks active alerts periodically and triggers them when conditions are met.
/// Note: This is a basic implementation. For production, consider using
/// background tasks or push notifications.
class AlertService {
  AlertService(this._ref);

  final Ref _ref;
  Timer? _timer;
  bool _isRunning = false;

  /// Start checking alerts periodically
  void start({Duration interval = const Duration(minutes: 5)}) {
    if (_isRunning) return;

    _isRunning = true;
    _timer = Timer.periodic(interval, (_) => _checkAlerts());
    
    // Check immediately on start
    _checkAlerts();
  }

  /// Stop checking alerts
  void stop() {
    _timer?.cancel();
    _timer = null;
    _isRunning = false;
  }

  /// Check all active alerts
  Future<void> _checkAlerts() async {
    final alertNotifier = _ref.read(alertProvider.notifier);
    final activeAlerts = alertNotifier.getActiveAlerts();

    for (final alert in activeAlerts) {
      await _checkAlert(alert, alertNotifier);
    }
  }

  /// Check a single alert
  Future<void> _checkAlert(
    PriceAlert alert,
    AlertNotifier alertNotifier,
  ) async {
    // TODO: Implement alert checking logic
    // This is a placeholder for future implementation
    // In production, you would:
    // 1. Fetch current price from your API
    // 2. Compare with alert.targetValue based on alert.type
    // 3. Trigger notification if condition is met
    // 4. Call alertNotifier.triggerAlert(alert.id)
  }

  /// Dispose resources
  void dispose() {
    stop();
  }
}

/// Alert Service Provider
final alertServiceProvider = Provider<AlertService>((ref) {
  final service = AlertService(ref);
  ref.onDispose(() => service.dispose());
  return service;
});
