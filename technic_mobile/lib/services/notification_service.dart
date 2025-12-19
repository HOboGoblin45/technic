/// Notification Service
///
/// Handles push notifications using Firebase Cloud Messaging.
/// Local notifications temporarily disabled due to package compatibility issues.
library;

import 'dart:async';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Background message handler - must be a top-level function
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  // Ensure Firebase is initialized
  await Firebase.initializeApp();
  debugPrint('[FCM] Background message: ${message.messageId}');
}

/// Notification Service
///
/// Manages push notifications using Firebase Cloud Messaging.
/// Note: Local notifications are temporarily disabled due to Android SDK compatibility.
class NotificationService {
  NotificationService._();

  static final NotificationService _instance = NotificationService._();
  factory NotificationService() => _instance;

  final FirebaseMessaging _fcm = FirebaseMessaging.instance;

  bool _isInitialized = false;
  String? _fcmToken;

  /// Get the current FCM token
  String? get fcmToken => _fcmToken;

  /// Initialize the notification service
  Future<void> initialize() async {
    if (_isInitialized) return;

    // Initialize FCM
    await _initializeFCM();

    _isInitialized = true;
    debugPrint('[Notifications] Initialized successfully (FCM only)');
  }

  /// Initialize Firebase Cloud Messaging
  Future<void> _initializeFCM() async {
    // Set up background message handler
    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

    // Request permission (iOS)
    final settings = await _fcm.requestPermission(
      alert: true,
      announcement: false,
      badge: true,
      carPlay: false,
      criticalAlert: false,
      provisional: false,
      sound: true,
    );

    debugPrint('[FCM] Permission status: ${settings.authorizationStatus}');

    // Get FCM token
    _fcmToken = await _fcm.getToken();
    debugPrint('[FCM] Token: $_fcmToken');

    // Listen for token refresh
    _fcm.onTokenRefresh.listen((token) {
      _fcmToken = token;
      debugPrint('[FCM] Token refreshed: $token');
    });

    // Handle foreground messages
    FirebaseMessaging.onMessage.listen(_handleForegroundMessage);

    // Handle notification taps when app is in background/terminated
    FirebaseMessaging.onMessageOpenedApp.listen(_handleMessageOpenedApp);

    // Check for initial message (app opened via notification)
    final initialMessage = await _fcm.getInitialMessage();
    if (initialMessage != null) {
      debugPrint('[FCM] App opened from notification');
      _handleMessageOpenedApp(initialMessage);
    }

    // Subscribe to topics for broadcast notifications
    await _fcm.subscribeToTopic('alerts');
    await _fcm.subscribeToTopic('announcements');
  }

  /// Handle foreground messages
  void _handleForegroundMessage(RemoteMessage message) {
    debugPrint('[FCM] Foreground message: ${message.messageId}');
    debugPrint('[FCM] Title: ${message.notification?.title}');
    debugPrint('[FCM] Body: ${message.notification?.body}');
  }

  /// Handle notification tap when app is in background
  void _handleMessageOpenedApp(RemoteMessage message) {
    debugPrint('[FCM] Message opened app: ${message.data}');

    final ticker = message.data['ticker'];
    if (ticker != null) {
      debugPrint('[FCM] Should navigate to $ticker');
    }
  }

  /// Request notification permissions
  Future<bool> requestPermissions() async {
    final settings = await _fcm.requestPermission(
      alert: true,
      badge: true,
      sound: true,
    );

    final granted = settings.authorizationStatus == AuthorizationStatus.authorized ||
        settings.authorizationStatus == AuthorizationStatus.provisional;

    debugPrint('[Notifications] Permissions granted: $granted');
    return granted;
  }

  /// Show a price alert notification (via FCM only)
  Future<void> showAlertNotification({
    required String ticker,
    required String title,
    required String body,
    String? payload,
  }) async {
    debugPrint('[Notifications] Alert: $title - $body (FCM only, no local notification)');
    // Local notifications disabled - would need backend to send FCM message
  }

  /// Show a general notification (via FCM only)
  Future<void> showNotification({
    required String title,
    required String body,
    String? payload,
  }) async {
    debugPrint('[Notifications] Notification: $title - $body (FCM only, no local notification)');
    // Local notifications disabled - would need backend to send FCM message
  }

  /// Cancel a specific notification (not supported without local notifications)
  Future<void> cancel(int id) async {
    debugPrint('[Notifications] Cancel not supported (local notifications disabled)');
  }

  /// Cancel all notifications (not supported without local notifications)
  Future<void> cancelAll() async {
    debugPrint('[Notifications] Cancel all not supported (local notifications disabled)');
  }

  /// Get pending notifications (not supported without local notifications)
  Future<List<dynamic>> getPendingNotifications() async {
    return [];
  }

  /// Check if notifications are enabled
  Future<bool> areNotificationsEnabled() async {
    final settings = await _fcm.getNotificationSettings();
    return settings.authorizationStatus == AuthorizationStatus.authorized ||
        settings.authorizationStatus == AuthorizationStatus.provisional;
  }

  /// Subscribe to a topic for targeted notifications
  Future<void> subscribeToTopic(String topic) async {
    await _fcm.subscribeToTopic(topic);
    debugPrint('[FCM] Subscribed to topic: $topic');
  }

  /// Unsubscribe from a topic
  Future<void> unsubscribeFromTopic(String topic) async {
    await _fcm.unsubscribeFromTopic(topic);
    debugPrint('[FCM] Unsubscribed from topic: $topic');
  }

  /// Subscribe to alerts for a specific symbol
  Future<void> subscribeToSymbol(String ticker) async {
    await _fcm.subscribeToTopic('symbol_$ticker');
    debugPrint('[FCM] Subscribed to symbol alerts: $ticker');
  }

  /// Unsubscribe from a symbol's alerts
  Future<void> unsubscribeFromSymbol(String ticker) async {
    await _fcm.unsubscribeFromTopic('symbol_$ticker');
    debugPrint('[FCM] Unsubscribed from symbol alerts: $ticker');
  }

  /// Delete FCM token (for logout)
  Future<void> deleteToken() async {
    await _fcm.deleteToken();
    _fcmToken = null;
    debugPrint('[FCM] Token deleted');
  }
}

/// Notification Service Provider
final notificationServiceProvider = Provider<NotificationService>((ref) {
  return NotificationService();
});
