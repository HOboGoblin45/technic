/// Notification Service
///
/// Handles local and push notifications for price alerts and other app events.
/// Integrates with Firebase Cloud Messaging (FCM) for remote push notifications.
library;

import 'dart:async';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Background message handler - must be a top-level function
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  // Ensure Firebase is initialized
  await Firebase.initializeApp();
  debugPrint('[FCM] Background message: ${message.messageId}');

  // Show local notification for background messages
  final notificationService = NotificationService();
  await notificationService.initialize();

  if (message.notification != null) {
    await notificationService.showNotification(
      title: message.notification!.title ?? 'Technic',
      body: message.notification!.body ?? '',
      payload: message.data['ticker'],
    );
  }
}

/// Notification Service
///
/// Manages local and push notifications for price alerts and app events.
/// Uses flutter_local_notifications for local notifications and
/// firebase_messaging for remote push notifications.
class NotificationService {
  NotificationService._();

  static final NotificationService _instance = NotificationService._();
  factory NotificationService() => _instance;

  final FlutterLocalNotificationsPlugin _localNotifications =
      FlutterLocalNotificationsPlugin();
  final FirebaseMessaging _fcm = FirebaseMessaging.instance;

  bool _isInitialized = false;
  String? _fcmToken;

  /// Get the current FCM token
  String? get fcmToken => _fcmToken;

  /// Initialize the notification service
  ///
  /// Must be called after Firebase.initializeApp() and before
  /// showing any notifications. Typically called in main.dart.
  Future<void> initialize() async {
    if (_isInitialized) return;

    // Initialize local notifications
    await _initializeLocalNotifications();

    // Initialize FCM
    await _initializeFCM();

    _isInitialized = true;
    debugPrint('[Notifications] Initialized successfully');
  }

  /// Initialize local notifications
  Future<void> _initializeLocalNotifications() async {
    // Android settings
    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');

    // iOS settings
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: false, // We'll request via FCM
      requestBadgePermission: false,
      requestSoundPermission: false,
    );

    const settings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _localNotifications.initialize(
      settings,
      onDidReceiveNotificationResponse: _onNotificationTapped,
    );
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
      // TODO: Send updated token to backend
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

    // Show local notification for foreground messages
    if (message.notification != null) {
      showNotification(
        title: message.notification!.title ?? 'Technic',
        body: message.notification!.body ?? '',
        payload: message.data['ticker'],
      );
    }
  }

  /// Handle notification tap when app is in background
  void _handleMessageOpenedApp(RemoteMessage message) {
    debugPrint('[FCM] Message opened app: ${message.data}');

    // Navigate to relevant screen based on notification data
    final ticker = message.data['ticker'];
    if (ticker != null) {
      // TODO: Navigate to symbol detail page
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

  /// Show a price alert notification
  Future<void> showAlertNotification({
    required String ticker,
    required String title,
    required String body,
    String? payload,
  }) async {
    if (!_isInitialized) {
      debugPrint('[Notifications] Service not initialized, skipping notification');
      return;
    }

    const androidDetails = AndroidNotificationDetails(
      'price_alerts',
      'Price Alerts',
      channelDescription: 'Notifications for price alert triggers',
      importance: Importance.high,
      priority: Priority.high,
      ticker: 'Price Alert',
      icon: '@mipmap/ic_launcher',
      color: Color(0xFF4A9EFF), // Technic blue
      enableVibration: true,
      playSound: true,
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
      threadIdentifier: 'price_alerts',
    );

    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    final id = DateTime.now().millisecondsSinceEpoch ~/ 1000;

    await _localNotifications.show(
      id,
      title,
      body,
      details,
      payload: payload ?? ticker,
    );

    debugPrint('[Notifications] Showed alert: $title - $body');
  }

  /// Show a general notification
  Future<void> showNotification({
    required String title,
    required String body,
    String? payload,
  }) async {
    if (!_isInitialized) {
      debugPrint('[Notifications] Service not initialized, skipping notification');
      return;
    }

    const androidDetails = AndroidNotificationDetails(
      'general',
      'General',
      channelDescription: 'General app notifications',
      importance: Importance.defaultImportance,
      priority: Priority.defaultPriority,
      icon: '@mipmap/ic_launcher',
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    final id = DateTime.now().millisecondsSinceEpoch ~/ 1000;

    await _localNotifications.show(
      id,
      title,
      body,
      details,
      payload: payload,
    );
  }

  /// Cancel a specific notification
  Future<void> cancel(int id) async {
    await _localNotifications.cancel(id);
  }

  /// Cancel all notifications
  Future<void> cancelAll() async {
    await _localNotifications.cancelAll();
  }

  /// Handle notification tap
  void _onNotificationTapped(NotificationResponse response) {
    debugPrint('[Notifications] Tapped: ${response.payload}');
    // TODO: Navigate to relevant screen based on payload
  }

  /// Get pending notifications
  Future<List<PendingNotificationRequest>> getPendingNotifications() async {
    return _localNotifications.pendingNotificationRequests();
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

/// Color class for Android notification (simple implementation)
class Color {
  final int value;
  const Color(this.value);
}

/// Notification Service Provider
final notificationServiceProvider = Provider<NotificationService>((ref) {
  final service = NotificationService();
  return service;
});
