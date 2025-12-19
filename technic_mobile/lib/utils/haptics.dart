/// Haptic Feedback Utilities
///
/// Provides consistent haptic feedback across the app using iOS
/// Taptic Engine for tactile user feedback.
library;

import 'dart:io';
import 'package:flutter/services.dart';

/// Haptic feedback utility class
///
/// Usage:
/// - Haptics.light() - For UI selection feedback
/// - Haptics.medium() - For successful actions
/// - Haptics.heavy() - For errors or warnings
/// - Haptics.selection() - For picker/slider changes
/// - Haptics.success() - For completed actions
/// - Haptics.warning() - For warnings
/// - Haptics.error() - For errors
class Haptics {
  Haptics._();

  static bool _enabled = true;

  /// Enable or disable haptic feedback globally
  static void setEnabled(bool enabled) {
    _enabled = enabled;
  }

  /// Check if haptics are enabled
  static bool get isEnabled => _enabled;

  /// Light impact feedback
  ///
  /// Use for: selection changes, button taps, list item selection
  static void light() {
    if (!_shouldVibrate()) return;
    HapticFeedback.lightImpact();
  }

  /// Medium impact feedback
  ///
  /// Use for: successful actions, toggles, confirmations
  static void medium() {
    if (!_shouldVibrate()) return;
    HapticFeedback.mediumImpact();
  }

  /// Heavy impact feedback
  ///
  /// Use for: errors, warnings, important alerts
  static void heavy() {
    if (!_shouldVibrate()) return;
    HapticFeedback.heavyImpact();
  }

  /// Selection feedback
  ///
  /// Use for: picker value changes, slider movements, segment changes
  static void selection() {
    if (!_shouldVibrate()) return;
    HapticFeedback.selectionClick();
  }

  /// Success notification feedback
  ///
  /// Use for: completed actions, successful submissions, confirmations
  static void success() {
    if (!_shouldVibrate()) return;
    // iOS uses notification feedback for success
    HapticFeedback.mediumImpact();
  }

  /// Warning notification feedback
  ///
  /// Use for: warnings, cautions, attention needed
  static void warning() {
    if (!_shouldVibrate()) return;
    // Double tap pattern for warning
    HapticFeedback.heavyImpact();
  }

  /// Error notification feedback
  ///
  /// Use for: errors, failures, invalid inputs
  static void error() {
    if (!_shouldVibrate()) return;
    // Strong feedback for error
    HapticFeedback.heavyImpact();
  }

  /// Button press feedback
  ///
  /// Standard feedback for button interactions
  static void button() {
    if (!_shouldVibrate()) return;
    HapticFeedback.lightImpact();
  }

  /// Toggle switch feedback
  ///
  /// Feedback when toggling switches
  static void toggle() {
    if (!_shouldVibrate()) return;
    HapticFeedback.mediumImpact();
  }

  /// Pull to refresh feedback
  ///
  /// Feedback when pull-to-refresh threshold is reached
  static void refresh() {
    if (!_shouldVibrate()) return;
    HapticFeedback.mediumImpact();
  }

  /// Swipe action feedback
  ///
  /// Feedback for swipe-to-delete or swipe actions
  static void swipe() {
    if (!_shouldVibrate()) return;
    HapticFeedback.lightImpact();
  }

  /// Alert trigger feedback
  ///
  /// Feedback when a price alert is triggered
  static void alertTriggered() {
    if (!_shouldVibrate()) return;
    HapticFeedback.heavyImpact();
  }

  /// Scan complete feedback
  ///
  /// Feedback when stock scan completes
  static void scanComplete() {
    if (!_shouldVibrate()) return;
    HapticFeedback.mediumImpact();
  }

  /// Add to watchlist feedback
  ///
  /// Feedback when adding stock to watchlist
  static void addToWatchlist() {
    if (!_shouldVibrate()) return;
    HapticFeedback.mediumImpact();
  }

  /// Remove from watchlist feedback
  ///
  /// Feedback when removing stock from watchlist
  static void removeFromWatchlist() {
    if (!_shouldVibrate()) return;
    HapticFeedback.lightImpact();
  }

  /// Tab selection feedback
  ///
  /// Feedback when switching bottom navigation tabs
  static void tabSwitch() {
    if (!_shouldVibrate()) return;
    HapticFeedback.selectionClick();
  }

  /// Long press feedback
  ///
  /// Feedback for long press actions
  static void longPress() {
    if (!_shouldVibrate()) return;
    HapticFeedback.mediumImpact();
  }

  /// Check if device should vibrate
  static bool _shouldVibrate() {
    // Only vibrate on iOS and when enabled
    return _enabled && Platform.isIOS;
  }
}

/// Extension for convenient haptic feedback on widgets
extension HapticFeedbackExtension on Function {
  /// Wrap a callback with haptic feedback
  ///
  /// Example:
  /// onTap: () => doSomething().withHaptic(Haptics.light)
  void withHaptic(void Function() haptic) {
    haptic();
    this.call();
  }
}
