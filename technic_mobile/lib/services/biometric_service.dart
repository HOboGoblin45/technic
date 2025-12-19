/// Biometric Authentication Service
///
/// Handles Face ID and Touch ID authentication for secure app access.
/// Uses local_auth package for iOS biometric APIs.
library;

import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:local_auth/local_auth.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Biometric type available on device
enum BiometricType {
  faceId,
  touchId,
  none,
}

/// Biometric authentication result
enum BiometricResult {
  success,
  failed,
  cancelled,
  notAvailable,
  notEnrolled,
  lockedOut,
  error,
}

/// Biometric Authentication Service
class BiometricService {
  BiometricService({
    LocalAuthentication? localAuth,
    FlutterSecureStorage? storage,
  })  : _localAuth = localAuth ?? LocalAuthentication(),
        _storage = storage ?? const FlutterSecureStorage();

  final LocalAuthentication _localAuth;
  final FlutterSecureStorage _storage;

  // Storage keys
  static const String _biometricEnabledKey = 'biometric_enabled';
  static const String _biometricSetupCompleteKey = 'biometric_setup_complete';

  /// Check if biometric authentication is available on this device
  ///
  /// Returns true if device has biometric hardware and user has enrolled.
  Future<bool> isAvailable() async {
    try {
      // Only supported on iOS
      if (!Platform.isIOS) return false;

      // Check if device supports biometrics
      final canCheckBiometrics = await _localAuth.canCheckBiometrics;
      final isDeviceSupported = await _localAuth.isDeviceSupported();

      return canCheckBiometrics && isDeviceSupported;
    } catch (e) {
      debugPrint('[Biometric] Error checking availability: $e');
      return false;
    }
  }

  /// Get the type of biometric available on this device
  ///
  /// Returns BiometricType.faceId, BiometricType.touchId, or BiometricType.none.
  Future<BiometricType> getAvailableBiometricType() async {
    try {
      if (!Platform.isIOS) return BiometricType.none;

      final availableBiometrics = await _localAuth.getAvailableBiometrics();

      if (availableBiometrics.contains(BiometricType.face)) {
        return BiometricType.faceId;
      } else if (availableBiometrics.contains(BiometricType.fingerprint)) {
        return BiometricType.touchId;
      }

      return BiometricType.none;
    } catch (e) {
      debugPrint('[Biometric] Error getting biometric type: $e');
      return BiometricType.none;
    }
  }

  /// Get user-friendly name for the biometric type
  Future<String> getBiometricName() async {
    final type = await getAvailableBiometricType();
    switch (type) {
      case BiometricType.faceId:
        return 'Face ID';
      case BiometricType.touchId:
        return 'Touch ID';
      case BiometricType.none:
        return 'Biometric';
    }
  }

  /// Check if biometric authentication is enabled by user
  Future<bool> isEnabled() async {
    try {
      final enabled = await _storage.read(key: _biometricEnabledKey);
      return enabled == 'true';
    } catch (e) {
      debugPrint('[Biometric] Error checking enabled status: $e');
      return false;
    }
  }

  /// Enable biometric authentication
  ///
  /// Must authenticate first to enable.
  Future<BiometricResult> enable() async {
    try {
      // First verify biometrics work
      final result = await authenticate(
        reason: 'Authenticate to enable biometric login',
      );

      if (result == BiometricResult.success) {
        await _storage.write(key: _biometricEnabledKey, value: 'true');
        await _storage.write(key: _biometricSetupCompleteKey, value: 'true');
        debugPrint('[Biometric] Biometric authentication enabled');
      }

      return result;
    } catch (e) {
      debugPrint('[Biometric] Error enabling biometrics: $e');
      return BiometricResult.error;
    }
  }

  /// Disable biometric authentication
  Future<void> disable() async {
    try {
      await _storage.write(key: _biometricEnabledKey, value: 'false');
      debugPrint('[Biometric] Biometric authentication disabled');
    } catch (e) {
      debugPrint('[Biometric] Error disabling biometrics: $e');
    }
  }

  /// Authenticate using biometrics
  ///
  /// Shows system biometric prompt and returns result.
  Future<BiometricResult> authenticate({
    String reason = 'Authenticate to access Technic',
  }) async {
    try {
      // Check availability first
      if (!await isAvailable()) {
        return BiometricResult.notAvailable;
      }

      debugPrint('[Biometric] Requesting authentication');

      final authenticated = await _localAuth.authenticate(
        localizedReason: reason,
        options: const AuthenticationOptions(
          stickyAuth: true,
          biometricOnly: true,
          useErrorDialogs: true,
          sensitiveTransaction: true,
        ),
      );

      if (authenticated) {
        debugPrint('[Biometric] Authentication successful');
        return BiometricResult.success;
      } else {
        debugPrint('[Biometric] Authentication failed');
        return BiometricResult.failed;
      }
    } on PlatformException catch (e) {
      debugPrint('[Biometric] Platform exception: ${e.code} - ${e.message}');

      switch (e.code) {
        case 'NotAvailable':
          return BiometricResult.notAvailable;
        case 'NotEnrolled':
          return BiometricResult.notEnrolled;
        case 'LockedOut':
        case 'PermanentlyLockedOut':
          return BiometricResult.lockedOut;
        case 'PasscodeNotSet':
          return BiometricResult.notAvailable;
        default:
          // User cancelled or other error
          if (e.message?.contains('cancel') == true) {
            return BiometricResult.cancelled;
          }
          return BiometricResult.error;
      }
    } catch (e) {
      debugPrint('[Biometric] Error during authentication: $e');
      return BiometricResult.error;
    }
  }

  /// Authenticate for app unlock (shows appropriate UI messages)
  Future<BiometricResult> authenticateForAppUnlock() async {
    final biometricName = await getBiometricName();
    return authenticate(
      reason: 'Use $biometricName to unlock Technic',
    );
  }

  /// Authenticate for sensitive action (like viewing account details)
  Future<BiometricResult> authenticateForSensitiveAction(String action) async {
    final biometricName = await getBiometricName();
    return authenticate(
      reason: 'Use $biometricName to $action',
    );
  }

  /// Check if biometric setup has been completed
  Future<bool> isSetupComplete() async {
    try {
      final complete = await _storage.read(key: _biometricSetupCompleteKey);
      return complete == 'true';
    } catch (e) {
      return false;
    }
  }

  /// Mark that user has been prompted about biometrics (skip future prompts)
  Future<void> markSetupComplete() async {
    try {
      await _storage.write(key: _biometricSetupCompleteKey, value: 'true');
    } catch (e) {
      debugPrint('[Biometric] Error marking setup complete: $e');
    }
  }

  /// Reset all biometric settings
  Future<void> reset() async {
    try {
      await _storage.delete(key: _biometricEnabledKey);
      await _storage.delete(key: _biometricSetupCompleteKey);
      debugPrint('[Biometric] Settings reset');
    } catch (e) {
      debugPrint('[Biometric] Error resetting settings: $e');
    }
  }

  /// Get error message for biometric result
  String getErrorMessage(BiometricResult result) {
    switch (result) {
      case BiometricResult.success:
        return 'Authentication successful';
      case BiometricResult.failed:
        return 'Authentication failed. Please try again.';
      case BiometricResult.cancelled:
        return 'Authentication cancelled';
      case BiometricResult.notAvailable:
        return 'Biometric authentication is not available on this device';
      case BiometricResult.notEnrolled:
        return 'No biometrics enrolled. Please set up Face ID or Touch ID in Settings.';
      case BiometricResult.lockedOut:
        return 'Too many failed attempts. Please use your passcode.';
      case BiometricResult.error:
        return 'An error occurred. Please try again.';
    }
  }
}
