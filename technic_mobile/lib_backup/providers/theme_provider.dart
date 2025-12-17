/// Theme Provider
/// 
/// Manages app theme mode (dark/light) with persistence
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/storage_service.dart';

/// Theme mode enum
enum AppThemeMode {
  light,
  dark,
  system,
}

/// Theme Provider
final themeProvider = StateNotifierProvider<ThemeNotifier, AppThemeMode>((ref) {
  return ThemeNotifier(ref.read(storageServiceProvider));
});

class ThemeNotifier extends StateNotifier<AppThemeMode> {
  ThemeNotifier(this._storage) : super(AppThemeMode.dark) {
    _loadTheme();
  }

  final StorageService _storage;

  Future<void> _loadTheme() async {
    final themeMode = await _storage.loadThemeMode();
    if (themeMode != null) {
      switch (themeMode) {
        case 'light':
          state = AppThemeMode.light;
          break;
        case 'dark':
          state = AppThemeMode.dark;
          break;
        case 'system':
          state = AppThemeMode.system;
          break;
      }
    }
  }

  /// Set theme mode
  Future<void> setTheme(AppThemeMode mode) async {
    state = mode;
    await _storage.saveThemeMode(mode.name);
  }

  /// Toggle between light and dark
  Future<void> toggleTheme() async {
    if (state == AppThemeMode.light) {
      await setTheme(AppThemeMode.dark);
    } else {
      await setTheme(AppThemeMode.light);
    }
  }

  /// Get Flutter ThemeMode
  ThemeMode get themeMode {
    switch (state) {
      case AppThemeMode.light:
        return ThemeMode.light;
      case AppThemeMode.dark:
        return ThemeMode.dark;
      case AppThemeMode.system:
        return ThemeMode.system;
    }
  }
}

/// Storage Service Provider
final storageServiceProvider = Provider<StorageService>((ref) {
  return StorageService.instance;
});
