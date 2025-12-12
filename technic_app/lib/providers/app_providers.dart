/// App Providers
/// 
/// Central location for all Riverpod providers used throughout the app.
/// Provides global state management for theme, API, storage, and app state.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/api_service.dart';
import '../services/storage_service.dart';
import '../models/scan_result.dart';
import '../models/market_mover.dart';
import '../models/watchlist_item.dart';

// ============================================================================
// SERVICE PROVIDERS
// ============================================================================

/// API Service Provider (singleton)
final apiServiceProvider = Provider<ApiService>((ref) {
  return ApiService();
});

/// Storage Service Provider (singleton)
final storageServiceProvider = Provider<StorageService>((ref) {
  return StorageService.instance;
});

// ============================================================================
// THEME PROVIDERS
// ============================================================================

/// Theme Mode Provider (dark/light)
final themeModeProvider = StateNotifierProvider<ThemeModeNotifier, bool>((ref) {
  return ThemeModeNotifier(ref.read(storageServiceProvider));
});

class ThemeModeNotifier extends StateNotifier<bool> {
  ThemeModeNotifier(this._storage) : super(false) {
    _loadThemeMode();
  }

  final StorageService _storage;

  Future<void> _loadThemeMode() async {
    final mode = await _storage.loadThemeMode();
    state = mode == 'dark';
  }

  Future<void> setDarkMode(bool isDark) async {
    state = isDark;
    await _storage.saveThemeMode(isDark ? 'dark' : 'light');
  }

  Future<void> toggleTheme() async {
    await setDarkMode(!state);
  }
}

// ============================================================================
// OPTIONS MODE PROVIDER
// ============================================================================

/// Options Mode Provider (stock_only / stock_plus_options)
final optionsModeProvider = StateNotifierProvider<OptionsModeNotifier, String>((ref) {
  return OptionsModeNotifier(ref.read(storageServiceProvider));
});

class OptionsModeNotifier extends StateNotifier<String> {
  OptionsModeNotifier(this._storage) : super('stock_plus_options') {
    _loadOptionsMode();
  }

  final StorageService _storage;

  Future<void> _loadOptionsMode() async {
    final mode = await _storage.loadOptionsMode();
    state = mode ?? 'stock_plus_options';
  }

  Future<void> setMode(String mode) async {
    state = mode;
    await _storage.saveOptionsMode(mode);
  }
}

// ============================================================================
// USER PROVIDER
// ============================================================================

/// User ID Provider
final userIdProvider = StateNotifierProvider<UserIdNotifier, String?>((ref) {
  return UserIdNotifier(ref.read(storageServiceProvider));
});

class UserIdNotifier extends StateNotifier<String?> {
  UserIdNotifier(this._storage) : super(null) {
    _loadUserId();
  }

  final StorageService _storage;

  Future<void> _loadUserId() async {
    state = await _storage.loadUser();
  }

  Future<void> signIn(String userId) async {
    state = userId;
    await _storage.saveUser(userId);
  }

  Future<void> signOut() async {
    state = null;
    await _storage.clearUser();
  }
}

// ============================================================================
// SCANNER STATE PROVIDERS
// ============================================================================

/// Last Scan Results Provider
final lastScanResultsProvider = StateProvider<List<ScanResult>>((ref) {
  return [];
});

/// Last Market Movers Provider
final lastMoversProvider = StateProvider<List<MarketMover>>((ref) {
  return [];
});

/// Scanner Loading State Provider
final scannerLoadingProvider = StateProvider<bool>((ref) {
  return false;
});

/// Scanner Progress Text Provider
final scannerProgressProvider = StateProvider<String?>((ref) {
  return null;
});

// ============================================================================
// COPILOT PROVIDERS
// ============================================================================

/// Copilot Status Provider (for offline/error states)
final copilotStatusProvider = StateProvider<String?>((ref) {
  return null;
});

/// Copilot Context Provider (current symbol being analyzed)
final copilotContextProvider = StateProvider<ScanResult?>((ref) {
  return null;
});

/// Copilot Prefill Provider (suggested prompt)
final copilotPrefillProvider = StateProvider<String?>((ref) {
  return null;
});

// ============================================================================
// WATCHLIST PROVIDER
// ============================================================================

/// Watchlist Provider
final watchlistProvider = StateNotifierProvider<WatchlistNotifier, List<WatchlistItem>>((ref) {
  return WatchlistNotifier(ref.read(storageServiceProvider));
});

class WatchlistNotifier extends StateNotifier<List<WatchlistItem>> {
  WatchlistNotifier(this._storage) : super([]) {
    _loadWatchlist();
  }

  final StorageService _storage;

  Future<void> _loadWatchlist() async {
    // Load watchlist from storage
    final items = await _storage.loadWatchlist();
    state = items;
  }

  Future<void> add(String ticker, {String? signal, String? note}) async {
    final item = WatchlistItem(
      ticker: ticker,
      signal: signal,
      note: note,
      addedAt: DateTime.now(),
    );
    state = [...state, item];
    await _saveWatchlist();
  }

  Future<void> remove(String ticker) async {
    state = state.where((item) => item.ticker != ticker).toList();
    await _saveWatchlist();
  }

  Future<void> toggle(String ticker, {String? signal, String? note}) async {
    if (contains(ticker)) {
      await remove(ticker);
    } else {
      await add(ticker, signal: signal, note: note);
    }
  }

  bool contains(String ticker) {
    return state.any((item) => item.ticker == ticker);
  }

  Future<void> _saveWatchlist() async {
    // Save watchlist to storage
    await _storage.saveWatchlist(state);
  }
}

// ============================================================================
// NAVIGATION PROVIDER
// ============================================================================

/// Current Tab Index Provider
final currentTabProvider = StateNotifierProvider<CurrentTabNotifier, int>((ref) {
  return CurrentTabNotifier(ref.read(storageServiceProvider));
});

class CurrentTabNotifier extends StateNotifier<int> {
  CurrentTabNotifier(this._storage) : super(0) {
    _loadLastTab();
  }

  final StorageService _storage;

  Future<void> _loadLastTab() async {
    final lastTab = await _storage.loadLastTab();
    if (lastTab != null) {
      state = lastTab;
    }
  }

  Future<void> setTab(int index) async {
    state = index;
    await _storage.saveLastTab(index);
  }
}
