/// App Providers
/// 
/// Central location for all Riverpod providers used throughout the app.
/// Provides global state management for theme, API, storage, and app state.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/api_service.dart';
import '../services/storage_service.dart';
import '../services/auth_service.dart';
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

/// Auth Service Provider (singleton)
final authServiceProvider = Provider<AuthService>((ref) {
  return AuthService();
});

// ============================================================================
// THEME PROVIDERS
// ============================================================================

/// Theme Mode Provider (dark/light)
final themeModeProvider = StateNotifierProvider<ThemeModeNotifier, bool>((ref) {
  return ThemeModeNotifier(ref.read(storageServiceProvider));
});

class ThemeModeNotifier extends StateNotifier<bool> {
  ThemeModeNotifier(this._storage) : super(true) {  // Default to dark mode
    _loadThemeMode();
  }

  final StorageService _storage;

  Future<void> _loadThemeMode() async {
    try {
      final mode = await _storage.loadThemeMode();
      // Default to dark if no preference saved
      state = mode == null ? true : mode == 'dark';
    } catch (e) {
      // Default to dark mode on error
      state = true;
    }
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
    try {
      final mode = await _storage.loadOptionsMode();
      state = mode ?? 'stock_plus_options';
    } catch (e) {
      // Default to stock_plus_options on error
      state = 'stock_plus_options';
    }
  }

  Future<void> setMode(String mode) async {
    state = mode;
    await _storage.saveOptionsMode(mode);
  }
}

// ============================================================================
// AUTHENTICATION PROVIDERS
// ============================================================================

/// Authentication State
class AuthState {
  final User? user;
  final bool isLoading;
  final String? error;
  final bool isAuthenticated;

  const AuthState({
    this.user,
    this.isLoading = false,
    this.error,
    this.isAuthenticated = false,
  });

  AuthState copyWith({
    User? user,
    bool? isLoading,
    String? error,
    bool? isAuthenticated,
  }) {
    return AuthState(
      user: user ?? this.user,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      isAuthenticated: isAuthenticated ?? this.isAuthenticated,
    );
  }

  AuthState clearError() {
    return AuthState(
      user: user,
      isLoading: isLoading,
      error: null,
      isAuthenticated: isAuthenticated,
    );
  }
}

/// Auth Provider
final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  return AuthNotifier(ref.read(authServiceProvider));
});

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier(this._authService) : super(const AuthState()) {
    _checkAuthStatus();
  }

  final AuthService _authService;

  /// Check if user is already authenticated on app start
  Future<void> _checkAuthStatus() async {
    try {
      final isAuth = await _authService.isAuthenticated();
      if (isAuth) {
        final user = await _authService.getCurrentUser();
        if (user != null) {
          state = AuthState(
            user: user,
            isAuthenticated: true,
          );
        }
      }
    } catch (e) {
      // Silent fail - user just needs to login
    }
  }

  /// Login with email and password
  Future<bool> login(String email, String password) async {
    state = state.copyWith(isLoading: true, error: null);
    
    try {
      final response = await _authService.login(email, password);
      state = AuthState(
        user: response.user,
        isAuthenticated: true,
        isLoading: false,
      );
      return true;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString().replaceAll('Exception: ', ''),
      );
      return false;
    }
  }

  /// Sign up new user
  Future<bool> signup(String email, String password, String name) async {
    state = state.copyWith(isLoading: true, error: null);
    
    try {
      final response = await _authService.signup(email, password, name);
      state = AuthState(
        user: response.user,
        isAuthenticated: true,
        isLoading: false,
      );
      return true;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString().replaceAll('Exception: ', ''),
      );
      return false;
    }
  }

  /// Logout user
  Future<void> logout() async {
    state = state.copyWith(isLoading: true);
    
    try {
      await _authService.logout();
      state = const AuthState();
    } catch (e) {
      // Even if logout fails, clear local state
      state = const AuthState();
    }
  }

  /// Clear error message
  void clearError() {
    state = state.clearError();
  }

  /// Refresh authentication status
  Future<void> refresh() async {
    await _checkAuthStatus();
  }

  /// Try to auto-login on app start
  Future<void> tryAutoLogin() async {
    await _checkAuthStatus();
  }
}

// ============================================================================
// USER PROVIDER (Legacy - kept for backward compatibility)
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
    try {
      state = await _storage.loadUser();
    } catch (e) {
      // Default to null (not logged in) on error
      state = null;
    }
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
    try {
      // Load watchlist from storage
      final items = await _storage.loadWatchlist();
      state = items;
    } catch (e) {
      // Default to empty watchlist on error
      state = [];
    }
  }

  Future<void> add(String ticker, {String? signal, String? note, List<String>? tags}) async {
    final item = WatchlistItem(
      ticker: ticker,
      signal: signal,
      note: note,
      tags: tags ?? [],
      addedAt: DateTime.now(),
    );
    state = [...state, item];
    await _saveWatchlist();
  }

  Future<void> remove(String ticker) async {
    state = state.where((item) => item.ticker != ticker).toList();
    await _saveWatchlist();
  }

  Future<void> toggle(String ticker, {String? signal, String? note, List<String>? tags}) async {
    if (contains(ticker)) {
      await remove(ticker);
    } else {
      await add(ticker, signal: signal, note: note, tags: tags);
    }
  }

  /// Update note for a watchlist item
  Future<void> updateNote(String ticker, String? note) async {
    state = state.map((item) {
      if (item.ticker == ticker) {
        return item.copyWith(note: note);
      }
      return item;
    }).toList();
    await _saveWatchlist();
  }

  /// Update tags for a watchlist item
  Future<void> updateTags(String ticker, List<String> tags) async {
    state = state.map((item) {
      if (item.ticker == ticker) {
        return item.copyWith(tags: tags);
      }
      return item;
    }).toList();
    await _saveWatchlist();
  }

  /// Get watchlist item by ticker
  WatchlistItem? getItem(String ticker) {
    try {
      return state.firstWhere((item) => item.ticker == ticker);
    } catch (e) {
      return null;
    }
  }

  /// Filter watchlist by tags
  List<WatchlistItem> filterByTags(List<String> tags) {
    if (tags.isEmpty) return state;
    return state.where((item) {
      return tags.any((tag) => item.tags.contains(tag));
    }).toList();
  }

  /// Search watchlist by ticker or note
  List<WatchlistItem> search(String query) {
    if (query.isEmpty) return state;
    final lowerQuery = query.toLowerCase();
    return state.where((item) {
      return item.ticker.toLowerCase().contains(lowerQuery) ||
             (item.note?.toLowerCase().contains(lowerQuery) ?? false);
    }).toList();
  }

  /// Get all unique tags from watchlist
  List<String> getAllTags() {
    final allTags = <String>{};
    for (final item in state) {
      allTags.addAll(item.tags);
    }
    return allTags.toList()..sort();
  }

  bool contains(String ticker) {
    return state.any((item) => item.ticker == ticker);
  }

  Future<void> _saveWatchlist() async {
    try {
      // Save watchlist to storage
      await _storage.saveWatchlist(state);
    } catch (e) {
      // Silent fail - storage error shouldn't crash the app
    }
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
    try {
      final lastTab = await _storage.loadLastTab();
      if (lastTab != null) {
        state = lastTab;
      }
    } catch (e) {
      // Default to first tab on error
      state = 0;
    }
  }

  Future<void> setTab(int index) async {
    state = index;
    await _storage.saveLastTab(index);
  }
}
