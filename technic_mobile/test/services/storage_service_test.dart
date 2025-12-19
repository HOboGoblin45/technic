import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:technic_mobile/models/watchlist_item.dart';
import 'package:technic_mobile/services/storage_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('StorageService', () {
    late StorageService storage;

    setUp(() async {
      SharedPreferences.setMockInitialValues({});
      storage = StorageService.instance;
      await storage.init();
    });

    group('User Management', () {
      test('saveUser and loadUser work correctly', () async {
        await storage.saveUser('user_123');
        final userId = await storage.loadUser();
        expect(userId, 'user_123');
      });

      test('loadUser returns null when no user saved', () async {
        final userId = await storage.loadUser();
        expect(userId, isNull);
      });

      test('clearUser removes stored user', () async {
        await storage.saveUser('user_456');
        await storage.clearUser();
        final userId = await storage.loadUser();
        expect(userId, isNull);
      });
    });

    group('Tab Navigation', () {
      test('saveLastTab and loadLastTab work correctly', () async {
        await storage.saveLastTab(2);
        final tab = await storage.loadLastTab();
        expect(tab, 2);
      });

      test('loadLastTab returns null when not set', () async {
        final tab = await storage.loadLastTab();
        expect(tab, isNull);
      });

      test('handles tab index 0', () async {
        await storage.saveLastTab(0);
        final tab = await storage.loadLastTab();
        expect(tab, 0);
      });
    });

    group('Theme & Preferences', () {
      test('saveThemeMode and loadThemeMode work correctly', () async {
        await storage.saveThemeMode('dark');
        final theme = await storage.loadThemeMode();
        expect(theme, 'dark');
      });

      test('loadThemeMode returns null when not set', () async {
        final theme = await storage.loadThemeMode();
        expect(theme, isNull);
      });

      test('saveOptionsMode and loadOptionsMode work correctly', () async {
        await storage.saveOptionsMode('advanced');
        final mode = await storage.loadOptionsMode();
        expect(mode, 'advanced');
      });
    });

    group('Watchlist', () {
      test('saveWatchlist and loadWatchlist work correctly', () async {
        final items = [
          WatchlistItem(
            ticker: 'AAPL',
            signal: 'bullish',
            addedAt: DateTime(2024, 1, 15),
          ),
          WatchlistItem(
            ticker: 'MSFT',
            note: 'Cloud play',
            tags: ['tech', 'cloud'],
            addedAt: DateTime(2024, 1, 16),
          ),
        ];

        await storage.saveWatchlist(items);
        final loaded = await storage.loadWatchlist();

        expect(loaded.length, 2);
        expect(loaded[0].ticker, 'AAPL');
        expect(loaded[0].signal, 'bullish');
        expect(loaded[1].ticker, 'MSFT');
        expect(loaded[1].note, 'Cloud play');
        expect(loaded[1].tags, ['tech', 'cloud']);
      });

      test('loadWatchlist returns empty list when not set', () async {
        final watchlist = await storage.loadWatchlist();
        expect(watchlist, isEmpty);
      });

      test('handles empty watchlist', () async {
        await storage.saveWatchlist([]);
        final loaded = await storage.loadWatchlist();
        expect(loaded, isEmpty);
      });
    });

    group('Scan History', () {
      test('saveScanHistory and loadScanHistory work correctly', () async {
        final history = [
          {'id': '1', 'timestamp': '2024-01-15T10:00:00.000'},
          {'id': '2', 'timestamp': '2024-01-16T11:00:00.000'},
        ];

        await storage.saveScanHistory(history);
        final loaded = await storage.loadScanHistory();

        expect(loaded.length, 2);
        expect(loaded[0]['id'], '1');
        expect(loaded[1]['id'], '2');
      });

      test('loadScanHistory returns empty list when not set', () async {
        final history = await storage.loadScanHistory();
        expect(history, isEmpty);
      });
    });

    group('Price Alerts', () {
      test('saveAlerts and loadAlerts work correctly', () async {
        final alerts = [
          {'ticker': 'AAPL', 'price': 150.0, 'condition': 'above'},
          {'ticker': 'TSLA', 'price': 200.0, 'condition': 'below'},
        ];

        await storage.saveAlerts(alerts);
        final loaded = await storage.loadAlerts();

        expect(loaded.length, 2);
        expect(loaded[0]['ticker'], 'AAPL');
        expect(loaded[1]['condition'], 'below');
      });

      test('loadAlerts returns empty list when not set', () async {
        final alerts = await storage.loadAlerts();
        expect(alerts, isEmpty);
      });
    });

    group('Onboarding', () {
      test('setOnboardingComplete and isOnboardingComplete work correctly', () async {
        expect(await storage.isOnboardingComplete(), false);

        await storage.setOnboardingComplete(true);
        expect(await storage.isOnboardingComplete(), true);

        await storage.setOnboardingComplete(false);
        expect(await storage.isOnboardingComplete(), false);
      });
    });

    group('Scanner State', () {
      test('saveScannerState and loadScannerState work correctly', () async {
        final lastScan = DateTime(2024, 1, 15, 14, 30);

        await storage.saveScannerState(
          filters: {'sector': 'Technology', 'minICS': '80'},
          savedScreens: [],
          scanCount: 25,
          streakDays: 5,
          lastScan: lastScan,
          advancedMode: true,
          showOnboarding: false,
        );

        final state = await storage.loadScannerState();

        expect(state, isNotNull);
        expect(state!.filters['sector'], 'Technology');
        expect(state.filters['minICS'], '80');
        expect(state.scanCount, 25);
        expect(state.streakDays, 5);
        expect(state.advancedMode, true);
        expect(state.showOnboarding, false);
      });

      test('loadScannerState returns null when not set', () async {
        final state = await storage.loadScannerState();
        expect(state, isNull);
      });
    });

    group('Utility Methods', () {
      test('clearAll removes all stored data', () async {
        await storage.saveUser('test_user');
        await storage.saveLastTab(1);
        await storage.saveThemeMode('dark');

        await storage.clearAll();

        expect(await storage.loadUser(), isNull);
        expect(await storage.loadLastTab(), isNull);
        expect(await storage.loadThemeMode(), isNull);
      });

      test('containsKey returns correct values', () async {
        await storage.saveUser('test_user');

        expect(await storage.containsKey('user_id'), true);
        expect(await storage.containsKey('nonexistent_key'), false);
      });

      test('remove deletes specific key', () async {
        await storage.saveUser('test_user');
        await storage.saveThemeMode('dark');

        await storage.remove('user_id');

        expect(await storage.loadUser(), isNull);
        expect(await storage.loadThemeMode(), 'dark');
      });

      test('getAllKeys returns all stored keys', () async {
        await storage.saveUser('test_user');
        await storage.saveLastTab(0);
        await storage.saveThemeMode('dark');

        final keys = await storage.getAllKeys();

        expect(keys.contains('user_id'), true);
        expect(keys.contains('last_tab'), true);
        expect(keys.contains('theme_mode'), true);
      });
    });

    group('SavedScreen', () {
      test('fromJson parses correctly', () {
        final json = {
          'name': 'Momentum Screen',
          'description': 'Find momentum stocks',
          'horizon': 'swing',
          'isActive': true,
          'params': {'minICS': '75', 'sector': 'Tech'},
        };

        final screen = SavedScreen.fromJson(json);

        expect(screen.name, 'Momentum Screen');
        expect(screen.description, 'Find momentum stocks');
        expect(screen.horizon, 'swing');
        expect(screen.isActive, true);
        expect(screen.params?['minICS'], '75');
      });

      test('toJson serializes correctly', () {
        const screen = SavedScreen(
          'Value Screen',
          'Find undervalued stocks',
          'position',
          false,
          params: {'maxPE': '20'},
        );

        final json = screen.toJson();

        expect(json['name'], 'Value Screen');
        expect(json['description'], 'Find undervalued stocks');
        expect(json['horizon'], 'position');
        expect(json['isActive'], false);
        expect(json['params']['maxPE'], '20');
      });

      test('copyWith creates copy with updated fields', () {
        const original = SavedScreen(
          'Original',
          'Original desc',
          'swing',
          true,
        );

        final updated = original.copyWith(
          name: 'Updated',
          isActive: false,
        );

        expect(updated.name, 'Updated');
        expect(updated.description, 'Original desc');
        expect(updated.isActive, false);
        expect(original.name, 'Original');
      });
    });

    group('Error Handling', () {
      test('loadWatchlist handles malformed JSON gracefully', () async {
        SharedPreferences.setMockInitialValues({
          'watchlist': 'not valid json',
        });
        await storage.init();

        final watchlist = await storage.loadWatchlist();
        expect(watchlist, isEmpty);
      });

      test('loadScanHistory handles malformed JSON gracefully', () async {
        SharedPreferences.setMockInitialValues({
          'scan_history': 'invalid json data',
        });
        await storage.init();

        final history = await storage.loadScanHistory();
        expect(history, isEmpty);
      });

      test('loadAlerts handles malformed JSON gracefully', () async {
        SharedPreferences.setMockInitialValues({
          'price_alerts': '{broken json',
        });
        await storage.init();

        final alerts = await storage.loadAlerts();
        expect(alerts, isEmpty);
      });
    });
  });
}
