import 'package:flutter_test/flutter_test.dart';
import 'package:technic_mobile/models/watchlist_item.dart';

void main() {
  group('WatchlistItem', () {
    final testDate = DateTime(2024, 1, 15, 10, 30);

    group('constructor', () {
      test('creates instance with required fields', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          addedAt: testDate,
        );

        expect(item.ticker, 'AAPL');
        expect(item.addedAt, testDate);
        expect(item.signal, isNull);
        expect(item.note, isNull);
        expect(item.tags, isEmpty);
      });

      test('creates instance with all fields', () {
        final item = WatchlistItem(
          ticker: 'TSLA',
          signal: 'bullish',
          note: 'EV momentum play',
          tags: ['tech', 'ev', 'growth'],
          addedAt: testDate,
        );

        expect(item.ticker, 'TSLA');
        expect(item.signal, 'bullish');
        expect(item.note, 'EV momentum play');
        expect(item.tags, ['tech', 'ev', 'growth']);
      });
    });

    group('fromJson', () {
      test('parses all fields correctly', () {
        final json = {
          'ticker': 'NVDA',
          'signal': 'bullish',
          'note': 'AI leader',
          'tags': ['ai', 'semiconductor'],
          'addedAt': '2024-01-15T10:30:00.000',
        };

        final item = WatchlistItem.fromJson(json);

        expect(item.ticker, 'NVDA');
        expect(item.signal, 'bullish');
        expect(item.note, 'AI leader');
        expect(item.tags, ['ai', 'semiconductor']);
        expect(item.addedAt.year, 2024);
        expect(item.addedAt.month, 1);
        expect(item.addedAt.day, 15);
      });

      test('handles missing optional fields', () {
        final json = {
          'ticker': 'MSFT',
          'addedAt': '2024-02-01T09:00:00.000',
        };

        final item = WatchlistItem.fromJson(json);

        expect(item.ticker, 'MSFT');
        expect(item.signal, isNull);
        expect(item.note, isNull);
        expect(item.tags, isEmpty);
      });

      test('handles null values gracefully', () {
        final json = {
          'ticker': 'GOOGL',
          'signal': null,
          'note': null,
          'tags': null,
          'addedAt': null,
        };

        final item = WatchlistItem.fromJson(json);

        expect(item.ticker, 'GOOGL');
        expect(item.signal, isNull);
        expect(item.tags, isEmpty);
        // addedAt defaults to now when null
        expect(item.addedAt.day, DateTime.now().day);
      });

      test('handles empty ticker gracefully', () {
        final json = <String, dynamic>{};

        final item = WatchlistItem.fromJson(json);

        expect(item.ticker, '');
      });
    });

    group('toJson', () {
      test('serializes all fields correctly', () {
        final item = WatchlistItem(
          ticker: 'AMD',
          signal: 'neutral',
          note: 'Watching support',
          tags: ['chip', 'value'],
          addedAt: testDate,
        );

        final json = item.toJson();

        expect(json['ticker'], 'AMD');
        expect(json['signal'], 'neutral');
        expect(json['note'], 'Watching support');
        expect(json['tags'], ['chip', 'value']);
        expect(json['addedAt'], testDate.toIso8601String());
      });

      test('roundtrip serialization preserves data', () {
        final original = WatchlistItem(
          ticker: 'META',
          signal: 'bullish',
          note: 'Social media leader',
          tags: ['social', 'advertising'],
          addedAt: testDate,
        );

        final json = original.toJson();
        final restored = WatchlistItem.fromJson(json);

        expect(restored.ticker, original.ticker);
        expect(restored.signal, original.signal);
        expect(restored.note, original.note);
        expect(restored.tags, original.tags);
        expect(restored.addedAt.toIso8601String(), original.addedAt.toIso8601String());
      });
    });

    group('hasSignal', () {
      test('returns true when signal is set', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          signal: 'bullish',
          addedAt: testDate,
        );

        expect(item.hasSignal, true);
      });

      test('returns false when signal is null', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          addedAt: testDate,
        );

        expect(item.hasSignal, false);
      });

      test('returns false when signal is empty', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          signal: '',
          addedAt: testDate,
        );

        expect(item.hasSignal, false);
      });
    });

    group('hasNote', () {
      test('returns true when note is set', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          note: 'Some note',
          addedAt: testDate,
        );

        expect(item.hasNote, true);
      });

      test('returns false when note is null', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          addedAt: testDate,
        );

        expect(item.hasNote, false);
      });

      test('returns false when note is empty', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          note: '',
          addedAt: testDate,
        );

        expect(item.hasNote, false);
      });
    });

    group('hasTags', () {
      test('returns true when tags exist', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          tags: ['tech'],
          addedAt: testDate,
        );

        expect(item.hasTags, true);
      });

      test('returns false when tags are empty', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          addedAt: testDate,
        );

        expect(item.hasTags, false);
      });
    });

    group('daysSinceAdded', () {
      test('calculates days correctly', () {
        final threeDaysAgo = DateTime.now().subtract(const Duration(days: 3));
        final item = WatchlistItem(
          ticker: 'AAPL',
          addedAt: threeDaysAgo,
        );

        expect(item.daysSinceAdded, 3);
      });

      test('returns 0 for item added today', () {
        final item = WatchlistItem(
          ticker: 'AAPL',
          addedAt: DateTime.now(),
        );

        expect(item.daysSinceAdded, 0);
      });
    });

    group('copyWith', () {
      test('creates copy with updated ticker', () {
        final original = WatchlistItem(
          ticker: 'AAPL',
          signal: 'bullish',
          addedAt: testDate,
        );

        final updated = original.copyWith(ticker: 'MSFT');

        expect(updated.ticker, 'MSFT');
        expect(updated.signal, 'bullish');
        expect(updated.addedAt, testDate);
        expect(original.ticker, 'AAPL'); // Original unchanged
      });

      test('creates copy with updated signal', () {
        final original = WatchlistItem(
          ticker: 'AAPL',
          signal: 'bullish',
          addedAt: testDate,
        );

        final updated = original.copyWith(signal: 'bearish');

        expect(updated.ticker, 'AAPL');
        expect(updated.signal, 'bearish');
      });

      test('creates copy with updated note', () {
        final original = WatchlistItem(
          ticker: 'AAPL',
          note: 'Original note',
          addedAt: testDate,
        );

        final updated = original.copyWith(note: 'Updated note');

        expect(updated.note, 'Updated note');
      });

      test('creates copy with updated tags', () {
        final original = WatchlistItem(
          ticker: 'AAPL',
          tags: ['tech'],
          addedAt: testDate,
        );

        final updated = original.copyWith(tags: ['tech', 'value']);

        expect(updated.tags, ['tech', 'value']);
        expect(original.tags, ['tech']); // Original unchanged
      });

      test('creates copy with multiple updated fields', () {
        final original = WatchlistItem(
          ticker: 'AAPL',
          signal: 'neutral',
          note: 'Original',
          tags: ['old'],
          addedAt: testDate,
        );

        final newDate = DateTime(2024, 6, 1);
        final updated = original.copyWith(
          signal: 'bullish',
          note: 'New note',
          tags: ['new', 'tags'],
          addedAt: newDate,
        );

        expect(updated.ticker, 'AAPL');
        expect(updated.signal, 'bullish');
        expect(updated.note, 'New note');
        expect(updated.tags, ['new', 'tags']);
        expect(updated.addedAt, newDate);
      });
    });
  });
}
