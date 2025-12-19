import 'package:flutter_test/flutter_test.dart';
import 'package:technic_mobile/models/scan_result.dart';

void main() {
  group('ScanResult', () {
    group('constructor', () {
      test('creates instance with required fields', () {
        const result = ScanResult(
          'AAPL',
          'bullish',
          '3:1',
          '150.00',
          '145.00',
          '160.00',
          'Strong momentum setup',
        );

        expect(result.ticker, 'AAPL');
        expect(result.signal, 'bullish');
        expect(result.rrr, '3:1');
        expect(result.entry, '150.00');
        expect(result.stop, '145.00');
        expect(result.target, '160.00');
        expect(result.note, 'Strong momentum setup');
      });

      test('creates instance with optional fields', () {
        const result = ScanResult(
          'TSLA',
          'bullish',
          '2:1',
          '200.00',
          '190.00',
          '220.00',
          'EV momentum',
          [195.0, 198.0, 200.0, 202.0],
          85.5,
          'CORE',
          0.72,
          88.0,
          'swing',
          false,
          'aggressive_growth',
          'Aggressive Growth',
          'Technology',
          'Electric Vehicles',
        );

        expect(result.sparkline, [195.0, 198.0, 200.0, 202.0]);
        expect(result.institutionalCoreScore, 85.5);
        expect(result.icsTier, 'CORE');
        expect(result.winProb10d, 0.72);
        expect(result.sector, 'Technology');
      });
    });

    group('fromJson', () {
      test('parses basic fields correctly', () {
        final json = {
          'ticker': 'MSFT',
          'signal': 'bullish',
          'rrr': '2.5:1',
          'entry': '380.00',
          'stop': '370.00',
          'target': '400.00',
          'note': 'Cloud strength',
        };

        final result = ScanResult.fromJson(json);

        expect(result.ticker, 'MSFT');
        expect(result.signal, 'bullish');
        expect(result.rrr, '2.5:1');
        expect(result.entry, '380.00');
        expect(result.stop, '370.00');
        expect(result.target, '400.00');
        expect(result.note, 'Cloud strength');
      });

      test('parses optional numeric fields', () {
        final json = {
          'ticker': 'NVDA',
          'signal': 'bullish',
          'rrr': '3:1',
          'entry': '450.00',
          'stop': '430.00',
          'target': '510.00',
          'note': '',
          'InstitutionalCoreScore': 92.5,
          'ICS_Tier': 'CORE',
          'win_prob_10d': 0.78,
          'QualityScore': 91.0,
          'TechRating': 85.0,
          'AlphaScore': 0.15,
        };

        final result = ScanResult.fromJson(json);

        expect(result.institutionalCoreScore, 92.5);
        expect(result.icsTier, 'CORE');
        expect(result.winProb10d, 0.78);
        expect(result.qualityScore, 91.0);
        expect(result.techRating, 85.0);
        expect(result.alphaScore, 0.15);
      });

      test('parses sparkline array', () {
        final json = {
          'ticker': 'AMD',
          'signal': 'bullish',
          'rrr': '2:1',
          'entry': '100.00',
          'stop': '95.00',
          'target': '110.00',
          'note': '',
          'sparkline': [95.0, 97.0, 99.0, 100.0, 102.0],
        };

        final result = ScanResult.fromJson(json);

        expect(result.sparkline, [95.0, 97.0, 99.0, 100.0, 102.0]);
        expect(result.sparkline.length, 5);
      });

      test('handles alternate key names', () {
        final json = {
          'ticker': 'GOOGL',
          'signal': 'bullish',
          'rr': '2:1', // alternate key for rrr
          'entry': '140.00',
          'stop': '135.00',
          'target': '150.00',
          'note': '',
          'ics': 78.0, // alternate key
          'Tier': 'SATELLITE', // alternate key
          'spark': [138.0, 139.0, 140.0], // alternate key
        };

        final result = ScanResult.fromJson(json);

        expect(result.rrr, '2:1');
        expect(result.institutionalCoreScore, 78.0);
        expect(result.icsTier, 'SATELLITE');
        expect(result.sparkline, [138.0, 139.0, 140.0]);
      });

      test('handles null and missing fields gracefully', () {
        final json = <String, dynamic>{
          'ticker': 'META',
        };

        final result = ScanResult.fromJson(json);

        expect(result.ticker, 'META');
        expect(result.signal, '');
        expect(result.rrr, '');
        expect(result.entry, '');
        expect(result.institutionalCoreScore, isNull);
        expect(result.sparkline, isEmpty);
      });

      test('parses boolean fields correctly', () {
        final json = {
          'ticker': 'COIN',
          'signal': 'bullish',
          'rrr': '4:1',
          'entry': '100.00',
          'stop': '80.00',
          'target': '180.00',
          'note': 'High volatility',
          'IsUltraRisky': true,
        };

        final result = ScanResult.fromJson(json);

        expect(result.isUltraRisky, true);
      });

      test('parses merit score fields', () {
        final json = {
          'ticker': 'AMZN',
          'signal': 'bullish',
          'rrr': '2:1',
          'entry': '180.00',
          'stop': '170.00',
          'target': '200.00',
          'note': '',
          'merit_score': 87.5,
          'merit_band': 'A',
          'merit_flags': 'momentum,volume',
          'merit_summary': 'Strong technical setup',
        };

        final result = ScanResult.fromJson(json);

        expect(result.meritScore, 87.5);
        expect(result.meritBand, 'A');
        expect(result.meritFlags, 'momentum,volume');
        expect(result.meritSummary, 'Strong technical setup');
      });
    });

    group('toJson', () {
      test('serializes all fields correctly', () {
        const result = ScanResult(
          'AAPL',
          'bullish',
          '3:1',
          '150.00',
          '145.00',
          '160.00',
          'Strong setup',
          [148.0, 149.0, 150.0],
          85.0,
          'CORE',
        );

        final json = result.toJson();

        expect(json['ticker'], 'AAPL');
        expect(json['signal'], 'bullish');
        expect(json['rrr'], '3:1');
        expect(json['entry'], '150.00');
        expect(json['stop'], '145.00');
        expect(json['target'], '160.00');
        expect(json['note'], 'Strong setup');
        expect(json['sparkline'], [148.0, 149.0, 150.0]);
        expect(json['InstitutionalCoreScore'], 85.0);
        expect(json['ICS_Tier'], 'CORE');
      });

      test('roundtrip serialization preserves data', () {
        final original = ScanResult.fromJson({
          'ticker': 'NFLX',
          'signal': 'bullish',
          'rrr': '2:1',
          'entry': '600.00',
          'stop': '580.00',
          'target': '640.00',
          'note': 'Streaming leader',
          'InstitutionalCoreScore': 75.0,
          'Sector': 'Communication Services',
        });

        final json = original.toJson();
        final restored = ScanResult.fromJson(json);

        expect(restored.ticker, original.ticker);
        expect(restored.signal, original.signal);
        expect(restored.institutionalCoreScore, original.institutionalCoreScore);
        expect(restored.sector, original.sector);
      });
    });

    group('getTierLabel', () {
      test('returns icsTier when available', () {
        const result = ScanResult(
          'TEST',
          'bullish',
          '2:1',
          '100',
          '95',
          '110',
          '',
          [],
          85.0,
          'CORE',
        );

        expect(result.getTierLabel(), 'CORE');
      });

      test('returns profileLabel when icsTier is null', () {
        const result = ScanResult(
          'TEST',
          'bullish',
          '2:1',
          '100',
          '95',
          '110',
          '',
          [],
          85.0,
          null,
          null,
          null,
          null,
          null,
          null,
          'Growth Leader',
        );

        expect(result.getTierLabel(), 'Growth Leader');
      });

      test('calculates tier from ICS score when no tier set', () {
        const coreResult = ScanResult(
          'TEST',
          'bullish',
          '2:1',
          '100',
          '95',
          '110',
          '',
          [],
          85.0, // >= 80 = CORE
        );
        expect(coreResult.getTierLabel(), 'CORE');

        const satelliteResult = ScanResult(
          'TEST',
          'bullish',
          '2:1',
          '100',
          '95',
          '110',
          '',
          [],
          70.0, // >= 65 = SATELLITE
        );
        expect(satelliteResult.getTierLabel(), 'SATELLITE');

        const watchResult = ScanResult(
          'TEST',
          'bullish',
          '2:1',
          '100',
          '95',
          '110',
          '',
          [],
          50.0, // < 65 = WATCH
        );
        expect(watchResult.getTierLabel(), 'WATCH');
      });
    });

    group('isHighQuality', () {
      test('returns true for ICS >= 80', () {
        const result = ScanResult(
          'AAPL',
          'bullish',
          '2:1',
          '150',
          '145',
          '160',
          '',
          [],
          80.0,
        );

        expect(result.isHighQuality, true);
      });

      test('returns false for ICS < 80', () {
        const result = ScanResult(
          'AAPL',
          'bullish',
          '2:1',
          '150',
          '145',
          '160',
          '',
          [],
          79.9,
        );

        expect(result.isHighQuality, false);
      });

      test('returns false when ICS is null', () {
        const result = ScanResult(
          'AAPL',
          'bullish',
          '2:1',
          '150',
          '145',
          '160',
          '',
        );

        expect(result.isHighQuality, false);
      });
    });

    group('copyWith', () {
      test('creates copy with updated fields', () {
        const original = ScanResult(
          'AAPL',
          'bullish',
          '2:1',
          '150',
          '145',
          '160',
          'Original note',
        );

        final updated = original.copyWith(
          signal: 'bearish',
          note: 'Updated note',
        );

        expect(updated.ticker, 'AAPL');
        expect(updated.signal, 'bearish');
        expect(updated.note, 'Updated note');
        expect(original.signal, 'bullish'); // Original unchanged
      });
    });
  });

  group('OptionStrategy', () {
    group('fromJson', () {
      test('parses all fields correctly', () {
        final json = {
          'id': 'opt_001',
          'label': 'Bull Call Spread',
          'side': 'long',
          'type': 'vertical_spread',
          'defined_risk': true,
          'description': 'Bullish defined-risk strategy',
          'expiry': '2024-03-15',
          'max_profit': 500.0,
          'max_loss': 200.0,
          'prob_itm': 0.65,
        };

        final strategy = OptionStrategy.fromJson(json);

        expect(strategy.id, 'opt_001');
        expect(strategy.label, 'Bull Call Spread');
        expect(strategy.side, 'long');
        expect(strategy.type, 'vertical_spread');
        expect(strategy.definedRisk, true);
        expect(strategy.description, 'Bullish defined-risk strategy');
        expect(strategy.expiry, '2024-03-15');
        expect(strategy.maxProfit, 500.0);
        expect(strategy.maxLoss, 200.0);
        expect(strategy.probabilityITM, 0.65);
      });

      test('handles alternate key names', () {
        final json = {
          'strategy_id': 'opt_002',
          'name': 'Iron Condor',
          'side': 'neutral',
          'strategy_type': 'iron_condor',
          'definedRisk': true,
          'expiration': '2024-04-19',
          'probability_itm': 0.45,
        };

        final strategy = OptionStrategy.fromJson(json);

        expect(strategy.id, 'opt_002');
        expect(strategy.label, 'Iron Condor');
        expect(strategy.type, 'iron_condor');
        expect(strategy.expiry, '2024-04-19');
        expect(strategy.probabilityITM, 0.45);
      });
    });

    group('toJson', () {
      test('serializes correctly', () {
        const strategy = OptionStrategy(
          id: 'opt_003',
          label: 'Put Credit Spread',
          side: 'short',
          type: 'vertical_spread',
          definedRisk: true,
          maxProfit: 150.0,
          maxLoss: 350.0,
        );

        final json = strategy.toJson();

        expect(json['id'], 'opt_003');
        expect(json['label'], 'Put Credit Spread');
        expect(json['defined_risk'], true);
        expect(json['max_profit'], 150.0);
        expect(json['max_loss'], 350.0);
      });
    });

    group('getShortLabel', () {
      test('appends defined risk indicator', () {
        const definedRisk = OptionStrategy(
          id: '1',
          label: 'Vertical Spread',
          side: 'long',
          type: 'spread',
          definedRisk: true,
        );

        expect(definedRisk.getShortLabel(), 'Vertical Spread (defined risk)');
      });

      test('returns label without suffix for undefined risk', () {
        const undefinedRisk = OptionStrategy(
          id: '2',
          label: 'Naked Put',
          side: 'short',
          type: 'naked',
          definedRisk: false,
        );

        expect(undefinedRisk.getShortLabel(), 'Naked Put');
      });
    });
  });
}
