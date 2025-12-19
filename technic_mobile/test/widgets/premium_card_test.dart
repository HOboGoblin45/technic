import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:technic_mobile/widgets/premium_card.dart';

void main() {
  // Helper to wrap widgets with MaterialApp for proper theming
  Widget wrapWithMaterialApp(Widget child) {
    return MaterialApp(
      home: Scaffold(body: Center(child: child)),
    );
  }

  group('PremiumCard', () {
    group('rendering', () {
      testWidgets('displays child widget', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            child: Text('Card Content'),
          ),
        ));

        expect(find.text('Card Content'), findsOneWidget);
      });

      testWidgets('applies custom padding', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            padding: EdgeInsets.all(32),
            child: Text('Padded'),
          ),
        ));

        expect(find.text('Padded'), findsOneWidget);
      });

      testWidgets('applies custom margin', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            margin: EdgeInsets.symmetric(horizontal: 16),
            child: Text('Margined'),
          ),
        ));

        expect(find.text('Margined'), findsOneWidget);
      });

      testWidgets('respects custom dimensions', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            width: 200,
            height: 100,
            child: Text('Sized'),
          ),
        ));

        expect(find.text('Sized'), findsOneWidget);
      });
    });

    group('variants', () {
      testWidgets('renders glass variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            variant: CardVariant.glass,
            child: Text('Glass'),
          ),
        ));

        expect(find.text('Glass'), findsOneWidget);
        // Glass variant uses BackdropFilter
        expect(find.byType(BackdropFilter), findsOneWidget);
      });

      testWidgets('renders elevated variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            variant: CardVariant.elevated,
            child: Text('Elevated'),
          ),
        ));

        expect(find.text('Elevated'), findsOneWidget);
      });

      testWidgets('renders gradient variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            variant: CardVariant.gradient,
            child: Text('Gradient'),
          ),
        ));

        expect(find.text('Gradient'), findsOneWidget);
      });

      testWidgets('renders outline variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            variant: CardVariant.outline,
            child: Text('Outline'),
          ),
        ));

        expect(find.text('Outline'), findsOneWidget);
      });
    });

    group('elevation', () {
      testWidgets('renders with no elevation', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            elevation: CardElevation.none,
            child: Text('Flat'),
          ),
        ));

        expect(find.text('Flat'), findsOneWidget);
      });

      testWidgets('renders with low elevation', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            elevation: CardElevation.low,
            child: Text('Low'),
          ),
        ));

        expect(find.text('Low'), findsOneWidget);
      });

      testWidgets('renders with medium elevation', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            elevation: CardElevation.medium,
            child: Text('Medium'),
          ),
        ));

        expect(find.text('Medium'), findsOneWidget);
      });

      testWidgets('renders with high elevation', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            elevation: CardElevation.high,
            child: Text('High'),
          ),
        ));

        expect(find.text('High'), findsOneWidget);
      });
    });

    group('interaction', () {
      testWidgets('calls onTap when tapped', (tester) async {
        bool tapped = false;

        await tester.pumpWidget(wrapWithMaterialApp(
          PremiumCard(
            onTap: () => tapped = true,
            child: const Text('Tappable'),
          ),
        ));

        await tester.tap(find.text('Tappable'));
        await tester.pump();

        expect(tapped, isTrue);
      });

      testWidgets('wraps with GestureDetector when onTap is provided', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          PremiumCard(
            onTap: () {},
            child: const Text('Interactive'),
          ),
        ));

        expect(find.byType(GestureDetector), findsOneWidget);
      });

      testWidgets('does not wrap with GestureDetector when onTap is null', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumCard(
            child: Text('Static'),
          ),
        ));

        // Only one Container, no GestureDetector at card level
        expect(find.text('Static'), findsOneWidget);
      });
    });

    group('animation', () {
      testWidgets('uses ScaleTransition when enablePressEffect is true', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          PremiumCard(
            onTap: () {},
            enablePressEffect: true,
            child: const Text('Animated'),
          ),
        ));

        expect(find.byType(ScaleTransition), findsOneWidget);
      });
    });
  });

  group('StockResultCard', () {
    testWidgets('displays stock symbol', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const StockResultCard(
          symbol: 'AAPL',
          companyName: 'Apple Inc.',
          price: 150.00,
          changePercent: 2.5,
          techRating: 85.0,
          meritScore: 90.0,
        ),
      ));

      expect(find.text('AAPL'), findsOneWidget);
    });

    testWidgets('displays company name', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const StockResultCard(
          symbol: 'MSFT',
          companyName: 'Microsoft Corporation',
          price: 380.00,
          changePercent: -1.2,
          techRating: 82.0,
          meritScore: 88.0,
        ),
      ));

      expect(find.text('Microsoft Corporation'), findsOneWidget);
    });

    testWidgets('displays formatted price', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const StockResultCard(
          symbol: 'GOOGL',
          companyName: 'Alphabet Inc.',
          price: 140.50,
          changePercent: 0.75,
          techRating: 78.0,
          meritScore: 85.0,
        ),
      ));

      expect(find.text('\$140.50'), findsOneWidget);
    });

    testWidgets('displays positive change with plus sign', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const StockResultCard(
          symbol: 'NVDA',
          companyName: 'NVIDIA Corporation',
          price: 450.00,
          changePercent: 3.25,
          techRating: 92.0,
          meritScore: 95.0,
        ),
      ));

      expect(find.text('+3.25%'), findsOneWidget);
    });

    testWidgets('displays negative change without plus sign', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const StockResultCard(
          symbol: 'META',
          companyName: 'Meta Platforms',
          price: 350.00,
          changePercent: -2.15,
          techRating: 75.0,
          meritScore: 80.0,
        ),
      ));

      expect(find.text('-2.15%'), findsOneWidget);
    });

    testWidgets('displays tech rating', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const StockResultCard(
          symbol: 'AMD',
          companyName: 'AMD',
          price: 100.00,
          changePercent: 1.5,
          techRating: 87.5,
          meritScore: 82.0,
        ),
      ));

      expect(find.text('Tech'), findsOneWidget);
      expect(find.text('87.5'), findsOneWidget);
    });

    testWidgets('displays merit score', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const StockResultCard(
          symbol: 'TSLA',
          companyName: 'Tesla Inc.',
          price: 200.00,
          changePercent: 4.2,
          techRating: 70.0,
          meritScore: 91.5,
        ),
      ));

      expect(find.text('Merit'), findsOneWidget);
      expect(find.text('91.5'), findsOneWidget);
    });

    testWidgets('calls onTap when tapped', (tester) async {
      bool tapped = false;

      await tester.pumpWidget(wrapWithMaterialApp(
        StockResultCard(
          symbol: 'AAPL',
          companyName: 'Apple Inc.',
          price: 150.00,
          changePercent: 2.5,
          techRating: 85.0,
          meritScore: 90.0,
          onTap: () => tapped = true,
        ),
      ));

      await tester.tap(find.text('AAPL'));
      await tester.pump();

      expect(tapped, isTrue);
    });
  });

  group('MetricCard', () {
    testWidgets('displays label', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const MetricCard(
          label: 'Total Scans',
          value: '1,234',
          icon: Icons.search,
        ),
      ));

      expect(find.text('Total Scans'), findsOneWidget);
    });

    testWidgets('displays value', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const MetricCard(
          label: 'Alerts',
          value: '42',
          icon: Icons.notifications,
        ),
      ));

      expect(find.text('42'), findsOneWidget);
    });

    testWidgets('displays icon', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const MetricCard(
          label: 'Watchlist',
          value: '15',
          icon: Icons.star,
        ),
      ));

      expect(find.byIcon(Icons.star), findsOneWidget);
    });

    testWidgets('displays subtitle when provided', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const MetricCard(
          label: 'Win Rate',
          value: '72%',
          icon: Icons.trending_up,
          subtitle: 'Last 30 days',
        ),
      ));

      expect(find.text('Last 30 days'), findsOneWidget);
    });

    testWidgets('does not display subtitle when null', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const MetricCard(
          label: 'Trades',
          value: '58',
          icon: Icons.swap_horiz,
        ),
      ));

      expect(find.text('Trades'), findsOneWidget);
      expect(find.text('58'), findsOneWidget);
    });
  });
}
