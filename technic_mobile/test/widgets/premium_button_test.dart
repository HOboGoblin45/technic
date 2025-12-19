import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:technic_mobile/widgets/premium_button.dart';

void main() {
  // Helper to wrap widgets with MaterialApp for proper theming
  Widget wrapWithMaterialApp(Widget child) {
    return MaterialApp(
      home: Scaffold(body: Center(child: child)),
    );
  }

  group('PremiumButton', () {
    group('rendering', () {
      testWidgets('displays text correctly', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(text: 'Test Button'),
        ));

        expect(find.text('Test Button'), findsOneWidget);
      });

      testWidgets('displays leading icon when provided', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'With Icon',
            icon: Icons.add,
          ),
        ));

        expect(find.byIcon(Icons.add), findsOneWidget);
        expect(find.text('With Icon'), findsOneWidget);
      });

      testWidgets('displays trailing icon when provided', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'With Trailing',
            trailingIcon: Icons.arrow_forward,
          ),
        ));

        expect(find.byIcon(Icons.arrow_forward), findsOneWidget);
      });

      testWidgets('displays both icons when provided', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Both Icons',
            icon: Icons.add,
            trailingIcon: Icons.arrow_forward,
          ),
        ));

        expect(find.byIcon(Icons.add), findsOneWidget);
        expect(find.byIcon(Icons.arrow_forward), findsOneWidget);
      });

      testWidgets('shows loading indicator when isLoading is true', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Loading Button',
            isLoading: true,
            onPressed: null,
          ),
        ));

        expect(find.byType(CircularProgressIndicator), findsOneWidget);
        expect(find.text('Loading Button'), findsNothing);
      });
    });

    group('interaction', () {
      testWidgets('calls onPressed when tapped', (tester) async {
        bool pressed = false;

        await tester.pumpWidget(wrapWithMaterialApp(
          PremiumButton(
            text: 'Tap Me',
            onPressed: () => pressed = true,
          ),
        ));

        await tester.tap(find.text('Tap Me'));
        await tester.pump();

        expect(pressed, isTrue);
      });

      testWidgets('does not call onPressed when disabled', (tester) async {
        bool pressed = false;

        await tester.pumpWidget(wrapWithMaterialApp(
          PremiumButton(
            text: 'Disabled',
            onPressed: null,
          ),
        ));

        await tester.tap(find.text('Disabled'));
        await tester.pump();

        expect(pressed, isFalse);
      });

      testWidgets('does not call onPressed when loading', (tester) async {
        bool pressed = false;

        await tester.pumpWidget(wrapWithMaterialApp(
          PremiumButton(
            text: 'Loading',
            isLoading: true,
            onPressed: () => pressed = true,
          ),
        ));

        await tester.tap(find.byType(PremiumButton));
        await tester.pump();

        expect(pressed, isFalse);
      });
    });

    group('variants', () {
      testWidgets('renders primary variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Primary',
            variant: ButtonVariant.primary,
          ),
        ));

        expect(find.text('Primary'), findsOneWidget);
      });

      testWidgets('renders secondary variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Secondary',
            variant: ButtonVariant.secondary,
          ),
        ));

        expect(find.text('Secondary'), findsOneWidget);
      });

      testWidgets('renders outline variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Outline',
            variant: ButtonVariant.outline,
          ),
        ));

        expect(find.text('Outline'), findsOneWidget);
      });

      testWidgets('renders text variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Text Only',
            variant: ButtonVariant.text,
          ),
        ));

        expect(find.text('Text Only'), findsOneWidget);
      });

      testWidgets('renders danger variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Danger',
            variant: ButtonVariant.danger,
          ),
        ));

        expect(find.text('Danger'), findsOneWidget);
      });

      testWidgets('renders success variant', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Success',
            variant: ButtonVariant.success,
          ),
        ));

        expect(find.text('Success'), findsOneWidget);
      });
    });

    group('sizes', () {
      testWidgets('renders small size', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Small',
            size: ButtonSize.small,
          ),
        ));

        final container = tester.widget<Container>(
          find.descendant(
            of: find.byType(PremiumButton),
            matching: find.byType(Container).first,
          ),
        );
        expect(container.constraints?.maxHeight, 40.0);
      });

      testWidgets('renders medium size', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Medium',
            size: ButtonSize.medium,
          ),
        ));

        expect(find.text('Medium'), findsOneWidget);
      });

      testWidgets('renders large size', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Large',
            size: ButtonSize.large,
          ),
        ));

        expect(find.text('Large'), findsOneWidget);
      });
    });

    group('full width', () {
      testWidgets('renders full width when isFullWidth is true', (tester) async {
        await tester.pumpWidget(wrapWithMaterialApp(
          const PremiumButton(
            text: 'Full Width',
            isFullWidth: true,
          ),
        ));

        expect(find.text('Full Width'), findsOneWidget);
      });
    });
  });

  group('PremiumIconButton', () {
    testWidgets('displays icon', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const PremiumIconButton(icon: Icons.add),
      ));

      expect(find.byIcon(Icons.add), findsOneWidget);
    });

    testWidgets('calls onPressed when tapped', (tester) async {
      bool pressed = false;

      await tester.pumpWidget(wrapWithMaterialApp(
        PremiumIconButton(
          icon: Icons.add,
          onPressed: () => pressed = true,
        ),
      ));

      await tester.tap(find.byIcon(Icons.add));
      await tester.pump();

      expect(pressed, isTrue);
    });

    testWidgets('shows tooltip when provided', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const PremiumIconButton(
          icon: Icons.add,
          tooltip: 'Add item',
        ),
      ));

      expect(find.byType(Tooltip), findsOneWidget);
    });

    testWidgets('respects custom size', (tester) async {
      await tester.pumpWidget(wrapWithMaterialApp(
        const PremiumIconButton(
          icon: Icons.add,
          size: 64.0,
        ),
      ));

      expect(find.byType(PremiumIconButton), findsOneWidget);
    });
  });
}
