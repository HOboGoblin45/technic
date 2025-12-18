/// UI Test Screen - Showcase all premium components
/// 
/// This screen demonstrates all Phase 1 UI enhancements:
/// - Premium buttons (all variants and sizes)
/// - Premium cards (all variants)
/// - Enhanced colors and gradients
/// - Animations and interactions
library;

import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import '../theme/spacing.dart';
import '../widgets/premium_button.dart';
import '../widgets/premium_card.dart';

class UITestScreen extends StatefulWidget {
  const UITestScreen({super.key});

  @override
  State<UITestScreen> createState() => _UITestScreenState();
}

class _UITestScreenState extends State<UITestScreen> {
  bool _isLoading = false;
  int _counter = 0;

  void _simulateLoading() {
    setState(() => _isLoading = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.darkBackground,
      appBar: AppBar(
        title: const Text('UI Component Test'),
        backgroundColor: AppColors.darkCard,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(Spacing.lg),
        children: [
          _buildSectionHeader('Premium Buttons'),
          _buildButtonShowcase(),
          
          const SizedBox(height: Spacing.xxl),
          _buildSectionHeader('Premium Cards'),
          _buildCardShowcase(),
          
          const SizedBox(height: Spacing.xxl),
          _buildSectionHeader('Stock Result Cards'),
          _buildStockCards(),
          
          const SizedBox(height: Spacing.xxl),
          _buildSectionHeader('Metric Cards'),
          _buildMetricCards(),
          
          const SizedBox(height: Spacing.xxl),
          _buildSectionHeader('Color Palette'),
          _buildColorPalette(),
          
          const SizedBox(height: Spacing.xxl),
          _buildSectionHeader('Gradients'),
          _buildGradientShowcase(),
          
          const SizedBox(height: Spacing.xxl),
        ],
      ),
      floatingActionButton: PremiumIconButton(
        icon: Icons.add,
        size: 56.0,
        tooltip: 'Add Item',
        onPressed: () {
          setState(() => _counter++);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Counter: $_counter'),
              duration: const Duration(seconds: 1),
            ),
          );
        },
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: Spacing.md),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 24,
          fontWeight: FontWeight.bold,
          color: AppColors.darkTextPrimary,
        ),
      ),
    );
  }

  Widget _buildButtonShowcase() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Primary buttons - all sizes
        const Text(
          'Primary Variant',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: AppColors.darkTextSecondary,
          ),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Large Primary',
          icon: Icons.rocket_launch,
          variant: ButtonVariant.primary,
          size: ButtonSize.large,
          isFullWidth: true,
          onPressed: () => _showSnackBar('Large Primary pressed'),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Medium Primary',
          icon: Icons.search,
          variant: ButtonVariant.primary,
          size: ButtonSize.medium,
          isFullWidth: true,
          onPressed: () => _showSnackBar('Medium Primary pressed'),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Small Primary',
          variant: ButtonVariant.primary,
          size: ButtonSize.small,
          isFullWidth: true,
          onPressed: () => _showSnackBar('Small Primary pressed'),
        ),
        
        const SizedBox(height: Spacing.lg),
        
        // Other variants
        const Text(
          'All Variants',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: AppColors.darkTextSecondary,
          ),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Secondary',
          icon: Icons.settings,
          variant: ButtonVariant.secondary,
          isFullWidth: true,
          onPressed: () => _showSnackBar('Secondary pressed'),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Outline',
          icon: Icons.favorite_border,
          variant: ButtonVariant.outline,
          isFullWidth: true,
          onPressed: () => _showSnackBar('Outline pressed'),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Text Button',
          variant: ButtonVariant.text,
          isFullWidth: true,
          onPressed: () => _showSnackBar('Text pressed'),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Success',
          icon: Icons.check_circle,
          variant: ButtonVariant.success,
          isFullWidth: true,
          onPressed: () => _showSnackBar('Success pressed'),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Danger',
          icon: Icons.delete,
          variant: ButtonVariant.danger,
          isFullWidth: true,
          onPressed: () => _showSnackBar('Danger pressed'),
        ),
        
        const SizedBox(height: Spacing.lg),
        
        // States
        const Text(
          'Button States',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: AppColors.darkTextSecondary,
          ),
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Loading State',
          variant: ButtonVariant.primary,
          isFullWidth: true,
          isLoading: _isLoading,
          onPressed: _simulateLoading,
        ),
        const SizedBox(height: Spacing.sm),
        PremiumButton(
          text: 'Disabled State',
          variant: ButtonVariant.primary,
          isFullWidth: true,
          onPressed: null, // Disabled
        ),
      ],
    );
  }

  Widget _buildCardShowcase() {
    return Column(
      children: [
        // Glass card
        PremiumCard(
          variant: CardVariant.glass,
          elevation: CardElevation.medium,
          onTap: () => _showSnackBar('Glass card tapped'),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: AppColors.technicBlue.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(
                      Icons.auto_awesome,
                      color: AppColors.technicBlue,
                    ),
                  ),
                  const SizedBox(width: Spacing.sm),
                  const Expanded(
                    child: Text(
                      'Glass Morphism Card',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: AppColors.darkTextPrimary,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: Spacing.sm),
              const Text(
                'This card features backdrop blur and frosted glass effect. Tap to test press animation.',
                style: TextStyle(
                  fontSize: 14,
                  color: AppColors.darkTextSecondary,
                ),
              ),
            ],
          ),
        ),
        
        // Elevated card
        PremiumCard(
          variant: CardVariant.elevated,
          elevation: CardElevation.high,
          onTap: () => _showSnackBar('Elevated card tapped'),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Elevated Card',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppColors.darkTextPrimary,
                ),
              ),
              SizedBox(height: Spacing.sm),
              Text(
                'Solid background with prominent shadow. High elevation for important content.',
                style: TextStyle(
                  fontSize: 14,
                  color: AppColors.darkTextSecondary,
                ),
              ),
            ],
          ),
        ),
        
        // Gradient card
        PremiumCard(
          variant: CardVariant.gradient,
          gradient: AppColors.primaryGradient,
          elevation: CardElevation.medium,
          onTap: () => _showSnackBar('Gradient card tapped'),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Gradient Card',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              SizedBox(height: Spacing.sm),
              Text(
                'Beautiful gradient background using Technic blue colors.',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.white70,
                ),
              ),
            ],
          ),
        ),
        
        // Outline card
        PremiumCard(
          variant: CardVariant.outline,
          elevation: CardElevation.none,
          onTap: () => _showSnackBar('Outline card tapped'),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Outline Card',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppColors.darkTextPrimary,
                ),
              ),
              SizedBox(height: Spacing.sm),
              Text(
                'Transparent background with subtle border. Minimal and clean.',
                style: TextStyle(
                  fontSize: 14,
                  color: AppColors.darkTextSecondary,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildStockCards() {
    return Column(
      children: [
        StockResultCard(
          symbol: 'AAPL',
          companyName: 'Apple Inc.',
          price: 178.50,
          changePercent: 2.34,
          techRating: 8.5,
          meritScore: 7.8,
          onTap: () => _showSnackBar('AAPL tapped'),
        ),
        StockResultCard(
          symbol: 'TSLA',
          companyName: 'Tesla, Inc.',
          price: 242.84,
          changePercent: -1.23,
          techRating: 7.2,
          meritScore: 8.9,
          onTap: () => _showSnackBar('TSLA tapped'),
        ),
        StockResultCard(
          symbol: 'NVDA',
          companyName: 'NVIDIA Corporation',
          price: 495.22,
          changePercent: 5.67,
          techRating: 9.1,
          meritScore: 9.3,
          onTap: () => _showSnackBar('NVDA tapped'),
        ),
      ],
    );
  }

  Widget _buildMetricCards() {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      mainAxisSpacing: Spacing.sm,
      crossAxisSpacing: Spacing.sm,
      childAspectRatio: 1.2,
      children: [
        MetricCard(
          label: 'Portfolio',
          value: '\$125.4K',
          icon: Icons.account_balance_wallet,
          color: AppColors.successGreen,
          subtitle: '+12.5%',
        ),
        MetricCard(
          label: 'Watchlist',
          value: '24',
          icon: Icons.favorite,
          color: AppColors.dangerRed,
          subtitle: '8 alerts',
        ),
        MetricCard(
          label: 'Scans Today',
          value: '156',
          icon: Icons.search,
          color: AppColors.technicBlue,
          subtitle: '12 new picks',
        ),
        MetricCard(
          label: 'Win Rate',
          value: '68%',
          icon: Icons.trending_up,
          color: AppColors.infoPurple,
          subtitle: 'Last 30 days',
        ),
      ],
    );
  }

  Widget _buildColorPalette() {
    return Wrap(
      spacing: Spacing.sm,
      runSpacing: Spacing.sm,
      children: [
        _buildColorSwatch('Technic Blue', AppColors.technicBlue),
        _buildColorSwatch('Success Green', AppColors.successGreen),
        _buildColorSwatch('Danger Red', AppColors.dangerRed),
        _buildColorSwatch('Warning Orange', AppColors.warningOrange),
        _buildColorSwatch('Info Purple', AppColors.infoPurple),
        _buildColorSwatch('Info Teal', AppColors.infoTeal),
      ],
    );
  }

  Widget _buildColorSwatch(String name, Color color) {
    return Container(
      width: 100,
      padding: const EdgeInsets.all(Spacing.sm),
      decoration: BoxDecoration(
        color: AppColors.darkCard,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Container(
            width: 60,
            height: 60,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(8),
              boxShadow: [
                BoxShadow(
                  color: color.withValues(alpha: 0.4),
                  blurRadius: 12,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
          ),
          const SizedBox(height: Spacing.xs),
          Text(
            name,
            style: const TextStyle(
              fontSize: 10,
              color: AppColors.darkTextSecondary,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildGradientShowcase() {
    return Column(
      children: [
        _buildGradientCard('Primary Gradient', AppColors.primaryGradient),
        const SizedBox(height: Spacing.sm),
        _buildGradientCard('Success Gradient', AppColors.successGradient),
        const SizedBox(height: Spacing.sm),
        _buildGradientCard('Danger Gradient', AppColors.dangerGradient),
        const SizedBox(height: Spacing.sm),
        _buildGradientCard('Card Gradient', AppColors.cardGradient),
        const SizedBox(height: Spacing.sm),
        _buildGradientCard('Premium Gradient', AppColors.premiumGradient),
      ],
    );
  }

  Widget _buildGradientCard(String name, Gradient gradient) {
    return Container(
      height: 80,
      decoration: BoxDecoration(
        gradient: gradient,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Center(
        child: Text(
          name,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
      ),
    );
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 1),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}
