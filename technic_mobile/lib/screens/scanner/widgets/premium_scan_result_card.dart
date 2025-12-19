/// Premium Scan Result Card Widget
/// 
/// Enhanced version with billion-dollar app quality:
/// - Glass morphism design
/// - Smooth animations
/// - Better visual hierarchy
/// - Premium typography
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../models/scan_result.dart';
import '../../../providers/app_providers.dart';
import '../../../theme/app_colors.dart';
import '../../../theme/spacing.dart';
import '../../../widgets/premium_card.dart';
import '../../../widgets/premium_button.dart';
import '../../../widgets/sparkline.dart';

class PremiumScanResultCard extends ConsumerWidget {
  final ScanResult result;
  final VoidCallback? onTap;
  final bool isTopPick;
  
  const PremiumScanResultCard({
    super.key,
    required this.result,
    this.onTap,
    this.isTopPick = false,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (isTopPick) {
      return _buildTopPickCard(context, ref);
    }
    return _buildStandardCard(context, ref);
  }
  
  /// Top pick card with gradient and prominent display
  Widget _buildTopPickCard(BuildContext context, WidgetRef ref) {
    final hasSparkline = result.sparkline.isNotEmpty;
    final isPositive = hasSparkline && result.sparkline.last > result.sparkline.first;
    
    return PremiumCard(
      variant: CardVariant.gradient,
      gradient: AppColors.primaryGradient,
      elevation: CardElevation.high,
      onTap: onTap,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Top Pick Badge
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.star,
                      size: 16,
                      color: Colors.white,
                    ),
                    SizedBox(width: 4),
                    Text(
                      'TOP PICK',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                        letterSpacing: 1.2,
                      ),
                    ),
                  ],
                ),
              ),
              const Spacer(),
              if (result.icsTier != null && result.icsTier!.isNotEmpty)
                _buildTierBadge(result.icsTier!, isLight: true),
            ],
          ),
          
          const SizedBox(height: Spacing.lg),
          
          // Ticker and Signal
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      result.ticker,
                      style: const TextStyle(
                        fontSize: 36,
                        fontWeight: FontWeight.w900,
                        color: Colors.white,
                        height: 1.0,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      result.signal,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Colors.white70,
                      ),
                    ),
                  ],
                ),
              ),
              
              // Sparkline
              if (hasSparkline)
                Container(
                  width: 100,
                  height: 50,
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Sparkline(
                    data: result.sparkline,
                    positive: isPositive,
                    color: Colors.white,
                  ),
                ),
            ],
          ),
          
          const SizedBox(height: Spacing.lg),
          
          // MERIT Score (if available)
          if (result.meritScore != null) ...[
            _buildMeritScoreDisplay(isLight: true),
            const SizedBox(height: Spacing.md),
          ],
          
          // Key Metrics
          _buildMetricsRow(isLight: true),
          
          const SizedBox(height: Spacing.lg),
          
          // Actions
          _buildActionButtons(context, ref, isLight: true),
        ],
      ),
    );
  }
  
  /// Standard card with glass morphism
  Widget _buildStandardCard(BuildContext context, WidgetRef ref) {
    final hasSparkline = result.sparkline.isNotEmpty;
    final isPositive = hasSparkline && result.sparkline.last > result.sparkline.first;
    
    return PremiumCard(
      variant: CardVariant.glass,
      elevation: CardElevation.medium,
      onTap: onTap,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Row
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Ticker and Signal
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          result.ticker,
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.w900,
                            color: AppColors.darkTextPrimary,
                          ),
                        ),
                        if (result.icsTier != null && result.icsTier!.isNotEmpty) ...[
                          const SizedBox(width: 8),
                          _buildTierBadge(result.icsTier!),
                        ],
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      result.signal,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: AppColors.darkTextSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              
              // Sparkline
              if (hasSparkline)
                Container(
                  width: 80,
                  height: 40,
                  padding: const EdgeInsets.all(4),
                  decoration: BoxDecoration(
                    color: AppColors.darkCard.withValues(alpha: 0.5),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Sparkline(
                    data: result.sparkline,
                    positive: isPositive,
                    color: isPositive ? AppColors.successGreen : AppColors.dangerRed,
                  ),
                ),
            ],
          ),
          
          const SizedBox(height: Spacing.md),
          
          // MERIT Score (if available)
          if (result.meritScore != null) ...[
            _buildMeritScoreDisplay(),
            const SizedBox(height: Spacing.md),
          ],
          
          // Key Metrics
          _buildMetricsRow(),
          
          // MERIT Flags (if any)
          if (result.meritFlags != null && result.meritFlags!.isNotEmpty) ...[
            const SizedBox(height: Spacing.sm),
            _buildMeritFlags(),
          ],
          
          const SizedBox(height: Spacing.md),
          
          // Trade Plan
          _buildTradePlan(),
          
          const SizedBox(height: Spacing.md),
          
          // Actions
          _buildActionButtons(context, ref),
        ],
      ),
    );
  }
  
  Widget _buildTierBadge(String tier, {bool isLight = false}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: isLight 
          ? Colors.white.withValues(alpha: 0.2)
          : _getTierColor(tier).withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: isLight ? Colors.white : _getTierColor(tier),
          width: 1.5,
        ),
      ),
      child: Text(
        tier,
        style: TextStyle(
          fontSize: 10,
          fontWeight: FontWeight.w800,
          color: isLight ? Colors.white : _getTierColor(tier),
          letterSpacing: 0.5,
        ),
      ),
    );
  }
  
  Widget _buildMeritScoreDisplay({bool isLight = false}) {
    return Container(
      padding: const EdgeInsets.all(Spacing.md),
      decoration: BoxDecoration(
        gradient: isLight
          ? LinearGradient(
              colors: [
                Colors.white.withValues(alpha: 0.2),
                Colors.white.withValues(alpha: 0.1),
              ],
            )
          : LinearGradient(
              colors: [
                AppColors.technicBlue.withValues(alpha: 0.2),
                AppColors.technicBlue.withValues(alpha: 0.1),
              ],
            ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isLight 
            ? Colors.white.withValues(alpha: 0.3)
            : AppColors.technicBlue.withValues(alpha: 0.4),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'MERIT SCORE',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w800,
                    color: isLight ? Colors.white70 : AppColors.darkTextSecondary,
                    letterSpacing: 1.2,
                  ),
                ),
                const SizedBox(height: 4),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      result.meritScore!.toStringAsFixed(0),
                      style: TextStyle(
                        fontSize: 40,
                        fontWeight: FontWeight.w900,
                        color: isLight ? Colors.white : AppColors.darkTextPrimary,
                        height: 1.0,
                      ),
                    ),
                    if (result.meritBand != null && result.meritBand!.isNotEmpty) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: _getMeritBandColor(result.meritBand!),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          result.meritBand!,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w900,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ),
          Icon(
            Icons.verified,
            size: 48,
            color: isLight 
              ? Colors.white.withValues(alpha: 0.5)
              : AppColors.technicBlue.withValues(alpha: 0.6),
          ),
        ],
      ),
    );
  }
  
  Widget _buildMetricsRow({bool isLight = false}) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        if (result.meritScore == null && result.rrr.isNotEmpty)
          _metricPill('RRR', result.rrr, Icons.trending_up, isLight: isLight),
        if (result.techRating != null)
          _metricPill(
            'Tech',
            result.techRating!.toStringAsFixed(1),
            Icons.analytics,
            isLight: isLight,
          ),
        if (result.winProb10d != null)
          _metricPill(
            'Win%',
            '${(result.winProb10d! * 100).toStringAsFixed(0)}%',
            Icons.percent,
            isLight: isLight,
          ),
        if (result.qualityScore != null)
          _metricPill(
            'Quality',
            result.qualityScore!.toStringAsFixed(0),
            Icons.star,
            isLight: isLight,
          ),
        if (result.institutionalCoreScore != null)
          _metricPill(
            'ICS',
            result.institutionalCoreScore!.toStringAsFixed(0),
            Icons.business,
            isLight: isLight,
          ),
      ],
    );
  }
  
  Widget _metricPill(String label, String value, IconData icon, {bool isLight = false}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: isLight
          ? Colors.white.withValues(alpha: 0.15)
          : AppColors.technicBlue.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: isLight
            ? Colors.white.withValues(alpha: 0.3)
            : AppColors.technicBlue.withValues(alpha: 0.3),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 16,
            color: isLight ? Colors.white : AppColors.technicBlue,
          ),
          const SizedBox(width: 6),
          Text(
            '$label: ',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: isLight ? Colors.white70 : AppColors.darkTextSecondary,
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w800,
              color: isLight ? Colors.white : AppColors.darkTextPrimary,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildMeritFlags() {
    final flagList = result.meritFlags!
        .split(',')
        .map((f) => f.trim())
        .where((f) => f.isNotEmpty)
        .toList();
    
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      children: flagList.map((flag) {
        final flagInfo = _getFlagInfo(flag);
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: flagInfo.color.withValues(alpha: 0.2),
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: flagInfo.color),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(flagInfo.icon, size: 12, color: flagInfo.color),
              const SizedBox(width: 4),
              Text(
                flag,
                style: TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  color: flagInfo.color,
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
  
  Widget _buildTradePlan() {
    return Container(
      padding: const EdgeInsets.all(Spacing.md),
      decoration: BoxDecoration(
        color: AppColors.darkCard.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: AppColors.darkCard.withValues(alpha: 0.5),
        ),
      ),
      child: Column(
        children: [
          _tradePlanRow('Entry', result.entry, AppColors.successGreen),
          const SizedBox(height: 8),
          _tradePlanRow('Stop', result.stop, AppColors.dangerRed),
          const SizedBox(height: 8),
          _tradePlanRow('Target', result.target, AppColors.technicBlue),
        ],
      ),
    );
  }
  
  Widget _tradePlanRow(String label, String value, Color color) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: AppColors.darkTextSecondary,
              ),
            ),
          ],
        ),
        Text(
          value.isNotEmpty ? value : '-',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w800,
            color: value.isNotEmpty ? color : AppColors.darkTextTertiary,
          ),
        ),
      ],
    );
  }
  
  Widget _buildActionButtons(BuildContext context, WidgetRef ref, {bool isLight = false}) {
    final watchlist = ref.watch(watchlistProvider);
    final isWatched = watchlist.any((item) => item.ticker == result.ticker);
    
    return Row(
      children: [
        Expanded(
          child: PremiumButton(
            text: 'Ask Copilot',
            icon: Icons.chat_bubble_outline,
            variant: isLight ? ButtonVariant.secondary : ButtonVariant.outline,
            size: ButtonSize.small,
            onPressed: () {
              ref.read(copilotContextProvider.notifier).state = result;
              ref.read(copilotPrefillProvider.notifier).state =
                  'Analyze ${result.ticker} ${result.signal} setup';
              ref.read(currentTabProvider.notifier).setTab(2);
            },
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: PremiumButton(
            text: isWatched ? 'Saved' : 'Save',
            icon: isWatched ? Icons.bookmark : Icons.bookmark_outline,
            variant: isWatched 
              ? ButtonVariant.success 
              : (isLight ? ButtonVariant.secondary : ButtonVariant.outline),
            size: ButtonSize.small,
            onPressed: () async {
              if (isWatched) {
                await ref.read(watchlistProvider.notifier).remove(result.ticker);
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('${result.ticker} removed from watchlist'),
                      duration: const Duration(seconds: 2),
                    ),
                  );
                }
              } else {
                await ref.read(watchlistProvider.notifier).add(
                  result.ticker,
                  signal: result.signal,
                );
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('${result.ticker} added to watchlist'),
                      duration: const Duration(seconds: 2),
                      backgroundColor: AppColors.successGreen,
                    ),
                  );
                }
              }
            },
          ),
        ),
      ],
    );
  }
  
  Color _getTierColor(String tier) {
    switch (tier.toUpperCase()) {
      case 'CORE':
        return AppColors.successGreen;
      case 'SATELLITE':
        return AppColors.technicBlue;
      case 'WATCH':
        return AppColors.warningOrange;
      default:
        return AppColors.darkTextSecondary;
    }
  }
  
  Color _getMeritBandColor(String band) {
    switch (band.toUpperCase()) {
      case 'A+':
      case 'A':
        return AppColors.successGreen;
      case 'B':
        return AppColors.technicBlue;
      case 'C':
        return AppColors.warningOrange;
      case 'D':
        return AppColors.dangerRed;
      default:
        return AppColors.darkTextSecondary;
    }
  }
  
  ({Color color, IconData icon}) _getFlagInfo(String flag) {
    if (flag.contains('EARNINGS')) {
      return (color: AppColors.dangerRed, icon: Icons.event);
    } else if (flag.contains('LIQUIDITY')) {
      return (color: AppColors.warningOrange, icon: Icons.water_drop);
    } else if (flag.contains('VOLATILITY') || flag.contains('ATR')) {
      return (color: AppColors.warningOrange, icon: Icons.show_chart);
    } else if (flag.contains('MICRO') || flag.contains('SMALL')) {
      return (color: AppColors.infoPurple, icon: Icons.business_center);
    } else {
      return (color: AppColors.darkTextSecondary, icon: Icons.flag);
    }
  }
}
