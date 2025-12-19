/// Premium Price Header - Hero section for stock detail page
/// 
/// Features:
/// - Large, prominent price display
/// - Gradient background
/// - Real-time change indicator with animation
/// - Glass morphism container
/// - Watchlist star with animation
/// - Market status indicator
/// - Volume and market cap badges
library;

import 'package:flutter/material.dart';
import 'dart:ui';

import '../../../theme/app_colors.dart';
import '../../../utils/formatters.dart';

class PremiumPriceHeader extends StatefulWidget {
  final String symbol;
  final String? companyName;
  final double? currentPrice;
  final double? changePct;
  final double? changeAmount;
  final String? icsTier;
  final bool isWatched;
  final VoidCallback onWatchlistToggle;
  final bool isMarketOpen;
  final double? volume;
  final double? marketCap;

  const PremiumPriceHeader({
    super.key,
    required this.symbol,
    this.companyName,
    this.currentPrice,
    this.changePct,
    this.changeAmount,
    this.icsTier,
    required this.isWatched,
    required this.onWatchlistToggle,
    this.isMarketOpen = false,
    this.volume,
    this.marketCap,
  });

  @override
  State<PremiumPriceHeader> createState() => _PremiumPriceHeaderState();
}

class _PremiumPriceHeaderState extends State<PremiumPriceHeader>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOut,
    ));

    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, -0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    ));

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isPositive = (widget.changePct ?? 0) >= 0;
    final changeColor = isPositive ? AppColors.successGreen : AppColors.dangerRed;

    return FadeTransition(
      opacity: _fadeAnimation,
      child: SlideTransition(
        position: _slideAnimation,
        child: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                AppColors.primaryBlue.withValues(alpha: 0.2),
                AppColors.primaryBlue.withValues(alpha: 0.05),
              ],
            ),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
              width: 1,
            ),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.05),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Top Row: Symbol, Tier, Watchlist
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              // Symbol
                              Text(
                                widget.symbol,
                                style: const TextStyle(
                                  fontSize: 32,
                                  fontWeight: FontWeight.w800,
                                  letterSpacing: -0.5,
                                  color: Colors.white,
                                  shadows: [
                                    Shadow(
                                      color: Colors.black26,
                                      offset: Offset(0, 2),
                                      blurRadius: 4,
                                    ),
                                  ],
                                ),
                              ),
                              // Company Name
                              if (widget.companyName != null) ...[
                                const SizedBox(height: 4),
                                Text(
                                  widget.companyName!,
                                  style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600,
                                    color: Colors.white.withValues(alpha: 0.7),
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ],
                            ],
                          ),
                        ),
                        // Watchlist Button
                        _buildWatchlistButton(),
                      ],
                    ),

                    const SizedBox(height: 20),

                    // Price Row
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        // Current Price
                        if (widget.currentPrice != null)
                          Text(
                            formatCurrency(widget.currentPrice!),
                            style: const TextStyle(
                              fontSize: 48,
                              fontWeight: FontWeight.w800,
                              letterSpacing: -1,
                              color: Colors.white,
                              height: 1,
                              shadows: [
                                Shadow(
                                  color: Colors.black26,
                                  offset: Offset(0, 2),
                                  blurRadius: 8,
                                ),
                              ],
                            ),
                          ),
                        const SizedBox(width: 16),
                        // Change
                        if (widget.changePct != null)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color: changeColor.withValues(alpha: 0.2),
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(
                                  color: changeColor.withValues(alpha: 0.3),
                                  width: 1,
                                ),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(
                                    isPositive
                                        ? Icons.arrow_upward
                                        : Icons.arrow_downward,
                                    size: 16,
                                    color: changeColor,
                                  ),
                                  const SizedBox(width: 4),
                                  Text(
                                    '${isPositive ? '+' : ''}${widget.changePct!.toStringAsFixed(2)}%',
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w700,
                                      color: changeColor,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                      ],
                    ),

                    // Change Amount
                    if (widget.changeAmount != null) ...[
                      const SizedBox(height: 8),
                      Text(
                        '${isPositive ? '+' : ''}${formatCurrency(widget.changeAmount!)} today',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: changeColor.withValues(alpha: 0.9),
                        ),
                      ),
                    ],

                    const SizedBox(height: 20),

                    // Bottom Row: Badges
                    Wrap(
                      spacing: 12,
                      runSpacing: 12,
                      children: [
                        // Market Status
                        _buildBadge(
                          icon: Icons.circle,
                          label: widget.isMarketOpen ? 'Market Open' : 'Market Closed',
                          color: widget.isMarketOpen
                              ? AppColors.successGreen
                              : Colors.orange,
                        ),
                        // ICS Tier
                        if (widget.icsTier != null)
                          _buildBadge(
                            icon: Icons.star,
                            label: widget.icsTier!,
                            color: _getTierColor(widget.icsTier!),
                          ),
                        // Volume
                        if (widget.volume != null)
                          _buildBadge(
                            icon: Icons.show_chart,
                            label: 'Vol ${formatCompact(widget.volume!)}',
                            color: AppColors.primaryBlue,
                          ),
                        // Market Cap
                        if (widget.marketCap != null)
                          _buildBadge(
                            icon: Icons.account_balance,
                            label: 'Cap ${formatCompact(widget.marketCap!)}',
                            color: AppColors.primaryBlue,
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildWatchlistButton() {
    return GestureDetector(
      onTap: widget.onWatchlistToggle,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: widget.isWatched
              ? AppColors.successGreen.withValues(alpha: 0.2)
              : Colors.white.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: widget.isWatched
                ? AppColors.successGreen.withValues(alpha: 0.3)
                : Colors.white.withValues(alpha: 0.2),
            width: 1,
          ),
        ),
        child: Icon(
          widget.isWatched ? Icons.bookmark : Icons.bookmark_outline,
          color: widget.isWatched ? AppColors.successGreen : Colors.white,
          size: 24,
        ),
      ),
    );
  }

  Widget _buildBadge({
    required IconData icon,
    required String label,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 12,
        vertical: 6,
      ),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: color.withValues(alpha: 0.3),
          width: 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 14,
            color: color,
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Color _getTierColor(String tier) {
    switch (tier.toUpperCase()) {
      case 'CORE':
        return AppColors.successGreen;
      case 'SATELLITE':
        return AppColors.primaryBlue;
      case 'WATCH':
        return Colors.orange;
      default:
        return Colors.grey;
    }
  }
}
