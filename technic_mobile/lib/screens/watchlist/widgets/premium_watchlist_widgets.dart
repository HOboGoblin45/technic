/// Premium Watchlist Widgets
///
/// Professional watchlist components with:
/// - Glass morphism design
/// - Smooth animations
/// - Interactive gestures
/// - Premium visual effects
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui';
import 'dart:math' as math;

import '../../../theme/app_colors.dart';
import '../../../models/watchlist_item.dart';

// =============================================================================
// PREMIUM WATCHLIST CARD
// =============================================================================

/// Premium watchlist card with glass morphism and animations
class PremiumWatchlistCard extends StatefulWidget {
  final WatchlistItem item;
  final VoidCallback? onTap;
  final VoidCallback? onDelete;
  final VoidCallback? onEditNote;
  final VoidCallback? onEditTags;
  final VoidCallback? onSetAlert;
  final int activeAlerts;
  final double? currentPrice;
  final double? changePercent;

  const PremiumWatchlistCard({
    super.key,
    required this.item,
    this.onTap,
    this.onDelete,
    this.onEditNote,
    this.onEditTags,
    this.onSetAlert,
    this.activeAlerts = 0,
    this.currentPrice,
    this.changePercent,
  });

  @override
  State<PremiumWatchlistCard> createState() => _PremiumWatchlistCardState();
}

class _PremiumWatchlistCardState extends State<PremiumWatchlistCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  bool _isPressed = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 150),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 0.98).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onTapDown(TapDownDetails details) {
    setState(() => _isPressed = true);
    _controller.forward();
  }

  void _onTapUp(TapUpDetails details) {
    setState(() => _isPressed = false);
    _controller.reverse();
  }

  void _onTapCancel() {
    setState(() => _isPressed = false);
    _controller.reverse();
  }

  @override
  Widget build(BuildContext context) {
    final isPositive = (widget.changePercent ?? 0) >= 0;
    final changeColor = isPositive ? AppColors.successGreen : AppColors.dangerRed;

    return AnimatedBuilder(
      animation: _scaleAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: _scaleAnimation.value,
          child: GestureDetector(
            onTapDown: _onTapDown,
            onTapUp: _onTapUp,
            onTapCancel: _onTapCancel,
            onTap: () {
              HapticFeedback.lightImpact();
              widget.onTap?.call();
            },
            child: Container(
              margin: const EdgeInsets.only(bottom: 12),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Colors.white.withValues(alpha: _isPressed ? 0.08 : 0.06),
                    Colors.white.withValues(alpha: _isPressed ? 0.04 : 0.02),
                  ],
                ),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: Colors.white.withValues(alpha: _isPressed ? 0.15 : 0.08),
                  width: 1,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.2),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Header Row
                        Row(
                          children: [
                            // Symbol Avatar
                            _buildSymbolAvatar(),
                            const SizedBox(width: 14),

                            // Symbol Info
                            Expanded(child: _buildSymbolInfo(changeColor, isPositive)),

                            // Price Info
                            if (widget.currentPrice != null) _buildPriceInfo(changeColor, isPositive),

                            // Action Menu
                            _buildActionMenu(),
                          ],
                        ),

                        // Note Section
                        if (widget.item.hasNote) ...[
                          const SizedBox(height: 12),
                          _buildNoteSection(),
                        ],

                        // Tags Section
                        if (widget.item.hasTags) ...[
                          const SizedBox(height: 12),
                          _buildTagsSection(),
                        ],

                        // Alerts Section
                        if (widget.activeAlerts > 0) ...[
                          const SizedBox(height: 12),
                          _buildAlertsSection(),
                        ],

                        // Quick Actions
                        const SizedBox(height: 12),
                        _buildQuickActions(),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildSymbolAvatar() {
    return Container(
      width: 52,
      height: 52,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            AppColors.primaryBlue.withValues(alpha: 0.3),
            AppColors.primaryBlue.withValues(alpha: 0.1),
          ],
        ),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: AppColors.primaryBlue.withValues(alpha: 0.4),
          width: 2,
        ),
        boxShadow: [
          BoxShadow(
            color: AppColors.primaryBlue.withValues(alpha: 0.2),
            blurRadius: 12,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Center(
        child: Text(
          widget.item.ticker.length > 2
              ? widget.item.ticker.substring(0, 2)
              : widget.item.ticker,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w800,
            color: Colors.white,
            letterSpacing: -0.5,
          ),
        ),
      ),
    );
  }

  Widget _buildSymbolInfo(Color changeColor, bool isPositive) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          widget.item.ticker,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w800,
            color: Colors.white,
            letterSpacing: -0.3,
          ),
        ),
        const SizedBox(height: 4),
        if (widget.item.hasSignal)
          Row(
            children: [
              Container(
                width: 6,
                height: 6,
                decoration: BoxDecoration(
                  color: AppColors.successGreen,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.successGreen.withValues(alpha: 0.5),
                      blurRadius: 4,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 6),
              Flexible(
                child: Text(
                  widget.item.signal!,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: AppColors.successGreen,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          )
        else
          Text(
            'Added ${widget.item.daysSinceAdded} ${widget.item.daysSinceAdded == 1 ? "day" : "days"} ago',
            style: TextStyle(
              fontSize: 13,
              color: Colors.white.withValues(alpha: 0.5),
            ),
          ),
      ],
    );
  }

  Widget _buildPriceInfo(Color changeColor, bool isPositive) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text(
          '\$${widget.currentPrice!.toStringAsFixed(2)}',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: Colors.white,
            letterSpacing: -0.3,
          ),
        ),
        const SizedBox(height: 2),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
          decoration: BoxDecoration(
            color: changeColor.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                isPositive ? Icons.arrow_upward : Icons.arrow_downward,
                size: 12,
                color: changeColor,
              ),
              const SizedBox(width: 2),
              Text(
                '${isPositive ? '+' : ''}${widget.changePercent!.toStringAsFixed(2)}%',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  color: changeColor,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildActionMenu() {
    return PopupMenuButton<String>(
      icon: Icon(
        Icons.more_vert,
        color: Colors.white.withValues(alpha: 0.6),
        size: 20,
      ),
      color: const Color(0xFF1A1F3A),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      onSelected: (value) {
        HapticFeedback.lightImpact();
        switch (value) {
          case 'delete':
            widget.onDelete?.call();
            break;
          case 'note':
            widget.onEditNote?.call();
            break;
          case 'tags':
            widget.onEditTags?.call();
            break;
          case 'alert':
            widget.onSetAlert?.call();
            break;
        }
      },
      itemBuilder: (context) => [
        _buildMenuItem('note', Icons.note_add, 'Edit Note'),
        _buildMenuItem('tags', Icons.label, 'Edit Tags'),
        _buildMenuItem('alert', Icons.notifications_active, 'Set Alert',
            color: AppColors.warningOrange),
        const PopupMenuDivider(),
        _buildMenuItem('delete', Icons.delete_outline, 'Remove',
            color: AppColors.dangerRed),
      ],
    );
  }

  PopupMenuItem<String> _buildMenuItem(String value, IconData icon, String text,
      {Color? color}) {
    return PopupMenuItem(
      value: value,
      child: Row(
        children: [
          Icon(icon, size: 18, color: color ?? Colors.white70),
          const SizedBox(width: 12),
          Text(text, style: TextStyle(color: color ?? Colors.white)),
        ],
      ),
    );
  }

  Widget _buildNoteSection() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.white.withValues(alpha: 0.06),
            Colors.white.withValues(alpha: 0.02),
          ],
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.08),
          width: 1,
        ),
      ),
      child: Row(
        children: [
          Icon(
            Icons.sticky_note_2,
            size: 16,
            color: Colors.white.withValues(alpha: 0.4),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              widget.item.note!,
              style: TextStyle(
                fontSize: 13,
                color: Colors.white.withValues(alpha: 0.7),
                fontStyle: FontStyle.italic,
                height: 1.4,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTagsSection() {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: widget.item.tags.map((tag) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                AppColors.primaryBlue.withValues(alpha: 0.2),
                AppColors.primaryBlue.withValues(alpha: 0.1),
              ],
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: AppColors.primaryBlue.withValues(alpha: 0.3),
              width: 1,
            ),
          ),
          child: Text(
            tag,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.primaryBlue,
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildAlertsSection() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            AppColors.warningOrange.withValues(alpha: 0.15),
            AppColors.warningOrange.withValues(alpha: 0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: AppColors.warningOrange.withValues(alpha: 0.3),
          width: 1,
        ),
      ),
      child: Row(
        children: [
          Icon(
            Icons.notifications_active,
            size: 16,
            color: AppColors.warningOrange,
          ),
          const SizedBox(width: 10),
          Text(
            '${widget.activeAlerts} active ${widget.activeAlerts == 1 ? "alert" : "alerts"}',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: AppColors.warningOrange,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickActions() {
    return Row(
      children: [
        Expanded(
          child: _PremiumQuickActionButton(
            icon: widget.item.hasNote ? Icons.edit_note : Icons.note_add,
            label: widget.item.hasNote ? 'Edit Note' : 'Add Note',
            onTap: widget.onEditNote,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _PremiumQuickActionButton(
            icon: widget.item.hasTags ? Icons.label : Icons.label_outline,
            label: widget.item.hasTags ? 'Edit Tags' : 'Add Tags',
            onTap: widget.onEditTags,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _PremiumQuickActionButton(
            icon: Icons.notifications_active,
            label: 'Alert',
            color: AppColors.warningOrange,
            onTap: widget.onSetAlert,
          ),
        ),
      ],
    );
  }
}

class _PremiumQuickActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onTap;
  final Color? color;

  const _PremiumQuickActionButton({
    required this.icon,
    required this.label,
    this.onTap,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    final buttonColor = color ?? Colors.white;

    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap?.call();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              buttonColor.withValues(alpha: 0.1),
              buttonColor.withValues(alpha: 0.05),
            ],
          ),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: buttonColor.withValues(alpha: 0.2),
            width: 1,
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 14, color: buttonColor.withValues(alpha: 0.8)),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: buttonColor.withValues(alpha: 0.8),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM WATCHLIST HEADER
// =============================================================================

/// Premium watchlist stats header with animated counters
class PremiumWatchlistHeader extends StatefulWidget {
  final int totalSymbols;
  final int withSignals;
  final int withAlerts;
  final VoidCallback? onAddSymbol;

  const PremiumWatchlistHeader({
    super.key,
    required this.totalSymbols,
    required this.withSignals,
    this.withAlerts = 0,
    this.onAddSymbol,
  });

  @override
  State<PremiumWatchlistHeader> createState() => _PremiumWatchlistHeaderState();
}

class _PremiumWatchlistHeaderState extends State<PremiumWatchlistHeader>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            AppColors.primaryBlue.withValues(alpha: 0.15),
            AppColors.primaryBlue.withValues(alpha: 0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.1),
          width: 1,
        ),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Row(
            children: [
              Expanded(
                child: _buildStatItem(
                  value: widget.totalSymbols,
                  label: 'Symbols',
                  color: Colors.white,
                ),
              ),
              _buildDivider(),
              Expanded(
                child: _buildStatItem(
                  value: widget.withSignals,
                  label: 'With Signals',
                  color: AppColors.successGreen,
                ),
              ),
              if (widget.withAlerts > 0) ...[
                _buildDivider(),
                Expanded(
                  child: _buildStatItem(
                    value: widget.withAlerts,
                    label: 'Alerts',
                    color: AppColors.warningOrange,
                  ),
                ),
              ],
              if (widget.onAddSymbol != null) ...[
                const SizedBox(width: 16),
                _buildAddButton(),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatItem({
    required int value,
    required String label,
    required Color color,
  }) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Column(
          children: [
            Text(
              '${(value * _animation.value).round()}',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w800,
                color: color,
                letterSpacing: -1,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 13,
                color: Colors.white.withValues(alpha: 0.6),
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildDivider() {
    return Container(
      width: 1,
      height: 40,
      margin: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Colors.white.withValues(alpha: 0.0),
            Colors.white.withValues(alpha: 0.2),
            Colors.white.withValues(alpha: 0.0),
          ],
        ),
      ),
    );
  }

  Widget _buildAddButton() {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        widget.onAddSymbol?.call();
      },
      child: Container(
        width: 48,
        height: 48,
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              AppColors.primaryBlue,
              Color(0xFF3B7FD9),
            ],
          ),
          borderRadius: BorderRadius.circular(14),
          boxShadow: [
            BoxShadow(
              color: AppColors.primaryBlue.withValues(alpha: 0.3),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: const Icon(
          Icons.add,
          color: Colors.white,
          size: 24,
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM PORTFOLIO SUMMARY
// =============================================================================

/// Premium portfolio summary card with total value and performance
class PremiumPortfolioSummary extends StatefulWidget {
  final double totalValue;
  final double totalGain;
  final double gainPercent;
  final double dayChange;
  final double dayChangePercent;
  final List<double>? sparklineData;

  const PremiumPortfolioSummary({
    super.key,
    required this.totalValue,
    required this.totalGain,
    required this.gainPercent,
    required this.dayChange,
    required this.dayChangePercent,
    this.sparklineData,
  });

  @override
  State<PremiumPortfolioSummary> createState() => _PremiumPortfolioSummaryState();
}

class _PremiumPortfolioSummaryState extends State<PremiumPortfolioSummary>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _valueAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );
    _valueAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isPositive = widget.totalGain >= 0;
    final isDayPositive = widget.dayChange >= 0;
    final gainColor = isPositive ? AppColors.successGreen : AppColors.dangerRed;
    final dayColor = isDayPositive ? AppColors.successGreen : AppColors.dangerRed;

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            gainColor.withValues(alpha: 0.12),
            gainColor.withValues(alpha: 0.04),
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
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Portfolio Value',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: Colors.white.withValues(alpha: 0.6),
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: dayColor.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          isDayPositive ? Icons.trending_up : Icons.trending_down,
                          size: 14,
                          color: dayColor,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          '${isDayPositive ? '+' : ''}${widget.dayChangePercent.toStringAsFixed(2)}% today',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: dayColor,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // Total Value
              AnimatedBuilder(
                animation: _valueAnimation,
                builder: (context, child) {
                  return Text(
                    '\$${(widget.totalValue * _valueAnimation.value).toStringAsFixed(2)}',
                    style: const TextStyle(
                      fontSize: 36,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      letterSpacing: -1,
                    ),
                  );
                },
              ),

              // Total Gain
              Row(
                children: [
                  Icon(
                    isPositive ? Icons.arrow_upward : Icons.arrow_downward,
                    size: 18,
                    color: gainColor,
                  ),
                  const SizedBox(width: 4),
                  AnimatedBuilder(
                    animation: _valueAnimation,
                    builder: (context, child) {
                      return Text(
                        '\$${(widget.totalGain.abs() * _valueAnimation.value).toStringAsFixed(2)} (${isPositive ? '+' : ''}${widget.gainPercent.toStringAsFixed(2)}%)',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                          color: gainColor,
                        ),
                      );
                    },
                  ),
                  Text(
                    ' all time',
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.white.withValues(alpha: 0.5),
                    ),
                  ),
                ],
              ),

              // Sparkline
              if (widget.sparklineData != null && widget.sparklineData!.isNotEmpty) ...[
                const SizedBox(height: 20),
                SizedBox(
                  height: 60,
                  child: CustomPaint(
                    size: const Size(double.infinity, 60),
                    painter: SparklinePainter(
                      data: widget.sparklineData!,
                      color: gainColor,
                      strokeWidth: 2,
                      fillGradient: true,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// SPARKLINE PAINTER
// =============================================================================

/// Custom sparkline painter for performance charts
class SparklinePainter extends CustomPainter {
  final List<double> data;
  final Color color;
  final double strokeWidth;
  final bool fillGradient;

  SparklinePainter({
    required this.data,
    required this.color,
    this.strokeWidth = 2,
    this.fillGradient = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final minValue = data.reduce(math.min);
    final maxValue = data.reduce(math.max);
    final range = maxValue - minValue;

    final path = Path();
    final fillPath = Path();

    for (int i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final y = range == 0
          ? size.height / 2
          : size.height - ((data[i] - minValue) / range) * size.height;

      if (i == 0) {
        path.moveTo(x, y);
        fillPath.moveTo(x, size.height);
        fillPath.lineTo(x, y);
      } else {
        path.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
    }

    // Draw fill gradient
    if (fillGradient) {
      fillPath.lineTo(size.width, size.height);
      fillPath.close();

      final gradient = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          color.withValues(alpha: 0.3),
          color.withValues(alpha: 0.0),
        ],
      );

      final paint = Paint()
        ..shader = gradient.createShader(Rect.fromLTWH(0, 0, size.width, size.height));

      canvas.drawPath(fillPath, paint);
    }

    // Draw line
    final linePaint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    canvas.drawPath(path, linePaint);

    // Draw end dot
    final lastX = size.width;
    final lastY = range == 0
        ? size.height / 2
        : size.height - ((data.last - minValue) / range) * size.height;

    final dotPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    canvas.drawCircle(Offset(lastX, lastY), 4, dotPaint);

    // Draw glow around dot
    final glowPaint = Paint()
      ..color = color.withValues(alpha: 0.3)
      ..style = PaintingStyle.fill;

    canvas.drawCircle(Offset(lastX, lastY), 8, glowPaint);
  }

  @override
  bool shouldRepaint(SparklinePainter oldDelegate) {
    return oldDelegate.data != data || oldDelegate.color != color;
  }
}

// =============================================================================
// PREMIUM HOLDING ROW
// =============================================================================

/// Compact premium holding row for lists
class PremiumHoldingRow extends StatelessWidget {
  final String ticker;
  final String? companyName;
  final double shares;
  final double avgCost;
  final double currentPrice;
  final double? changePercent;
  final VoidCallback? onTap;

  const PremiumHoldingRow({
    super.key,
    required this.ticker,
    this.companyName,
    required this.shares,
    required this.avgCost,
    required this.currentPrice,
    this.changePercent,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final marketValue = shares * currentPrice;
    final costBasis = shares * avgCost;
    final gain = marketValue - costBasis;
    // ignore: unused_local_variable - Reserved for percentage display feature
    final gainPercent = costBasis > 0 ? (gain / costBasis) * 100 : 0;
    final isPositive = gain >= 0;
    final color = isPositive ? AppColors.successGreen : AppColors.dangerRed;

    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap?.call();
      },
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Colors.white.withValues(alpha: 0.06),
              Colors.white.withValues(alpha: 0.02),
            ],
          ),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.08),
            width: 1,
          ),
        ),
        child: Row(
          children: [
            // Symbol
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    color.withValues(alpha: 0.2),
                    color.withValues(alpha: 0.1),
                  ],
                ),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: color.withValues(alpha: 0.3),
                  width: 1.5,
                ),
              ),
              child: Center(
                child: Text(
                  ticker.length > 2 ? ticker.substring(0, 2) : ticker,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 12),

            // Info
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    ticker,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    '${shares.toStringAsFixed(shares == shares.roundToDouble() ? 0 : 2)} shares',
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.white.withValues(alpha: 0.5),
                    ),
                  ),
                ],
              ),
            ),

            // Value & Gain
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  '\$${marketValue.toStringAsFixed(2)}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 2),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isPositive ? Icons.arrow_upward : Icons.arrow_downward,
                      size: 12,
                      color: color,
                    ),
                    const SizedBox(width: 2),
                    Text(
                      '${isPositive ? '+' : ''}\$${gain.abs().toStringAsFixed(2)}',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: color,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM FILTER CHIP
// =============================================================================

/// Premium filter chip for tag filtering
class PremiumFilterChip extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback? onTap;

  const PremiumFilterChip({
    super.key,
    required this.label,
    required this.isSelected,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap?.call();
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          gradient: isSelected
              ? const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    AppColors.primaryBlue,
                    Color(0xFF3B7FD9),
                  ],
                )
              : LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Colors.white.withValues(alpha: 0.1),
                    Colors.white.withValues(alpha: 0.05),
                  ],
                ),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected
                ? AppColors.primaryBlue.withValues(alpha: 0.5)
                : Colors.white.withValues(alpha: 0.15),
            width: 1,
          ),
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: AppColors.primaryBlue.withValues(alpha: 0.3),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ]
              : null,
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 13,
            fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
            color: isSelected ? Colors.white : Colors.white.withValues(alpha: 0.7),
          ),
        ),
      ),
    );
  }
}
