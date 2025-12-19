/// Premium Card Component - Billion-dollar app quality
/// 
/// Features:
/// - Enhanced glass morphism with backdrop blur
/// - Gradient backgrounds
/// - Smooth hover/press states
/// - Elevation shadows
/// - Multiple variants
library;

import 'dart:ui';
import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import '../theme/spacing.dart';
import '../theme/border_radii.dart';

enum CardVariant {
  glass,      // Glass morphism with blur
  elevated,   // Solid with shadow
  gradient,   // Gradient background
  outline,    // Outlined with transparent background
}

enum CardElevation {
  none,       // Flat, no shadow
  low,        // Subtle shadow
  medium,     // Standard shadow
  high,       // Prominent shadow
}

/// Premium card with glass morphism and smooth animations
class PremiumCard extends StatefulWidget {
  final Widget child;
  final VoidCallback? onTap;
  final CardVariant variant;
  final CardElevation elevation;
  final EdgeInsets? padding;
  final EdgeInsets? margin;
  final double? width;
  final double? height;
  final Gradient? gradient;
  final Color? backgroundColor;
  final bool enablePressEffect;
  
  const PremiumCard({
    super.key,
    required this.child,
    this.onTap,
    this.variant = CardVariant.glass,
    this.elevation = CardElevation.medium,
    this.padding,
    this.margin,
    this.width,
    this.height,
    this.gradient,
    this.backgroundColor,
    this.enablePressEffect = true,
  });
  
  @override
  State<PremiumCard> createState() => _PremiumCardState();
}

class _PremiumCardState extends State<PremiumCard>
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
  
  void _handleTapDown(TapDownDetails details) {
    if (widget.onTap != null && widget.enablePressEffect) {
      setState(() => _isPressed = true);
      _controller.forward();
    }
  }
  
  void _handleTapUp(TapUpDetails details) {
    if (widget.enablePressEffect) {
      setState(() => _isPressed = false);
      _controller.reverse();
    }
  }
  
  void _handleTapCancel() {
    if (widget.enablePressEffect) {
      setState(() => _isPressed = false);
      _controller.reverse();
    }
  }
  
  @override
  Widget build(BuildContext context) {
    final card = Container(
      width: widget.width,
      height: widget.height,
      margin: widget.margin ?? const EdgeInsets.all(Spacing.sm),
      child: _buildCardContent(),
    );
    
    if (widget.onTap != null) {
      return GestureDetector(
        onTapDown: _handleTapDown,
        onTapUp: _handleTapUp,
        onTapCancel: _handleTapCancel,
        onTap: widget.onTap,
        child: widget.enablePressEffect
            ? ScaleTransition(scale: _scaleAnimation, child: card)
            : card,
      );
    }
    
    return card;
  }
  
  Widget _buildCardContent() {
    switch (widget.variant) {
      case CardVariant.glass:
        return _buildGlassCard();
      case CardVariant.elevated:
        return _buildElevatedCard();
      case CardVariant.gradient:
        return _buildGradientCard();
      case CardVariant.outline:
        return _buildOutlineCard();
    }
  }
  
  Widget _buildGlassCard() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(BorderRadii.card),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: widget.padding ?? const EdgeInsets.all(Spacing.md),
          decoration: BoxDecoration(
            color: AppColors.darkCard.withValues(alpha: 0.7),
            borderRadius: BorderRadius.circular(BorderRadii.card),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
              width: 1,
            ),
            boxShadow: _getBoxShadow(),
          ),
          child: widget.child,
        ),
      ),
    );
  }
  
  Widget _buildElevatedCard() {
    return Container(
      padding: widget.padding ?? const EdgeInsets.all(Spacing.md),
      decoration: BoxDecoration(
        color: widget.backgroundColor ?? AppColors.darkCard,
        borderRadius: BorderRadius.circular(BorderRadii.card),
        boxShadow: _getBoxShadow(),
      ),
      child: widget.child,
    );
  }
  
  Widget _buildGradientCard() {
    return Container(
      padding: widget.padding ?? const EdgeInsets.all(Spacing.md),
      decoration: BoxDecoration(
        gradient: widget.gradient ?? AppColors.cardGradient,
        borderRadius: BorderRadius.circular(BorderRadii.card),
        boxShadow: _getBoxShadow(),
      ),
      child: widget.child,
    );
  }
  
  Widget _buildOutlineCard() {
    return Container(
      padding: widget.padding ?? const EdgeInsets.all(Spacing.md),
      decoration: BoxDecoration(
        color: Colors.transparent,
        borderRadius: BorderRadius.circular(BorderRadii.card),
        border: Border.all(
          color: AppColors.darkBorder,
          width: 1,
        ),
      ),
      child: widget.child,
    );
  }
  
  List<BoxShadow>? _getBoxShadow() {
    if (_isPressed) return null;
    
    switch (widget.elevation) {
      case CardElevation.none:
        return null;
        
      case CardElevation.low:
        return [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ];
        
      case CardElevation.medium:
        return [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.2),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ];
        
      case CardElevation.high:
        return [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            blurRadius: 24,
            offset: const Offset(0, 8),
          ),
        ];
    }
  }
}

/// Stock result card with enhanced visuals
class StockResultCard extends StatelessWidget {
  final String symbol;
  final String companyName;
  final double price;
  final double changePercent;
  final double techRating;
  final double meritScore;
  final VoidCallback? onTap;
  
  const StockResultCard({
    super.key,
    required this.symbol,
    required this.companyName,
    required this.price,
    required this.changePercent,
    required this.techRating,
    required this.meritScore,
    this.onTap,
  });
  
  @override
  Widget build(BuildContext context) {
    final isPositive = changePercent >= 0;
    final changeColor = isPositive ? AppColors.successGreen : AppColors.dangerRed;
    
    return PremiumCard(
      onTap: onTap,
      variant: CardVariant.glass,
      elevation: CardElevation.medium,
      child: Row(
        children: [
          // Left: Symbol and company
          Expanded(
            flex: 2,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  symbol,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.darkTextPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  companyName,
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppColors.darkTextSecondary,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          
          // Middle: Price and change
          Expanded(
            flex: 2,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  '\$${price.toStringAsFixed(2)}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: AppColors.darkTextPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: changeColor.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    '${isPositive ? '+' : ''}${changePercent.toStringAsFixed(2)}%',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: changeColor,
                    ),
                  ),
                ),
              ],
            ),
          ),
          
          const SizedBox(width: Spacing.md),
          
          // Right: Ratings
          Column(
            children: [
              _buildRatingPill('Tech', techRating, AppColors.technicBlue),
              const SizedBox(height: 4),
              _buildRatingPill('Merit', meritScore, AppColors.infoPurple),
            ],
          ),
        ],
      ),
    );
  }
  
  Widget _buildRatingPill(String label, double value, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(BorderRadii.sm),
        border: Border.all(
          color: color.withValues(alpha: 0.4),
          width: 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              color: color,
            ),
          ),
          const SizedBox(width: 4),
          Text(
            value.toStringAsFixed(1),
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

/// Metric card for displaying key statistics
class MetricCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color? color;
  final String? subtitle;
  
  const MetricCard({
    super.key,
    required this.label,
    required this.value,
    required this.icon,
    this.color,
    this.subtitle,
  });
  
  @override
  Widget build(BuildContext context) {
    final accentColor = color ?? AppColors.technicBlue;
    
    return PremiumCard(
      variant: CardVariant.elevated,
      elevation: CardElevation.low,
      padding: const EdgeInsets.all(Spacing.md),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: accentColor.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(
                  icon,
                  size: 20,
                  color: accentColor,
                ),
              ),
              const SizedBox(width: Spacing.sm),
              Expanded(
                child: Text(
                  label,
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppColors.darkTextSecondary,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: Spacing.sm),
          Text(
            value,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: AppColors.darkTextPrimary,
            ),
          ),
          if (subtitle != null) ...[
            const SizedBox(height: 4),
            Text(
              subtitle!,
              style: const TextStyle(
                fontSize: 11,
                color: AppColors.darkTextTertiary,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
