/// Premium Button Component - Billion-dollar app quality
/// 
/// Features:
/// - Smooth scale animation on press
/// - Haptic feedback
/// - Gradient support
/// - Loading states
/// - Multiple variants (primary, secondary, outline, text)
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../theme/app_colors.dart';
import '../theme/spacing.dart';
import '../theme/border_radii.dart';

enum ButtonVariant {
  primary,    // Filled with gradient
  secondary,  // Filled with muted color
  outline,    // Outlined with transparent background
  text,       // Text only, no background
  danger,     // Red for destructive actions
  success,    // Green for positive actions
}

enum ButtonSize {
  small,      // Compact, 40px height
  medium,     // Standard, 48px height
  large,      // Prominent, 56px height
}

/// Premium button with smooth animations and haptic feedback
class PremiumButton extends StatefulWidget {
  final String text;
  final VoidCallback? onPressed;
  final IconData? icon;
  final IconData? trailingIcon;
  final ButtonVariant variant;
  final ButtonSize size;
  final bool isLoading;
  final bool isFullWidth;
  final bool enableHaptic;
  
  const PremiumButton({
    super.key,
    required this.text,
    this.onPressed,
    this.icon,
    this.trailingIcon,
    this.variant = ButtonVariant.primary,
    this.size = ButtonSize.medium,
    this.isLoading = false,
    this.isFullWidth = false,
    this.enableHaptic = true,
  });
  
  @override
  State<PremiumButton> createState() => _PremiumButtonState();
}

class _PremiumButtonState extends State<PremiumButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  bool _isPressed = false;
  
  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 100),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 0.95).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }
  
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  
  void _handleTapDown(TapDownDetails details) {
    if (widget.onPressed != null && !widget.isLoading) {
      setState(() => _isPressed = true);
      _controller.forward();
      if (widget.enableHaptic) {
        HapticFeedback.lightImpact();
      }
    }
  }
  
  void _handleTapUp(TapUpDetails details) {
    setState(() => _isPressed = false);
    _controller.reverse();
  }
  
  void _handleTapCancel() {
    setState(() => _isPressed = false);
    _controller.reverse();
  }
  
  @override
  Widget build(BuildContext context) {
    // ignore: unused_local_variable - Reserved for theme-aware styling
    final theme = Theme.of(context);
    final isDisabled = widget.onPressed == null || widget.isLoading;
    
    // Get button dimensions based on size
    final double height = switch (widget.size) {
      ButtonSize.small => 40.0,
      ButtonSize.medium => 48.0,
      ButtonSize.large => 56.0,
    };
    
    final double horizontalPadding = switch (widget.size) {
      ButtonSize.small => Spacing.md,
      ButtonSize.medium => Spacing.lg,
      ButtonSize.large => Spacing.xl,
    };
    
    final double fontSize = switch (widget.size) {
      ButtonSize.small => 14.0,
      ButtonSize.medium => 16.0,
      ButtonSize.large => 18.0,
    };
    
    final double iconSize = switch (widget.size) {
      ButtonSize.small => 18.0,
      ButtonSize.medium => 20.0,
      ButtonSize.large => 24.0,
    };
    
    // Get colors and decoration based on variant
    final decoration = _getDecoration(isDisabled);
    final textColor = _getTextColor(isDisabled);
    
    return GestureDetector(
      onTapDown: _handleTapDown,
      onTapUp: _handleTapUp,
      onTapCancel: _handleTapCancel,
      onTap: isDisabled ? null : widget.onPressed,
      child: ScaleTransition(
        scale: _scaleAnimation,
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 200),
          opacity: isDisabled ? 0.5 : 1.0,
          child: Container(
            height: height,
            width: widget.isFullWidth ? double.infinity : null,
            padding: EdgeInsets.symmetric(horizontal: horizontalPadding),
            decoration: decoration,
            child: widget.isLoading
                ? _buildLoadingIndicator(textColor)
                : _buildContent(textColor, fontSize, iconSize),
          ),
        ),
      ),
    );
  }
  
  Widget _buildContent(Color textColor, double fontSize, double iconSize) {
    return Row(
      mainAxisSize: widget.isFullWidth ? MainAxisSize.max : MainAxisSize.min,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        if (widget.icon != null) ...[
          Icon(widget.icon, size: iconSize, color: textColor),
          const SizedBox(width: Spacing.sm),
        ],
        Text(
          widget.text,
          style: TextStyle(
            fontSize: fontSize,
            fontWeight: FontWeight.w600,
            color: textColor,
            letterSpacing: 0.5,
          ),
        ),
        if (widget.trailingIcon != null) ...[
          const SizedBox(width: Spacing.sm),
          Icon(widget.trailingIcon, size: iconSize, color: textColor),
        ],
      ],
    );
  }
  
  Widget _buildLoadingIndicator(Color color) {
    return Center(
      child: SizedBox(
        width: 20,
        height: 20,
        child: CircularProgressIndicator(
          strokeWidth: 2,
          valueColor: AlwaysStoppedAnimation<Color>(color),
        ),
      ),
    );
  }
  
  BoxDecoration _getDecoration(bool isDisabled) {
    final borderRadius = BorderRadius.circular(BorderRadii.button);
    
    switch (widget.variant) {
      case ButtonVariant.primary:
        return BoxDecoration(
          gradient: isDisabled ? null : AppColors.primaryGradient,
          color: isDisabled ? AppColors.darkCard : null,
          borderRadius: borderRadius,
          boxShadow: isDisabled || _isPressed
              ? null
              : [
                  BoxShadow(
                    color: AppColors.technicBlue.withValues(alpha: 0.4),
                    blurRadius: 16,
                    offset: const Offset(0, 8),
                  ),
                ],
        );
        
      case ButtonVariant.secondary:
        return BoxDecoration(
          color: AppColors.darkCard,
          borderRadius: borderRadius,
          border: Border.all(
            color: AppColors.darkBorder,
            width: 1,
          ),
        );
        
      case ButtonVariant.outline:
        return BoxDecoration(
          color: Colors.transparent,
          borderRadius: borderRadius,
          border: Border.all(
            color: AppColors.technicBlue,
            width: 2,
          ),
        );
        
      case ButtonVariant.text:
        return BoxDecoration(
          color: _isPressed
              ? AppColors.technicBlue.withValues(alpha: 0.1)
              : Colors.transparent,
          borderRadius: borderRadius,
        );
        
      case ButtonVariant.danger:
        return BoxDecoration(
          gradient: isDisabled ? null : AppColors.dangerGradient,
          color: isDisabled ? AppColors.darkCard : null,
          borderRadius: borderRadius,
          boxShadow: isDisabled || _isPressed
              ? null
              : [
                  BoxShadow(
                    color: AppColors.dangerRed.withValues(alpha: 0.4),
                    blurRadius: 16,
                    offset: const Offset(0, 8),
                  ),
                ],
        );
        
      case ButtonVariant.success:
        return BoxDecoration(
          gradient: isDisabled ? null : AppColors.successGradient,
          color: isDisabled ? AppColors.darkCard : null,
          borderRadius: borderRadius,
          boxShadow: isDisabled || _isPressed
              ? null
              : [
                  BoxShadow(
                    color: AppColors.successGreen.withValues(alpha: 0.4),
                    blurRadius: 16,
                    offset: const Offset(0, 8),
                  ),
                ],
        );
    }
  }
  
  Color _getTextColor(bool isDisabled) {
    if (isDisabled) {
      return AppColors.darkTextTertiary;
    }
    
    switch (widget.variant) {
      case ButtonVariant.primary:
      case ButtonVariant.danger:
      case ButtonVariant.success:
        return Colors.white;
        
      case ButtonVariant.secondary:
        return AppColors.darkTextPrimary;
        
      case ButtonVariant.outline:
      case ButtonVariant.text:
        return AppColors.technicBlue;
    }
  }
}

/// Icon-only premium button (for FABs, toolbar actions)
class PremiumIconButton extends StatefulWidget {
  final IconData icon;
  final VoidCallback? onPressed;
  final ButtonVariant variant;
  final double size;
  final bool enableHaptic;
  final String? tooltip;
  
  const PremiumIconButton({
    super.key,
    required this.icon,
    this.onPressed,
    this.variant = ButtonVariant.primary,
    this.size = 48.0,
    this.enableHaptic = true,
    this.tooltip,
  });
  
  @override
  State<PremiumIconButton> createState() => _PremiumIconButtonState();
}

class _PremiumIconButtonState extends State<PremiumIconButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  
  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 100),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 0.9).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }
  
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  
  void _handleTap() {
    if (widget.onPressed != null) {
      _controller.forward().then((_) => _controller.reverse());
      if (widget.enableHaptic) {
        HapticFeedback.lightImpact();
      }
      widget.onPressed!();
    }
  }
  
  @override
  Widget build(BuildContext context) {
    final button = ScaleTransition(
      scale: _scaleAnimation,
      child: GestureDetector(
        onTap: _handleTap,
        child: Container(
          width: widget.size,
          height: widget.size,
          decoration: BoxDecoration(
            gradient: widget.variant == ButtonVariant.primary
                ? AppColors.primaryGradient
                : null,
            color: widget.variant == ButtonVariant.secondary
                ? AppColors.darkCard
                : null,
            shape: BoxShape.circle,
            boxShadow: widget.variant == ButtonVariant.primary
                ? [
                    BoxShadow(
                      color: AppColors.technicBlue.withValues(alpha: 0.4),
                      blurRadius: 16,
                      offset: const Offset(0, 8),
                    ),
                  ]
                : null,
          ),
          child: Icon(
            widget.icon,
            color: Colors.white,
            size: widget.size * 0.5,
          ),
        ),
      ),
    );
    
    if (widget.tooltip != null) {
      return Tooltip(
        message: widget.tooltip!,
        child: button,
      );
    }
    
    return button;
  }
}
