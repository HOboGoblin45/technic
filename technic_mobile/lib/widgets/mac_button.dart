/// Mac-style button widget with proper spacing, shadows, and animations
library;

import 'package:flutter/material.dart';
import '../theme/spacing.dart';
import '../theme/border_radii.dart';
import '../theme/shadows.dart';
import '../theme/animations.dart';

/// A button widget that follows Mac design principles
/// 
/// Features:
/// - Proper spacing using Spacing constants
/// - Rounded corners using BorderRadii constants
/// - Subtle shadows using Shadows constants
/// - Smooth animations using Animations constants
/// - Press feedback with scale animation
class MacButton extends StatefulWidget {
  final String text;
  final VoidCallback? onPressed;
  final Color? backgroundColor;
  final Color? textColor;
  final IconData? icon;
  final bool isLoading;
  final bool isFullWidth;
  final MacButtonSize size;
  
  const MacButton({
    super.key,
    required this.text,
    this.onPressed,
    this.backgroundColor,
    this.textColor,
    this.icon,
    this.isLoading = false,
    this.isFullWidth = false,
    this.size = MacButtonSize.medium,
  });
  
  @override
  State<MacButton> createState() => _MacButtonState();
}

class _MacButtonState extends State<MacButton> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  bool _isPressed = false;
  
  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: Animations.fast,
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 0.95).animate(
      CurvedAnimation(parent: _controller, curve: Animations.defaultCurve),
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
    }
  }
  
  void _handleTapUp(TapUpDetails details) {
    if (_isPressed) {
      setState(() => _isPressed = false);
      _controller.reverse();
    }
  }
  
  void _handleTapCancel() {
    if (_isPressed) {
      setState(() => _isPressed = false);
      _controller.reverse();
    }
  }
  
  EdgeInsets _getPadding() {
    switch (widget.size) {
      case MacButtonSize.small:
        return const EdgeInsets.symmetric(
          horizontal: Spacing.md,
          vertical: Spacing.sm,
        );
      case MacButtonSize.medium:
        return const EdgeInsets.symmetric(
          horizontal: Spacing.lg,
          vertical: Spacing.md,
        );
      case MacButtonSize.large:
        return const EdgeInsets.symmetric(
          horizontal: Spacing.xl,
          vertical: Spacing.lg,
        );
    }
  }
  
  double _getFontSize() {
    switch (widget.size) {
      case MacButtonSize.small:
        return 14.0;
      case MacButtonSize.medium:
        return 16.0;
      case MacButtonSize.large:
        return 18.0;
    }
  }
  
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final backgroundColor = widget.backgroundColor ?? theme.colorScheme.primary;
    final textColor = widget.textColor ?? Colors.white;
    final isDisabled = widget.onPressed == null || widget.isLoading;
    
    Widget buttonContent = Row(
      mainAxisSize: widget.isFullWidth ? MainAxisSize.max : MainAxisSize.min,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        if (widget.isLoading)
          SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              valueColor: AlwaysStoppedAnimation<Color>(textColor),
            ),
          )
        else if (widget.icon != null)
          Icon(
            widget.icon,
            size: _getFontSize(),
            color: textColor,
          ),
        if ((widget.isLoading || widget.icon != null) && widget.text.isNotEmpty)
          const SizedBox(width: Spacing.sm),
        if (widget.text.isNotEmpty)
          Text(
            widget.text,
            style: TextStyle(
              color: textColor,
              fontSize: _getFontSize(),
              fontWeight: FontWeight.w600,
            ),
          ),
      ],
    );
    
    return GestureDetector(
      onTapDown: _handleTapDown,
      onTapUp: _handleTapUp,
      onTapCancel: _handleTapCancel,
      onTap: isDisabled ? null : widget.onPressed,
      child: AnimatedBuilder(
        animation: _scaleAnimation,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: Container(
              padding: _getPadding(),
              decoration: BoxDecoration(
                color: isDisabled 
                    ? backgroundColor.withValues(alpha: 0.5)
                    : backgroundColor,
                borderRadius: BorderRadii.buttonBorderRadius,
                boxShadow: isDisabled || _isPressed ? null : Shadows.button,
              ),
              child: buttonContent,
            ),
          );
        },
      ),
    );
  }
}

/// Button size variants
enum MacButtonSize {
  small,
  medium,
  large,
}

/// Secondary button variant (outlined style)
class MacButtonSecondary extends StatelessWidget {
  final String text;
  final VoidCallback? onPressed;
  final Color? borderColor;
  final Color? textColor;
  final IconData? icon;
  final bool isLoading;
  final MacButtonSize size;
  
  const MacButtonSecondary({
    super.key,
    required this.text,
    this.onPressed,
    this.borderColor,
    this.textColor,
    this.icon,
    this.isLoading = false,
    this.size = MacButtonSize.medium,
  });
  
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final textColorFinal = textColor ?? theme.colorScheme.primary;
    
    return MacButton(
      text: text,
      onPressed: onPressed,
      backgroundColor: Colors.transparent,
      textColor: textColorFinal,
      icon: icon,
      isLoading: isLoading,
      size: size,
    );
  }
}
