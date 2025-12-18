/// Glassmorphism container widget with frosted glass effect
library;

import 'dart:ui';
import 'package:flutter/material.dart';
import '../theme/spacing.dart';
import '../theme/border_radii.dart';

/// A container with glassmorphism (frosted glass) effect
/// 
/// Features:
/// - Backdrop blur filter
/// - Semi-transparent background
/// - Subtle border
/// - Mac-style aesthetic
class GlassContainer extends StatelessWidget {
  final Widget child;
  final double? width;
  final double? height;
  final EdgeInsets? padding;
  final EdgeInsets? margin;
  final double blur;
  final double opacity;
  final Color? color;
  final BorderRadius? borderRadius;
  final Border? border;
  
  const GlassContainer({
    super.key,
    required this.child,
    this.width,
    this.height,
    this.padding,
    this.margin,
    this.blur = 10.0,
    this.opacity = 0.1,
    this.color,
    this.borderRadius,
    this.border,
  });
  
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    
    // Default color based on theme
    final defaultColor = isDark 
        ? Colors.white.withValues(alpha: opacity)
        : Colors.black.withValues(alpha: opacity);
    
    // Default border
    final defaultBorder = Border.all(
      color: isDark 
          ? Colors.white.withValues(alpha: 0.2)
          : Colors.black.withValues(alpha: 0.1),
      width: 1,
    );
    
    return Container(
      width: width,
      height: height,
      margin: margin,
      child: ClipRRect(
        borderRadius: borderRadius ?? BorderRadii.cardBorderRadius,
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
          child: Container(
            padding: padding ?? Spacing.edgeInsetsMD,
            decoration: BoxDecoration(
              color: color ?? defaultColor,
              borderRadius: borderRadius ?? BorderRadii.cardBorderRadius,
              border: border ?? defaultBorder,
            ),
            child: child,
          ),
        ),
      ),
    );
  }
}

/// Glass card variant with standard padding
class GlassCard extends StatelessWidget {
  final Widget child;
  final double? width;
  final double? height;
  final VoidCallback? onTap;
  
  const GlassCard({
    super.key,
    required this.child,
    this.width,
    this.height,
    this.onTap,
  });
  
  @override
  Widget build(BuildContext context) {
    Widget card = GlassContainer(
      width: width,
      height: height,
      padding: Spacing.cardEdgeInsets,
      margin: Spacing.edgeInsetsMD,
      child: child,
    );
    
    if (onTap != null) {
      return InkWell(
        onTap: onTap,
        borderRadius: BorderRadii.cardBorderRadius,
        child: card,
      );
    }
    
    return card;
  }
}

/// Glass navigation bar with frosted effect
class GlassNavigationBar extends StatelessWidget {
  final Widget child;
  final double height;
  
  const GlassNavigationBar({
    super.key,
    required this.child,
    this.height = 80.0,
  });
  
  @override
  Widget build(BuildContext context) {
    return GlassContainer(
      height: height,
      padding: EdgeInsets.zero,
      margin: EdgeInsets.zero,
      blur: 20.0,
      opacity: 0.15,
      borderRadius: BorderRadius.zero,
      border: Border(
        top: BorderSide(
          color: Colors.white.withValues(alpha: 0.1),
          width: 1,
        ),
      ),
      child: child,
    );
  }
}

/// Glass modal/dialog with strong blur
class GlassModal extends StatelessWidget {
  final Widget child;
  final double? width;
  final double? height;
  
  const GlassModal({
    super.key,
    required this.child,
    this.width,
    this.height,
  });
  
  @override
  Widget build(BuildContext context) {
    return GlassContainer(
      width: width,
      height: height,
      padding: Spacing.edgeInsetsXL,
      blur: 30.0,
      opacity: 0.2,
      borderRadius: BorderRadii.modalBorderRadius,
      child: child,
    );
  }
}

/// Glass button with subtle effect
class GlassButton extends StatelessWidget {
  final String text;
  final VoidCallback? onPressed;
  final IconData? icon;
  
  const GlassButton({
    super.key,
    required this.text,
    this.onPressed,
    this.icon,
  });
  
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return InkWell(
      onTap: onPressed,
      borderRadius: BorderRadii.buttonBorderRadius,
      child: GlassContainer(
        padding: const EdgeInsets.symmetric(
          horizontal: Spacing.lg,
          vertical: Spacing.md,
        ),
        margin: EdgeInsets.zero,
        blur: 15.0,
        opacity: 0.15,
        borderRadius: BorderRadii.buttonBorderRadius,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[
              Icon(icon, size: 20),
              const SizedBox(width: Spacing.sm),
            ],
            Text(
              text,
              style: theme.textTheme.bodyLarge?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
