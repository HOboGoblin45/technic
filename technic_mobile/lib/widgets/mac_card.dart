/// Mac-style card widget with proper spacing, shadows, and rounded corners
library;

import 'package:flutter/material.dart';
import '../theme/spacing.dart';
import '../theme/border_radii.dart';
import '../theme/shadows.dart';

/// A card widget that follows Mac design principles
/// 
/// Features:
/// - Proper spacing using Spacing constants
/// - Rounded corners using BorderRadii constants
/// - Subtle shadows using Shadows constants
/// - Clean, minimal design
class MacCard extends StatelessWidget {
  final Widget child;
  final EdgeInsets? padding;
  final EdgeInsets? margin;
  final Color? backgroundColor;
  final VoidCallback? onTap;
  final bool elevated;
  final double? width;
  final double? height;
  
  const MacCard({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.backgroundColor,
    this.onTap,
    this.elevated = false,
    this.width,
    this.height,
  });
  
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bgColor = backgroundColor ?? theme.cardTheme.color ?? theme.colorScheme.surface;
    
    Widget card = Container(
      width: width,
      height: height,
      padding: padding ?? Spacing.cardEdgeInsets,
      margin: margin ?? Spacing.edgeInsetsMD,
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadii.cardBorderRadius,
        boxShadow: elevated ? Shadows.card : Shadows.subtle,
      ),
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

/// Compact card variant with less padding
class MacCardCompact extends StatelessWidget {
  final Widget child;
  final Color? backgroundColor;
  final VoidCallback? onTap;
  
  const MacCardCompact({
    super.key,
    required this.child,
    this.backgroundColor,
    this.onTap,
  });
  
  @override
  Widget build(BuildContext context) {
    return MacCard(
      padding: Spacing.edgeInsetsSM,
      margin: Spacing.edgeInsetsSM,
      backgroundColor: backgroundColor,
      onTap: onTap,
      child: child,
    );
  }
}

/// Elevated card variant with stronger shadow
class MacCardElevated extends StatelessWidget {
  final Widget child;
  final EdgeInsets? padding;
  final Color? backgroundColor;
  final VoidCallback? onTap;
  
  const MacCardElevated({
    super.key,
    required this.child,
    this.padding,
    this.backgroundColor,
    this.onTap,
  });
  
  @override
  Widget build(BuildContext context) {
    return MacCard(
      padding: padding,
      backgroundColor: backgroundColor,
      onTap: onTap,
      elevated: true,
      child: child,
    );
  }
}

/// Card with a header section
class MacCardWithHeader extends StatelessWidget {
  final String title;
  final Widget child;
  final Widget? trailing;
  final EdgeInsets? padding;
  final Color? backgroundColor;
  
  const MacCardWithHeader({
    super.key,
    required this.title,
    required this.child,
    this.trailing,
    this.padding,
    this.backgroundColor,
  });
  
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return MacCard(
      padding: EdgeInsets.zero,
      backgroundColor: backgroundColor,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header
          Padding(
            padding: padding ?? Spacing.edgeInsetsMD,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  title,
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (trailing != null) trailing!,
              ],
            ),
          ),
          // Divider
          Divider(
            height: 1,
            thickness: 1,
            color: theme.dividerColor.withOpacity(0.1),
          ),
          // Content
          Padding(
            padding: padding ?? Spacing.edgeInsetsMD,
            child: child,
          ),
        ],
      ),
    );
  }
}
