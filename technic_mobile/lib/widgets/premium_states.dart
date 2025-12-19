/// Premium State Components
///
/// Professional loading, empty, error, and success states with:
/// - Glass morphism design
/// - Smooth animations
/// - Premium visual effects
/// - Consistent design language
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui';
import 'dart:math' as math;

import '../theme/app_colors.dart';

// =============================================================================
// PREMIUM SHIMMER LOADING
// =============================================================================

/// Premium shimmer effect with glass morphism
class PremiumShimmer extends StatefulWidget {
  final Widget child;
  final bool isLoading;
  final Color? baseColor;
  final Color? highlightColor;
  final Duration duration;

  const PremiumShimmer({
    super.key,
    required this.child,
    this.isLoading = true,
    this.baseColor,
    this.highlightColor,
    this.duration = const Duration(milliseconds: 1500),
  });

  @override
  State<PremiumShimmer> createState() => _PremiumShimmerState();
}

class _PremiumShimmerState extends State<PremiumShimmer>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.duration,
    )..repeat();

    _animation = Tween<double>(begin: -1.0, end: 2.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOutSine),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isLoading) {
      return widget.child;
    }

    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return ShaderMask(
          blendMode: BlendMode.srcATop,
          shaderCallback: (bounds) {
            return LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                widget.baseColor ?? const Color(0xFF1A1F3A),
                widget.highlightColor ?? const Color(0xFF2A3050),
                widget.baseColor ?? const Color(0xFF1A1F3A),
              ],
              stops: [
                (_animation.value - 0.3).clamp(0.0, 1.0),
                _animation.value.clamp(0.0, 1.0),
                (_animation.value + 0.3).clamp(0.0, 1.0),
              ],
            ).createShader(bounds);
          },
          child: child,
        );
      },
      child: widget.child,
    );
  }
}

/// Premium skeleton box with glass morphism
class PremiumSkeletonBox extends StatelessWidget {
  final double? width;
  final double? height;
  final BorderRadius? borderRadius;
  final bool animate;

  const PremiumSkeletonBox({
    super.key,
    this.width,
    this.height,
    this.borderRadius,
    this.animate = true,
  });

  @override
  Widget build(BuildContext context) {
    final box = Container(
      width: width,
      height: height,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.white.withValues(alpha: 0.08),
            Colors.white.withValues(alpha: 0.04),
          ],
        ),
        borderRadius: borderRadius ?? BorderRadius.circular(8),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.05),
          width: 1,
        ),
      ),
    );

    if (animate) {
      return PremiumShimmer(child: box);
    }
    return box;
  }
}

/// Premium loading card skeleton
class PremiumCardSkeleton extends StatelessWidget {
  final double? height;
  final int rows;

  const PremiumCardSkeleton({
    super.key,
    this.height,
    this.rows = 3,
  });

  @override
  Widget build(BuildContext context) {
    return PremiumShimmer(
      child: Container(
        height: height,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Colors.white.withValues(alpha: 0.06),
              Colors.white.withValues(alpha: 0.02),
            ],
          ),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.08),
            width: 1,
          ),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Header row
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      PremiumSkeletonBox(
                        width: 100,
                        height: 24,
                        borderRadius: BorderRadius.circular(6),
                        animate: false,
                      ),
                      PremiumSkeletonBox(
                        width: 60,
                        height: 24,
                        borderRadius: BorderRadius.circular(12),
                        animate: false,
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  // Content rows
                  ...List.generate(rows, (index) => Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: PremiumSkeletonBox(
                      width: double.infinity,
                      height: 16,
                      borderRadius: BorderRadius.circular(4),
                      animate: false,
                    ),
                  )),
                  const Spacer(),
                  // Footer row
                  Row(
                    children: [
                      Expanded(
                        child: PremiumSkeletonBox(
                          height: 40,
                          borderRadius: BorderRadius.circular(12),
                          animate: false,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: PremiumSkeletonBox(
                          height: 40,
                          borderRadius: BorderRadius.circular(12),
                          animate: false,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Premium list skeleton
class PremiumListSkeleton extends StatelessWidget {
  final int itemCount;
  final double itemHeight;
  final double spacing;

  const PremiumListSkeleton({
    super.key,
    this.itemCount = 5,
    this.itemHeight = 80,
    this.spacing = 12,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: List.generate(itemCount, (index) {
        return Padding(
          padding: EdgeInsets.only(bottom: index < itemCount - 1 ? spacing : 0),
          child: PremiumShimmer(
            child: Container(
              height: itemHeight,
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
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    PremiumSkeletonBox(
                      width: 48,
                      height: 48,
                      borderRadius: BorderRadius.circular(12),
                      animate: false,
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          PremiumSkeletonBox(
                            width: 120,
                            height: 16,
                            borderRadius: BorderRadius.circular(4),
                            animate: false,
                          ),
                          const SizedBox(height: 8),
                          PremiumSkeletonBox(
                            width: 80,
                            height: 12,
                            borderRadius: BorderRadius.circular(4),
                            animate: false,
                          ),
                        ],
                      ),
                    ),
                    PremiumSkeletonBox(
                      width: 60,
                      height: 32,
                      borderRadius: BorderRadius.circular(8),
                      animate: false,
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      }),
    );
  }
}

// =============================================================================
// PREMIUM EMPTY STATE
// =============================================================================

/// Premium empty state with animations
class PremiumEmptyState extends StatefulWidget {
  final IconData icon;
  final String title;
  final String message;
  final String? actionLabel;
  final VoidCallback? onAction;
  final Color? accentColor;

  const PremiumEmptyState({
    super.key,
    required this.icon,
    required this.title,
    required this.message,
    this.actionLabel,
    this.onAction,
    this.accentColor,
  });

  /// Watchlist empty state
  factory PremiumEmptyState.watchlist({VoidCallback? onAddSymbol}) {
    return PremiumEmptyState(
      icon: Icons.bookmark_outline,
      title: 'Your Watchlist is Empty',
      message: 'Add symbols to track your favorite stocks and get notified of opportunities.',
      actionLabel: 'Add Symbol',
      onAction: onAddSymbol,
      accentColor: AppColors.primaryBlue,
    );
  }

  /// No scan results
  factory PremiumEmptyState.noResults({VoidCallback? onAdjustFilters}) {
    return PremiumEmptyState(
      icon: Icons.search_off,
      title: 'No Results Found',
      message: 'Try adjusting your filters or scan parameters to find more opportunities.',
      actionLabel: 'Adjust Filters',
      onAction: onAdjustFilters,
      accentColor: Colors.grey,
    );
  }

  /// No internet
  factory PremiumEmptyState.noInternet({VoidCallback? onRetry}) {
    return PremiumEmptyState(
      icon: Icons.wifi_off,
      title: 'No Internet Connection',
      message: 'Please check your connection and try again.',
      actionLabel: 'Retry',
      onAction: onRetry,
      accentColor: AppColors.warningOrange,
    );
  }

  /// No history
  factory PremiumEmptyState.noHistory({VoidCallback? onStartScan}) {
    return PremiumEmptyState(
      icon: Icons.history,
      title: 'No Scan History',
      message: 'Your scan history will appear here once you run your first scan.',
      actionLabel: 'Run Scan',
      onAction: onStartScan,
      accentColor: AppColors.primaryBlue,
    );
  }

  @override
  State<PremiumEmptyState> createState() => _PremiumEmptyStateState();
}

class _PremiumEmptyStateState extends State<PremiumEmptyState>
    with TickerProviderStateMixin {
  late AnimationController _fadeController;
  late AnimationController _pulseController;
  late Animation<double> _fadeAnimation;
  late Animation<double> _slideAnimation;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();

    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );

    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 2000),
      vsync: this,
    )..repeat(reverse: true);

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _fadeController,
        curve: const Interval(0.0, 0.6, curve: Curves.easeOut),
      ),
    );

    _slideAnimation = Tween<double>(begin: 30.0, end: 0.0).animate(
      CurvedAnimation(
        parent: _fadeController,
        curve: Curves.easeOutCubic,
      ),
    );

    _pulseAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(
        parent: _pulseController,
        curve: Curves.easeInOut,
      ),
    );

    _fadeController.forward();
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = widget.accentColor ?? AppColors.primaryBlue;

    return AnimatedBuilder(
      animation: Listenable.merge([_fadeAnimation, _slideAnimation]),
      builder: (context, child) {
        return Opacity(
          opacity: _fadeAnimation.value,
          child: Transform.translate(
            offset: Offset(0, _slideAnimation.value),
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // Animated icon container
                    AnimatedBuilder(
                      animation: _pulseAnimation,
                      builder: (context, child) {
                        return Transform.scale(
                          scale: _pulseAnimation.value,
                          child: Container(
                            width: 120,
                            height: 120,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                                colors: [
                                  color.withValues(alpha: 0.15),
                                  color.withValues(alpha: 0.05),
                                ],
                              ),
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: color.withValues(alpha: 0.2),
                                width: 2,
                              ),
                              boxShadow: [
                                BoxShadow(
                                  color: color.withValues(alpha: 0.2),
                                  blurRadius: 30,
                                  spreadRadius: 5,
                                ),
                              ],
                            ),
                            child: Icon(
                              widget.icon,
                              size: 56,
                              color: color,
                            ),
                          ),
                        );
                      },
                    ),
                    const SizedBox(height: 32),

                    // Title
                    Text(
                      widget.title,
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                        letterSpacing: -0.5,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 12),

                    // Message
                    Text(
                      widget.message,
                      style: TextStyle(
                        fontSize: 15,
                        color: Colors.white.withValues(alpha: 0.6),
                        height: 1.5,
                      ),
                      textAlign: TextAlign.center,
                    ),

                    // Action button
                    if (widget.actionLabel != null && widget.onAction != null) ...[
                      const SizedBox(height: 32),
                      _buildActionButton(color),
                    ],
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildActionButton(Color color) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        widget.onAction?.call();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              color,
              color.withValues(alpha: 0.8),
            ],
          ),
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: color.withValues(alpha: 0.3),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Text(
          widget.actionLabel!,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: Colors.white,
            letterSpacing: -0.3,
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM ERROR STATE
// =============================================================================

/// Premium error state with animations
class PremiumErrorState extends StatefulWidget {
  final IconData icon;
  final String title;
  final String message;
  final String? retryLabel;
  final VoidCallback? onRetry;
  final String? secondaryLabel;
  final VoidCallback? onSecondary;

  const PremiumErrorState({
    super.key,
    required this.icon,
    required this.title,
    required this.message,
    this.retryLabel = 'Try Again',
    this.onRetry,
    this.secondaryLabel,
    this.onSecondary,
  });

  /// Network error
  factory PremiumErrorState.network({VoidCallback? onRetry}) {
    return PremiumErrorState(
      icon: Icons.wifi_off,
      title: 'No Internet Connection',
      message: 'Please check your connection and try again.',
      onRetry: onRetry,
    );
  }

  /// Server error
  factory PremiumErrorState.server({VoidCallback? onRetry}) {
    return PremiumErrorState(
      icon: Icons.cloud_off,
      title: 'Server Error',
      message: 'Our servers are experiencing issues. Please try again later.',
      onRetry: onRetry,
    );
  }

  /// Timeout error
  factory PremiumErrorState.timeout({VoidCallback? onRetry}) {
    return PremiumErrorState(
      icon: Icons.access_time,
      title: 'Request Timed Out',
      message: 'The request took too long. Please try again.',
      onRetry: onRetry,
    );
  }

  /// Generic error
  factory PremiumErrorState.generic({
    required String message,
    VoidCallback? onRetry,
  }) {
    return PremiumErrorState(
      icon: Icons.error_outline,
      title: 'Something Went Wrong',
      message: message,
      onRetry: onRetry,
    );
  }

  @override
  State<PremiumErrorState> createState() => _PremiumErrorStateState();
}

class _PremiumErrorStateState extends State<PremiumErrorState>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _shakeAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );

    _shakeAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _controller, curve: Curves.elasticOut),
    );

    // Shake animation on mount
    Future.delayed(const Duration(milliseconds: 300), () {
      if (mounted) {
        _controller.forward();
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Animated error icon
            AnimatedBuilder(
              animation: _shakeAnimation,
              builder: (context, child) {
                return Transform.rotate(
                  angle: math.sin(_shakeAnimation.value * math.pi * 4) * 0.05,
                  child: Container(
                    width: 100,
                    height: 100,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [
                          AppColors.dangerRed.withValues(alpha: 0.15),
                          AppColors.dangerRed.withValues(alpha: 0.05),
                        ],
                      ),
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: AppColors.dangerRed.withValues(alpha: 0.3),
                        width: 2,
                      ),
                    ),
                    child: Icon(
                      widget.icon,
                      size: 48,
                      color: AppColors.dangerRed,
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 32),

            // Title
            Text(
              widget.title,
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.w800,
                color: Colors.white,
                letterSpacing: -0.5,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),

            // Message
            Text(
              widget.message,
              style: TextStyle(
                fontSize: 15,
                color: Colors.white.withValues(alpha: 0.6),
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),

            // Buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                if (widget.onSecondary != null && widget.secondaryLabel != null) ...[
                  _buildSecondaryButton(),
                  const SizedBox(width: 16),
                ],
                if (widget.onRetry != null) _buildRetryButton(),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRetryButton() {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        widget.onRetry?.call();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              AppColors.primaryBlue,
              Color(0xFF3B7FD9),
            ],
          ),
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: AppColors.primaryBlue.withValues(alpha: 0.3),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.refresh, size: 20, color: Colors.white),
            const SizedBox(width: 8),
            Text(
              widget.retryLabel!,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSecondaryButton() {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        widget.onSecondary?.call();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Colors.white.withValues(alpha: 0.1),
              Colors.white.withValues(alpha: 0.05),
            ],
          ),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.2),
            width: 1,
          ),
        ),
        child: Text(
          widget.secondaryLabel!,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Colors.white.withValues(alpha: 0.8),
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM SUCCESS ANIMATION
// =============================================================================

/// Premium success animation overlay
class PremiumSuccessAnimation extends StatefulWidget {
  final String? message;
  final VoidCallback? onComplete;
  final Duration duration;

  const PremiumSuccessAnimation({
    super.key,
    this.message,
    this.onComplete,
    this.duration = const Duration(milliseconds: 2000),
  });

  /// Show success animation as overlay
  static void show(
    BuildContext context, {
    String? message,
    VoidCallback? onComplete,
  }) {
    showDialog(
      context: context,
      barrierDismissible: false,
      barrierColor: Colors.black54,
      builder: (context) => PremiumSuccessAnimation(
        message: message,
        onComplete: () {
          Navigator.of(context).pop();
          onComplete?.call();
        },
      ),
    );
  }

  @override
  State<PremiumSuccessAnimation> createState() => _PremiumSuccessAnimationState();
}

class _PremiumSuccessAnimationState extends State<PremiumSuccessAnimation>
    with TickerProviderStateMixin {
  late AnimationController _scaleController;
  late AnimationController _checkController;
  late AnimationController _ringController;
  late Animation<double> _scaleAnimation;
  late Animation<double> _checkAnimation;
  late Animation<double> _ringAnimation;

  @override
  void initState() {
    super.initState();

    // Scale animation
    _scaleController = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _scaleController, curve: Curves.elasticOut),
    );

    // Check animation
    _checkController = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    _checkAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _checkController, curve: Curves.easeOutCubic),
    );

    // Ring animation
    _ringController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _ringAnimation = Tween<double>(begin: 0.8, end: 1.5).animate(
      CurvedAnimation(parent: _ringController, curve: Curves.easeOut),
    );

    // Sequence animations
    _scaleController.forward();
    Future.delayed(const Duration(milliseconds: 200), () {
      if (mounted) {
        _checkController.forward();
        _ringController.forward();
        HapticFeedback.mediumImpact();
      }
    });

    // Auto dismiss
    Future.delayed(widget.duration, () {
      if (mounted) {
        widget.onComplete?.call();
      }
    });
  }

  @override
  void dispose() {
    _scaleController.dispose();
    _checkController.dispose();
    _ringController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Animated success circle
          AnimatedBuilder(
            animation: Listenable.merge([_scaleAnimation, _ringAnimation]),
            builder: (context, child) {
              return Stack(
                alignment: Alignment.center,
                children: [
                  // Expanding ring
                  Transform.scale(
                    scale: _ringAnimation.value,
                    child: Opacity(
                      opacity: (1 - (_ringAnimation.value - 0.8) / 0.7).clamp(0.0, 1.0),
                      child: Container(
                        width: 100,
                        height: 100,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: AppColors.successGreen,
                            width: 3,
                          ),
                        ),
                      ),
                    ),
                  ),
                  // Main circle
                  Transform.scale(
                    scale: _scaleAnimation.value,
                    child: Container(
                      width: 100,
                      height: 100,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [
                            AppColors.successGreen,
                            AppColors.successGreen.withValues(alpha: 0.8),
                          ],
                        ),
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: AppColors.successGreen.withValues(alpha: 0.4),
                            blurRadius: 30,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                      child: AnimatedBuilder(
                        animation: _checkAnimation,
                        builder: (context, child) {
                          return CustomPaint(
                            painter: CheckmarkPainter(
                              progress: _checkAnimation.value,
                              color: Colors.white,
                              strokeWidth: 4,
                            ),
                          );
                        },
                      ),
                    ),
                  ),
                ],
              );
            },
          ),

          // Message
          if (widget.message != null) ...[
            const SizedBox(height: 32),
            AnimatedBuilder(
              animation: _scaleAnimation,
              builder: (context, child) {
                return Opacity(
                  opacity: _scaleAnimation.value,
                  child: Transform.translate(
                    offset: Offset(0, 20 * (1 - _scaleAnimation.value)),
                    child: Text(
                      widget.message!,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                        letterSpacing: -0.3,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                );
              },
            ),
          ],
        ],
      ),
    );
  }
}

/// Custom painter for animated checkmark
class CheckmarkPainter extends CustomPainter {
  final double progress;
  final Color color;
  final double strokeWidth;

  CheckmarkPainter({
    required this.progress,
    required this.color,
    required this.strokeWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final center = Offset(size.width / 2, size.height / 2);

    // Checkmark points (relative to center)
    final start = Offset(center.dx - 18, center.dy);
    final middle = Offset(center.dx - 5, center.dy + 15);
    final end = Offset(center.dx + 20, center.dy - 12);

    final path = Path();

    if (progress <= 0.5) {
      // First half of animation: draw first part of check
      final t = progress * 2;
      path.moveTo(start.dx, start.dy);
      path.lineTo(
        start.dx + (middle.dx - start.dx) * t,
        start.dy + (middle.dy - start.dy) * t,
      );
    } else {
      // Second half: complete the check
      path.moveTo(start.dx, start.dy);
      path.lineTo(middle.dx, middle.dy);

      final t = (progress - 0.5) * 2;
      path.lineTo(
        middle.dx + (end.dx - middle.dx) * t,
        middle.dy + (end.dy - middle.dy) * t,
      );
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(CheckmarkPainter oldDelegate) {
    return oldDelegate.progress != progress;
  }
}

// =============================================================================
// PREMIUM LOADING OVERLAY
// =============================================================================

/// Premium loading overlay with blur
class PremiumLoadingOverlay extends StatelessWidget {
  final String? message;
  final bool show;
  final Widget child;

  const PremiumLoadingOverlay({
    super.key,
    this.message,
    required this.show,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        child,
        if (show)
          Positioned.fill(
            child: Container(
              color: Colors.black54,
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 5, sigmaY: 5),
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Animated loading indicator
                      Container(
                        width: 60,
                        height: 60,
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              Colors.white.withValues(alpha: 0.1),
                              Colors.white.withValues(alpha: 0.05),
                            ],
                          ),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.1),
                            width: 1,
                          ),
                        ),
                        child: const Center(
                          child: SizedBox(
                            width: 30,
                            height: 30,
                            child: CircularProgressIndicator(
                              strokeWidth: 3,
                              valueColor: AlwaysStoppedAnimation<Color>(
                                AppColors.primaryBlue,
                              ),
                            ),
                          ),
                        ),
                      ),
                      if (message != null) ...[
                        const SizedBox(height: 20),
                        Text(
                          message!,
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.white.withValues(alpha: 0.8),
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }
}
