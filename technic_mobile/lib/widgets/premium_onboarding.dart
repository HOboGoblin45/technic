/// Premium Onboarding & Tutorial Widgets
///
/// A collection of premium onboarding components with glass morphism,
/// smooth animations, and professional styling.
///
/// Components:
/// - PremiumPageIndicator: Animated page indicators
/// - PremiumOnboardingPage: Full-screen onboarding page
/// - PremiumFeatureSpotlight: Overlay feature highlight
/// - PremiumCoachMark: Tooltip-style guided tour
/// - PremiumProgressStepper: Multi-step progress indicator
/// - PremiumWelcomeCard: Welcome banner card
/// - PremiumFeatureCard: Feature highlight card
/// - PremiumSplashLogo: Animated splash logo
library;

import 'dart:async';
import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_colors.dart';
import '../utils/helpers.dart';

// =============================================================================
// PAGE INDICATOR STYLES
// =============================================================================

/// Style options for page indicators
enum PageIndicatorStyle {
  dots,
  expandingDots,
  line,
  numbers,
  dashes,
}

// =============================================================================
// PREMIUM PAGE INDICATOR
// =============================================================================

/// Animated page indicator with multiple styles
class PremiumPageIndicator extends StatelessWidget {
  final int currentPage;
  final int pageCount;
  final PageIndicatorStyle style;
  final Color? activeColor;
  final Color? inactiveColor;
  final double size;
  final double spacing;
  final ValueChanged<int>? onPageTap;

  const PremiumPageIndicator({
    super.key,
    required this.currentPage,
    required this.pageCount,
    this.style = PageIndicatorStyle.expandingDots,
    this.activeColor,
    this.inactiveColor,
    this.size = 8,
    this.spacing = 8,
    this.onPageTap,
  });

  @override
  Widget build(BuildContext context) {
    final active = activeColor ?? AppColors.primaryBlue;
    final inactive = inactiveColor ?? Colors.white.withValues(alpha: 0.3);

    switch (style) {
      case PageIndicatorStyle.dots:
        return _buildDots(active, inactive);
      case PageIndicatorStyle.expandingDots:
        return _buildExpandingDots(active, inactive);
      case PageIndicatorStyle.line:
        return _buildLine(active, inactive);
      case PageIndicatorStyle.numbers:
        return _buildNumbers(active, inactive);
      case PageIndicatorStyle.dashes:
        return _buildDashes(active, inactive);
    }
  }

  Widget _buildDots(Color active, Color inactive) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(pageCount, (index) {
        final isActive = index == currentPage;
        return GestureDetector(
          onTap: onPageTap != null ? () => onPageTap!(index) : null,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            margin: EdgeInsets.symmetric(horizontal: spacing / 2),
            width: size,
            height: size,
            decoration: BoxDecoration(
              color: isActive ? active : inactive,
              shape: BoxShape.circle,
              boxShadow: isActive
                  ? [
                      BoxShadow(
                        color: active.withValues(alpha: 0.5),
                        blurRadius: 8,
                      ),
                    ]
                  : null,
            ),
          ),
        );
      }),
    );
  }

  Widget _buildExpandingDots(Color active, Color inactive) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(pageCount, (index) {
        final isActive = index == currentPage;
        return GestureDetector(
          onTap: onPageTap != null ? () => onPageTap!(index) : null,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOutCubic,
            margin: EdgeInsets.symmetric(horizontal: spacing / 2),
            width: isActive ? size * 3 : size,
            height: size,
            decoration: BoxDecoration(
              color: isActive ? active : inactive,
              borderRadius: BorderRadius.circular(size / 2),
              boxShadow: isActive
                  ? [
                      BoxShadow(
                        color: active.withValues(alpha: 0.5),
                        blurRadius: 8,
                      ),
                    ]
                  : null,
            ),
          ),
        );
      }),
    );
  }

  Widget _buildLine(Color active, Color inactive) {
    return Container(
      width: 200,
      height: 4,
      decoration: BoxDecoration(
        color: inactive,
        borderRadius: BorderRadius.circular(2),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          return Stack(
            children: [
              AnimatedPositioned(
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeOutCubic,
                left: (constraints.maxWidth / pageCount) * currentPage,
                child: Container(
                  width: constraints.maxWidth / pageCount,
                  height: 4,
                  decoration: BoxDecoration(
                    color: active,
                    borderRadius: BorderRadius.circular(2),
                    boxShadow: [
                      BoxShadow(
                        color: active.withValues(alpha: 0.5),
                        blurRadius: 8,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildNumbers(Color active, Color inactive) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '${currentPage + 1}',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: active,
          ),
        ),
        Text(
          ' / $pageCount',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w400,
            color: inactive,
          ),
        ),
      ],
    );
  }

  Widget _buildDashes(Color active, Color inactive) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(pageCount, (index) {
        final isActive = index == currentPage;
        final isPast = index < currentPage;
        return GestureDetector(
          onTap: onPageTap != null ? () => onPageTap!(index) : null,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            margin: EdgeInsets.symmetric(horizontal: spacing / 2),
            width: 24,
            height: 4,
            decoration: BoxDecoration(
              color: isActive || isPast ? active : inactive,
              borderRadius: BorderRadius.circular(2),
              boxShadow: isActive
                  ? [
                      BoxShadow(
                        color: active.withValues(alpha: 0.5),
                        blurRadius: 8,
                      ),
                    ]
                  : null,
            ),
          ),
        );
      }),
    );
  }
}

// =============================================================================
// PREMIUM ONBOARDING PAGE
// =============================================================================

/// Data model for onboarding page content
class OnboardingPageData {
  final String title;
  final String description;
  final IconData icon;
  final Color? accentColor;
  final String? buttonText;
  final List<String>? features;

  const OnboardingPageData({
    required this.title,
    required this.description,
    required this.icon,
    this.accentColor,
    this.buttonText,
    this.features,
  });
}

/// Premium onboarding page with animations
class PremiumOnboardingPage extends StatefulWidget {
  final OnboardingPageData data;
  final bool isActive;
  final VoidCallback? onAction;

  const PremiumOnboardingPage({
    super.key,
    required this.data,
    this.isActive = true,
    this.onAction,
  });

  @override
  State<PremiumOnboardingPage> createState() => _PremiumOnboardingPageState();
}

class _PremiumOnboardingPageState extends State<PremiumOnboardingPage>
    with TickerProviderStateMixin {
  late AnimationController _iconController;
  late AnimationController _contentController;
  late Animation<double> _iconScale;
  late Animation<double> _iconGlow;
  late Animation<double> _contentFade;
  late Animation<Offset> _contentSlide;

  @override
  void initState() {
    super.initState();

    // Icon animation
    _iconController = AnimationController(
      duration: const Duration(milliseconds: 2000),
      vsync: this,
    );

    _iconScale = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _iconController, curve: Curves.easeInOut),
    );

    _iconGlow = Tween<double>(begin: 0.3, end: 0.6).animate(
      CurvedAnimation(parent: _iconController, curve: Curves.easeInOut),
    );

    // Content animation
    _contentController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );

    _contentFade = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _contentController,
        curve: const Interval(0.0, 0.6, curve: Curves.easeOut),
      ),
    );

    _contentSlide = Tween<Offset>(
      begin: const Offset(0, 0.1),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(
        parent: _contentController,
        curve: const Interval(0.0, 0.8, curve: Curves.easeOutCubic),
      ),
    );

    if (widget.isActive) {
      _startAnimations();
    }
  }

  @override
  void didUpdateWidget(PremiumOnboardingPage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isActive && !oldWidget.isActive) {
      _startAnimations();
    } else if (!widget.isActive && oldWidget.isActive) {
      _stopAnimations();
    }
  }

  void _startAnimations() {
    _iconController.repeat(reverse: true);
    _contentController.forward();
  }

  void _stopAnimations() {
    _iconController.stop();
    _contentController.reset();
  }

  @override
  void dispose() {
    _iconController.dispose();
    _contentController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final accentColor = widget.data.accentColor ?? AppColors.primaryBlue;

    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Animated icon
          AnimatedBuilder(
            animation: _iconController,
            builder: (context, child) {
              return Transform.scale(
                scale: _iconScale.value,
                child: Container(
                  width: 160,
                  height: 160,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        accentColor.withValues(alpha: 0.3),
                        accentColor.withValues(alpha: 0.1),
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: accentColor.withValues(alpha: 0.4),
                      width: 2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: accentColor.withValues(alpha: _iconGlow.value),
                        blurRadius: 40,
                        spreadRadius: 8,
                      ),
                    ],
                  ),
                  child: Icon(
                    widget.data.icon,
                    size: 72,
                    color: accentColor,
                  ),
                ),
              );
            },
          ),

          const SizedBox(height: 48),

          // Content with slide animation
          SlideTransition(
            position: _contentSlide,
            child: FadeTransition(
              opacity: _contentFade,
              child: Column(
                children: [
                  // Title
                  Text(
                    widget.data.title,
                    style: const TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      letterSpacing: -0.5,
                    ),
                    textAlign: TextAlign.center,
                  ),

                  const SizedBox(height: 16),

                  // Description
                  Text(
                    widget.data.description,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w400,
                      color: Colors.white.withValues(alpha: 0.7),
                      height: 1.6,
                    ),
                    textAlign: TextAlign.center,
                  ),

                  // Features list
                  if (widget.data.features != null) ...[
                    const SizedBox(height: 32),
                    ...widget.data.features!.map((feature) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 12),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Container(
                              width: 24,
                              height: 24,
                              decoration: BoxDecoration(
                                color: accentColor.withValues(alpha: 0.2),
                                shape: BoxShape.circle,
                              ),
                              child: Icon(
                                Icons.check,
                                size: 14,
                                color: accentColor,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Text(
                              feature,
                              style: const TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w500,
                                color: Colors.white70,
                              ),
                            ),
                          ],
                        ),
                      );
                    }),
                  ],

                  // Action button
                  if (widget.data.buttonText != null) ...[
                    const SizedBox(height: 32),
                    _PremiumOnboardingButton(
                      label: widget.data.buttonText!,
                      color: accentColor,
                      onTap: widget.onAction,
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Premium onboarding action button
class _PremiumOnboardingButton extends StatefulWidget {
  final String label;
  final Color color;
  final VoidCallback? onTap;

  const _PremiumOnboardingButton({
    required this.label,
    required this.color,
    this.onTap,
  });

  @override
  State<_PremiumOnboardingButton> createState() =>
      _PremiumOnboardingButtonState();
}

class _PremiumOnboardingButtonState extends State<_PremiumOnboardingButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scale;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 150),
      vsync: this,
    );
    _scale = Tween<double>(begin: 1.0, end: 0.95).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => _controller.forward(),
      onTapUp: (_) {
        _controller.reverse();
        HapticFeedback.lightImpact();
        widget.onTap?.call();
      },
      onTapCancel: () => _controller.reverse(),
      child: AnimatedBuilder(
        animation: _scale,
        builder: (context, child) {
          return Transform.scale(
            scale: _scale.value,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 48, vertical: 16),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    widget.color,
                    widget.color.withValues(alpha: 0.8),
                  ],
                ),
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: widget.color.withValues(alpha: 0.4),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: Text(
                widget.label,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                  letterSpacing: 0.5,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

// =============================================================================
// PREMIUM FEATURE SPOTLIGHT
// =============================================================================

/// Spotlight overlay for highlighting UI features
class PremiumFeatureSpotlight extends StatefulWidget {
  final Offset targetPosition;
  final Size targetSize;
  final String title;
  final String description;
  final VoidCallback? onNext;
  final VoidCallback? onSkip;
  final int currentStep;
  final int totalSteps;
  final bool showSkip;

  const PremiumFeatureSpotlight({
    super.key,
    required this.targetPosition,
    required this.targetSize,
    required this.title,
    required this.description,
    this.onNext,
    this.onSkip,
    this.currentStep = 1,
    this.totalSteps = 1,
    this.showSkip = true,
  });

  /// Show spotlight overlay
  static OverlayEntry show(
    BuildContext context, {
    required GlobalKey targetKey,
    required String title,
    required String description,
    VoidCallback? onNext,
    VoidCallback? onSkip,
    int currentStep = 1,
    int totalSteps = 1,
    bool showSkip = true,
  }) {
    final renderBox =
        targetKey.currentContext?.findRenderObject() as RenderBox?;
    final position = renderBox?.localToGlobal(Offset.zero) ?? Offset.zero;
    final size = renderBox?.size ?? Size.zero;

    final overlay = Overlay.of(context);
    late OverlayEntry entry;

    entry = OverlayEntry(
      builder: (context) => PremiumFeatureSpotlight(
        targetPosition: position,
        targetSize: size,
        title: title,
        description: description,
        currentStep: currentStep,
        totalSteps: totalSteps,
        showSkip: showSkip,
        onNext: () {
          entry.remove();
          onNext?.call();
        },
        onSkip: () {
          entry.remove();
          onSkip?.call();
        },
      ),
    );

    overlay.insert(entry);
    return entry;
  }

  @override
  State<PremiumFeatureSpotlight> createState() =>
      _PremiumFeatureSpotlightState();
}

class _PremiumFeatureSpotlightState extends State<PremiumFeatureSpotlight>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.0, 0.3, curve: Curves.easeOut),
      ),
    );

    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.1).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );

    _controller.repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final targetCenter = widget.targetPosition +
        Offset(widget.targetSize.width / 2, widget.targetSize.height / 2);

    // Determine tooltip position (above or below target)
    final showAbove = targetCenter.dy > screenSize.height / 2;

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return FadeTransition(
          opacity: _fadeAnimation,
          child: Stack(
            children: [
              // Dark overlay with spotlight cutout
              CustomPaint(
                size: screenSize,
                painter: _SpotlightPainter(
                  targetPosition: widget.targetPosition,
                  targetSize: widget.targetSize,
                  pulseScale: _pulseAnimation.value,
                ),
              ),

              // Tooltip card
              Positioned(
                left: 24,
                right: 24,
                top: showAbove
                    ? widget.targetPosition.dy - 180
                    : widget.targetPosition.dy + widget.targetSize.height + 24,
                child: _buildTooltipCard(),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildTooltipCard() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white.withValues(alpha: 0.2)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Step indicator
              if (widget.totalSteps > 1)
                Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: Text(
                    'Step ${widget.currentStep} of ${widget.totalSteps}',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: AppColors.primaryBlue,
                      letterSpacing: 0.5,
                    ),
                  ),
                ),

              // Title
              Text(
                widget.title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),

              const SizedBox(height: 8),

              // Description
              Text(
                widget.description,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w400,
                  color: Colors.white.withValues(alpha: 0.7),
                  height: 1.5,
                ),
              ),

              const SizedBox(height: 20),

              // Buttons
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  if (widget.showSkip)
                    TextButton(
                      onPressed: () {
                        HapticFeedback.lightImpact();
                        widget.onSkip?.call();
                      },
                      child: const Text(
                        'Skip Tour',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                          color: Colors.white54,
                        ),
                      ),
                    )
                  else
                    const SizedBox(),
                  ElevatedButton(
                    onPressed: () {
                      HapticFeedback.lightImpact();
                      widget.onNext?.call();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primaryBlue,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 24,
                        vertical: 12,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: Text(
                      widget.currentStep == widget.totalSteps
                          ? 'Got it!'
                          : 'Next',
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Custom painter for spotlight overlay
class _SpotlightPainter extends CustomPainter {
  final Offset targetPosition;
  final Size targetSize;
  final double pulseScale;

  _SpotlightPainter({
    required this.targetPosition,
    required this.targetSize,
    required this.pulseScale,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.black.withValues(alpha: 0.85);

    // Draw dark overlay
    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      paint,
    );

    // Calculate spotlight rect with padding
    final padding = 8.0;
    final spotlightRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(
        targetPosition.dx - padding,
        targetPosition.dy - padding,
        targetSize.width + padding * 2,
        targetSize.height + padding * 2,
      ),
      const Radius.circular(12),
    );

    // Cut out spotlight area
    final clearPaint = Paint()
      ..blendMode = BlendMode.clear
      ..style = PaintingStyle.fill;

    canvas.drawRRect(spotlightRect, clearPaint);

    // Draw pulsing border
    final borderPaint = Paint()
      ..color = AppColors.primaryBlue.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final scaledRect = RRect.fromRectAndRadius(
      Rect.fromCenter(
        center: spotlightRect.center,
        width: spotlightRect.width * pulseScale,
        height: spotlightRect.height * pulseScale,
      ),
      const Radius.circular(12),
    );

    canvas.drawRRect(scaledRect, borderPaint);
  }

  @override
  bool shouldRepaint(covariant _SpotlightPainter oldDelegate) {
    return pulseScale != oldDelegate.pulseScale;
  }
}

// =============================================================================
// PREMIUM COACH MARK
// =============================================================================

/// Position options for coach mark tooltip
enum CoachMarkPosition {
  top,
  bottom,
  left,
  right,
  auto,
}

/// Tooltip-style coach mark for guided tours
class PremiumCoachMark extends StatefulWidget {
  final Widget child;
  final String title;
  final String description;
  final bool isVisible;
  final CoachMarkPosition position;
  final VoidCallback? onDismiss;
  final Color? accentColor;

  const PremiumCoachMark({
    super.key,
    required this.child,
    required this.title,
    required this.description,
    this.isVisible = false,
    this.position = CoachMarkPosition.auto,
    this.onDismiss,
    this.accentColor,
  });

  @override
  State<PremiumCoachMark> createState() => _PremiumCoachMarkState();
}

class _PremiumCoachMarkState extends State<PremiumCoachMark>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _fadeAnimation;
  final GlobalKey _childKey = GlobalKey();

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutBack),
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    if (widget.isVisible) {
      _controller.forward();
    }
  }

  @override
  void didUpdateWidget(PremiumCoachMark oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isVisible && !oldWidget.isVisible) {
      _controller.forward();
    } else if (!widget.isVisible && oldWidget.isVisible) {
      _controller.reverse();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        // Child widget with highlight
        Container(
          key: _childKey,
          decoration: widget.isVisible
              ? BoxDecoration(
                  borderRadius: BorderRadius.circular(8),
                  boxShadow: [
                    BoxShadow(
                      color: (widget.accentColor ?? AppColors.primaryBlue)
                          .withValues(alpha: 0.4),
                      blurRadius: 16,
                      spreadRadius: 2,
                    ),
                  ],
                )
              : null,
          child: widget.child,
        ),

        // Tooltip
        if (widget.isVisible)
          Positioned(
            bottom: -120,
            left: 0,
            right: 0,
            child: AnimatedBuilder(
              animation: _controller,
              builder: (context, child) {
                return Transform.scale(
                  scale: _scaleAnimation.value,
                  child: Opacity(
                    opacity: _fadeAnimation.value,
                    child: _buildTooltip(),
                  ),
                );
              },
            ),
          ),
      ],
    );
  }

  Widget _buildTooltip() {
    final accentColor = widget.accentColor ?? AppColors.primaryBlue;

    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        widget.onDismiss?.call();
      },
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 16),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.white.withValues(alpha: 0.12),
                    Colors.white.withValues(alpha: 0.06),
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: accentColor.withValues(alpha: 0.3),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 32,
                        height: 32,
                        decoration: BoxDecoration(
                          color: accentColor.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Icon(
                          Icons.lightbulb_outline,
                          size: 18,
                          color: accentColor,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          widget.title,
                          style: const TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      Icon(
                        Icons.close,
                        size: 18,
                        color: Colors.white.withValues(alpha: 0.5),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    widget.description,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w400,
                      color: Colors.white.withValues(alpha: 0.7),
                      height: 1.4,
                    ),
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

// =============================================================================
// PREMIUM PROGRESS STEPPER
// =============================================================================

/// Step data for progress stepper
class StepData {
  final String label;
  final IconData icon;
  final String? description;

  const StepData({
    required this.label,
    required this.icon,
    this.description,
  });
}

/// Multi-step progress indicator
class PremiumProgressStepper extends StatelessWidget {
  final int currentStep;
  final List<StepData> steps;
  final Color? activeColor;
  final Color? completedColor;
  final Color? inactiveColor;
  final bool showLabels;
  final bool vertical;
  final ValueChanged<int>? onStepTap;

  const PremiumProgressStepper({
    super.key,
    required this.currentStep,
    required this.steps,
    this.activeColor,
    this.completedColor,
    this.inactiveColor,
    this.showLabels = true,
    this.vertical = false,
    this.onStepTap,
  });

  @override
  Widget build(BuildContext context) {
    final active = activeColor ?? AppColors.primaryBlue;
    final completed = completedColor ?? AppColors.successGreen;
    final inactive = inactiveColor ?? Colors.white.withValues(alpha: 0.2);

    if (vertical) {
      return _buildVerticalStepper(active, completed, inactive);
    }
    return _buildHorizontalStepper(active, completed, inactive);
  }

  Widget _buildHorizontalStepper(
      Color active, Color completed, Color inactive) {
    return Row(
      children: List.generate(steps.length * 2 - 1, (index) {
        if (index.isOdd) {
          // Connector line
          final stepIndex = index ~/ 2;
          final isCompleted = stepIndex < currentStep;
          return Expanded(
            child: Container(
              height: 3,
              margin: const EdgeInsets.symmetric(horizontal: 4),
              decoration: BoxDecoration(
                color: isCompleted ? completed : inactive,
                borderRadius: BorderRadius.circular(1.5),
              ),
            ),
          );
        }

        // Step indicator
        final stepIndex = index ~/ 2;
        final isActive = stepIndex == currentStep;
        final isCompleted = stepIndex < currentStep;

        return GestureDetector(
          onTap: onStepTap != null ? () => onStepTap!(stepIndex) : null,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildStepIndicator(
                stepIndex,
                isActive,
                isCompleted,
                active,
                completed,
                inactive,
              ),
              if (showLabels) ...[
                const SizedBox(height: 8),
                SizedBox(
                  width: 70,
                  child: Text(
                    steps[stepIndex].label,
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
                      color: isActive || isCompleted
                          ? Colors.white
                          : Colors.white54,
                    ),
                    textAlign: TextAlign.center,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ],
          ),
        );
      }),
    );
  }

  Widget _buildVerticalStepper(Color active, Color completed, Color inactive) {
    return Column(
      children: List.generate(steps.length * 2 - 1, (index) {
        if (index.isOdd) {
          // Connector line
          final stepIndex = index ~/ 2;
          final isCompleted = stepIndex < currentStep;
          return Container(
            width: 3,
            height: 40,
            margin: const EdgeInsets.symmetric(vertical: 4),
            decoration: BoxDecoration(
              color: isCompleted ? completed : inactive,
              borderRadius: BorderRadius.circular(1.5),
            ),
          );
        }

        // Step indicator
        final stepIndex = index ~/ 2;
        final isActive = stepIndex == currentStep;
        final isCompleted = stepIndex < currentStep;

        return GestureDetector(
          onTap: onStepTap != null ? () => onStepTap!(stepIndex) : null,
          child: Row(
            children: [
              _buildStepIndicator(
                stepIndex,
                isActive,
                isCompleted,
                active,
                completed,
                inactive,
              ),
              if (showLabels) ...[
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        steps[stepIndex].label,
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight:
                              isActive ? FontWeight.w700 : FontWeight.w500,
                          color: isActive || isCompleted
                              ? Colors.white
                              : Colors.white54,
                        ),
                      ),
                      if (steps[stepIndex].description != null) ...[
                        const SizedBox(height: 4),
                        Text(
                          steps[stepIndex].description!,
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w400,
                            color: Colors.white.withValues(alpha: 0.5),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ],
            ],
          ),
        );
      }),
    );
  }

  Widget _buildStepIndicator(
    int stepIndex,
    bool isActive,
    bool isCompleted,
    Color active,
    Color completed,
    Color inactive,
  ) {
    final color = isCompleted
        ? completed
        : isActive
            ? active
            : inactive;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: isActive ? 48 : 40,
      height: isActive ? 48 : 40,
      decoration: BoxDecoration(
        color: isCompleted
            ? completed
            : isActive
                ? active.withValues(alpha: 0.2)
                : inactive.withValues(alpha: 0.1),
        shape: BoxShape.circle,
        border: Border.all(
          color: color,
          width: isActive ? 2 : 1.5,
        ),
        boxShadow: isActive
            ? [
                BoxShadow(
                  color: active.withValues(alpha: 0.4),
                  blurRadius: 12,
                ),
              ]
            : null,
      ),
      child: Icon(
        isCompleted ? Icons.check : steps[stepIndex].icon,
        size: isActive ? 24 : 20,
        color: isCompleted
            ? Colors.white
            : isActive
                ? active
                : inactive,
      ),
    );
  }
}

// =============================================================================
// PREMIUM WELCOME CARD
// =============================================================================

/// Premium welcome banner card
class PremiumWelcomeCard extends StatefulWidget {
  final String title;
  final String subtitle;
  final String? message;
  final IconData icon;
  final Color? accentColor;
  final VoidCallback? onDismiss;
  final VoidCallback? onAction;
  final String? actionLabel;

  const PremiumWelcomeCard({
    super.key,
    required this.title,
    required this.subtitle,
    this.message,
    this.icon = Icons.waving_hand,
    this.accentColor,
    this.onDismiss,
    this.onAction,
    this.actionLabel,
  });

  @override
  State<PremiumWelcomeCard> createState() => _PremiumWelcomeCardState();
}

class _PremiumWelcomeCardState extends State<PremiumWelcomeCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _waveAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );

    _waveAnimation = Tween<double>(begin: -0.1, end: 0.1).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );

    // Wave animation
    _controller.repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final accentColor = widget.accentColor ?? AppColors.primaryBlue;

    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                accentColor.withValues(alpha: 0.15),
                accentColor.withValues(alpha: 0.05),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: accentColor.withValues(alpha: 0.3),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  // Animated icon
                  AnimatedBuilder(
                    animation: _waveAnimation,
                    builder: (context, child) {
                      return Transform.rotate(
                        angle: _waveAnimation.value,
                        child: Container(
                          width: 48,
                          height: 48,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                accentColor,
                                accentColor.withValues(alpha: 0.7),
                              ],
                            ),
                            borderRadius: BorderRadius.circular(14),
                            boxShadow: [
                              BoxShadow(
                                color: accentColor.withValues(alpha: 0.4),
                                blurRadius: 12,
                              ),
                            ],
                          ),
                          child: Icon(
                            widget.icon,
                            color: Colors.white,
                            size: 24,
                          ),
                        ),
                      );
                    },
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          widget.title,
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          widget.subtitle,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w400,
                            color: Colors.white.withValues(alpha: 0.7),
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (widget.onDismiss != null)
                    IconButton(
                      onPressed: () {
                        HapticFeedback.lightImpact();
                        widget.onDismiss?.call();
                      },
                      icon: Icon(
                        Icons.close,
                        size: 20,
                        color: Colors.white.withValues(alpha: 0.5),
                      ),
                    ),
                ],
              ),
              if (widget.message != null) ...[
                const SizedBox(height: 16),
                Text(
                  widget.message!,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w400,
                    color: Colors.white.withValues(alpha: 0.7),
                    height: 1.5,
                  ),
                ),
              ],
              if (widget.actionLabel != null) ...[
                const SizedBox(height: 16),
                GestureDetector(
                  onTap: () {
                    HapticFeedback.lightImpact();
                    widget.onAction?.call();
                  },
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 12,
                    ),
                    decoration: BoxDecoration(
                      color: accentColor,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      widget.actionLabel!,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
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
// PREMIUM FEATURE CARD
// =============================================================================

/// Feature highlight card for onboarding
class PremiumFeatureCard extends StatefulWidget {
  final String title;
  final String description;
  final IconData icon;
  final Color? accentColor;
  final VoidCallback? onTap;
  final bool isNew;

  const PremiumFeatureCard({
    super.key,
    required this.title,
    required this.description,
    required this.icon,
    this.accentColor,
    this.onTap,
    this.isNew = false,
  });

  @override
  State<PremiumFeatureCard> createState() => _PremiumFeatureCardState();
}

class _PremiumFeatureCardState extends State<PremiumFeatureCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;

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

  @override
  Widget build(BuildContext context) {
    final accentColor = widget.accentColor ?? AppColors.primaryBlue;

    return GestureDetector(
      onTapDown: (_) => _controller.forward(),
      onTapUp: (_) {
        _controller.reverse();
        HapticFeedback.lightImpact();
        widget.onTap?.call();
      },
      onTapCancel: () => _controller.reverse(),
      child: AnimatedBuilder(
        animation: _scaleAnimation,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Colors.white.withValues(alpha: 0.08),
                        Colors.white.withValues(alpha: 0.04),
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.1),
                    ),
                  ),
                  child: Row(
                    children: [
                      // Icon container
                      Container(
                        width: 52,
                        height: 52,
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              accentColor.withValues(alpha: 0.3),
                              accentColor.withValues(alpha: 0.1),
                            ],
                          ),
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child: Icon(
                          widget.icon,
                          color: accentColor,
                          size: 26,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Text(
                                  widget.title,
                                  style: const TextStyle(
                                    fontSize: 15,
                                    fontWeight: FontWeight.w700,
                                    color: Colors.white,
                                  ),
                                ),
                                if (widget.isNew) ...[
                                  const SizedBox(width: 8),
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 8,
                                      vertical: 2,
                                    ),
                                    decoration: BoxDecoration(
                                      color:
                                          AppColors.successGreen.withValues(alpha: 0.2),
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Text(
                                      'NEW',
                                      style: TextStyle(
                                        fontSize: 9,
                                        fontWeight: FontWeight.w700,
                                        color: AppColors.successGreen,
                                        letterSpacing: 0.5,
                                      ),
                                    ),
                                  ),
                                ],
                              ],
                            ),
                            const SizedBox(height: 4),
                            Text(
                              widget.description,
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w400,
                                color: Colors.white.withValues(alpha: 0.6),
                                height: 1.4,
                              ),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ],
                        ),
                      ),
                      if (widget.onTap != null)
                        Icon(
                          Icons.chevron_right,
                          color: Colors.white.withValues(alpha: 0.4),
                          size: 24,
                        ),
                    ],
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

// =============================================================================
// PREMIUM SPLASH LOGO
// =============================================================================

/// Animated splash logo with effects
class PremiumSplashLogo extends StatefulWidget {
  final Widget logo;
  final String? title;
  final String? subtitle;
  final Color? accentColor;
  final VoidCallback? onAnimationComplete;
  final Duration duration;

  const PremiumSplashLogo({
    super.key,
    required this.logo,
    this.title,
    this.subtitle,
    this.accentColor,
    this.onAnimationComplete,
    this.duration = const Duration(milliseconds: 2000),
  });

  @override
  State<PremiumSplashLogo> createState() => _PremiumSplashLogoState();
}

class _PremiumSplashLogoState extends State<PremiumSplashLogo>
    with TickerProviderStateMixin {
  late AnimationController _logoController;
  late AnimationController _contentController;
  late Animation<double> _logoScale;
  late Animation<double> _logoFade;
  late Animation<double> _logoGlow;
  late Animation<double> _contentFade;
  late Animation<Offset> _contentSlide;

  @override
  void initState() {
    super.initState();

    // Logo animation
    _logoController = AnimationController(
      duration: Duration(milliseconds: widget.duration.inMilliseconds ~/ 2),
      vsync: this,
    );

    _logoScale = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _logoController, curve: Curves.easeOutBack),
    );

    _logoFade = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _logoController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeOut),
      ),
    );

    _logoGlow = Tween<double>(begin: 0.0, end: 0.6).animate(
      CurvedAnimation(
        parent: _logoController,
        curve: const Interval(0.3, 1.0, curve: Curves.easeOut),
      ),
    );

    // Content animation
    _contentController = AnimationController(
      duration: Duration(milliseconds: widget.duration.inMilliseconds ~/ 2),
      vsync: this,
    );

    _contentFade = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _contentController, curve: Curves.easeOut),
    );

    _contentSlide = Tween<Offset>(
      begin: const Offset(0, 0.2),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(parent: _contentController, curve: Curves.easeOutCubic),
    );

    // Start animations
    _startAnimations();
  }

  void _startAnimations() async {
    await _logoController.forward();
    await _contentController.forward();

    // Notify completion
    Timer(const Duration(milliseconds: 500), () {
      widget.onAnimationComplete?.call();
    });
  }

  @override
  void dispose() {
    _logoController.dispose();
    _contentController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final accentColor = widget.accentColor ?? AppColors.primaryBlue;

    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Animated logo
          AnimatedBuilder(
            animation: _logoController,
            builder: (context, child) {
              return Transform.scale(
                scale: _logoScale.value,
                child: Opacity(
                  opacity: _logoFade.value,
                  child: Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(28),
                      boxShadow: [
                        BoxShadow(
                          color: accentColor.withValues(alpha: _logoGlow.value),
                          blurRadius: 40,
                          spreadRadius: 8,
                        ),
                      ],
                    ),
                    child: widget.logo,
                  ),
                ),
              );
            },
          ),

          if (widget.title != null || widget.subtitle != null) ...[
            const SizedBox(height: 32),

            // Animated content
            SlideTransition(
              position: _contentSlide,
              child: FadeTransition(
                opacity: _contentFade,
                child: Column(
                  children: [
                    if (widget.title != null)
                      Text(
                        widget.title!,
                        style: const TextStyle(
                          fontSize: 36,
                          fontWeight: FontWeight.w300,
                          letterSpacing: 4,
                          color: Colors.white,
                        ),
                      ),
                    if (widget.subtitle != null) ...[
                      const SizedBox(height: 12),
                      Text(
                        widget.subtitle!,
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w400,
                          letterSpacing: 1.2,
                          color: Colors.white.withValues(alpha: 0.6),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// =============================================================================
// PREMIUM ANIMATED LOADING
// =============================================================================

/// Animated loading indicator for splash/onboarding
class PremiumAnimatedLoading extends StatefulWidget {
  final String? message;
  final Color? color;
  final double size;

  const PremiumAnimatedLoading({
    super.key,
    this.message,
    this.color,
    this.size = 48,
  });

  @override
  State<PremiumAnimatedLoading> createState() => _PremiumAnimatedLoadingState();
}

class _PremiumAnimatedLoadingState extends State<PremiumAnimatedLoading>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = widget.color ?? AppColors.primaryBlue;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: widget.size,
          height: widget.size,
          child: AnimatedBuilder(
            animation: _controller,
            builder: (context, child) {
              return CustomPaint(
                painter: _LoadingPainter(
                  progress: _controller.value,
                  color: color,
                ),
              );
            },
          ),
        ),
        if (widget.message != null) ...[
          const SizedBox(height: 16),
          Text(
            widget.message!,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: Colors.white.withValues(alpha: 0.6),
              letterSpacing: 0.5,
            ),
          ),
        ],
      ],
    );
  }
}

/// Custom painter for loading animation
class _LoadingPainter extends CustomPainter {
  final double progress;
  final Color color;

  _LoadingPainter({
    required this.progress,
    required this.color,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2 - 4;

    // Background circle
    final bgPaint = Paint()
      ..color = color.withValues(alpha: 0.2)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4;

    canvas.drawCircle(center, radius, bgPaint);

    // Animated arc
    final arcPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;

    final startAngle = progress * 2 * math.pi;
    final sweepAngle = 0.8 * math.pi;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle - math.pi / 2,
      sweepAngle,
      false,
      arcPaint,
    );

    // Glow effect
    final glowPaint = Paint()
      ..color = color.withValues(alpha: 0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 8
      ..strokeCap = StrokeCap.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4);

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle - math.pi / 2,
      sweepAngle,
      false,
      glowPaint,
    );
  }

  @override
  bool shouldRepaint(covariant _LoadingPainter oldDelegate) {
    return progress != oldDelegate.progress;
  }
}

// =============================================================================
// PREMIUM CHECKLIST
// =============================================================================

/// Checklist item for onboarding tasks
class ChecklistItem {
  final String title;
  final String? description;
  final bool isCompleted;

  const ChecklistItem({
    required this.title,
    this.description,
    this.isCompleted = false,
  });
}

/// Premium checklist widget for onboarding tasks
class PremiumChecklist extends StatelessWidget {
  final String? title;
  final List<ChecklistItem> items;
  final Color? accentColor;
  final ValueChanged<int>? onItemTap;

  const PremiumChecklist({
    super.key,
    this.title,
    required this.items,
    this.accentColor,
    this.onItemTap,
  });

  @override
  Widget build(BuildContext context) {
    final accent = accentColor ?? AppColors.primaryBlue;
    final completedCount = items.where((i) => i.isCompleted).length;

    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                Colors.white.withValues(alpha: 0.08),
                Colors.white.withValues(alpha: 0.04),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (title != null) ...[
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      title!,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: accent.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '$completedCount/${items.length}',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: accent,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                // Progress bar
                Container(
                  height: 4,
                  margin: const EdgeInsets.only(top: 8, bottom: 16),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(2),
                  ),
                  child: FractionallySizedBox(
                    alignment: Alignment.centerLeft,
                    widthFactor: items.isEmpty ? 0 : completedCount / items.length,
                    child: Container(
                      decoration: BoxDecoration(
                        color: accent,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                ),
              ],
              ...items.asMap().entries.map((entry) {
                final index = entry.key;
                final item = entry.value;

                return GestureDetector(
                  onTap: onItemTap != null
                      ? () {
                          HapticFeedback.lightImpact();
                          onItemTap!(index);
                        }
                      : null,
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Row(
                      children: [
                        AnimatedContainer(
                          duration: const Duration(milliseconds: 300),
                          width: 28,
                          height: 28,
                          decoration: BoxDecoration(
                            color: item.isCompleted
                                ? AppColors.successGreen
                                : Colors.white.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(8),
                            border: item.isCompleted
                                ? null
                                : Border.all(
                                    color: Colors.white.withValues(alpha: 0.2),
                                  ),
                          ),
                          child: item.isCompleted
                              ? const Icon(
                                  Icons.check,
                                  size: 16,
                                  color: Colors.white,
                                )
                              : null,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                item.title,
                                style: TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                  color: item.isCompleted
                                      ? Colors.white.withValues(alpha: 0.5)
                                      : Colors.white,
                                  decoration: item.isCompleted
                                      ? TextDecoration.lineThrough
                                      : null,
                                ),
                              ),
                              if (item.description != null) ...[
                                const SizedBox(height: 2),
                                Text(
                                  item.description!,
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w400,
                                    color: Colors.white.withValues(alpha: 0.4),
                                  ),
                                ),
                              ],
                            ],
                          ),
                        ),
                        if (!item.isCompleted && onItemTap != null)
                          Icon(
                            Icons.chevron_right,
                            size: 20,
                            color: Colors.white.withValues(alpha: 0.3),
                          ),
                      ],
                    ),
                  ),
                );
              }),
            ],
          ),
        ),
      ),
    );
  }
}
