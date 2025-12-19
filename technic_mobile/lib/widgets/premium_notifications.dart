/// Premium Notifications & Alerts
///
/// Premium notification and alert components with glass morphism design,
/// smooth animations, and professional styling.
library;

import 'dart:async';
import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../theme/app_colors.dart';

// =============================================================================
// NOTIFICATION TYPES
// =============================================================================

/// Notification type enumeration
enum NotificationType {
  info,
  success,
  warning,
  error,
  priceAlert,
  signal,
}

/// Notification data model
class NotificationData {
  final String id;
  final String title;
  final String message;
  final NotificationType type;
  final DateTime timestamp;
  final bool isRead;
  final String? ticker;
  final String? actionLabel;
  final VoidCallback? onAction;
  final Map<String, dynamic>? metadata;

  const NotificationData({
    required this.id,
    required this.title,
    required this.message,
    required this.type,
    required this.timestamp,
    this.isRead = false,
    this.ticker,
    this.actionLabel,
    this.onAction,
    this.metadata,
  });

  NotificationData copyWith({bool? isRead}) {
    return NotificationData(
      id: id,
      title: title,
      message: message,
      type: type,
      timestamp: timestamp,
      isRead: isRead ?? this.isRead,
      ticker: ticker,
      actionLabel: actionLabel,
      onAction: onAction,
      metadata: metadata,
    );
  }
}

// =============================================================================
// PREMIUM NOTIFICATION CARD
// =============================================================================

/// Premium notification card with glass morphism and animations.
class PremiumNotificationCard extends StatefulWidget {
  final NotificationData notification;
  final VoidCallback? onTap;
  final VoidCallback? onDismiss;
  final VoidCallback? onMarkRead;
  final bool showDismiss;
  final bool animate;

  const PremiumNotificationCard({
    super.key,
    required this.notification,
    this.onTap,
    this.onDismiss,
    this.onMarkRead,
    this.showDismiss = true,
    this.animate = true,
  });

  @override
  State<PremiumNotificationCard> createState() => _PremiumNotificationCardState();
}

class _PremiumNotificationCardState extends State<PremiumNotificationCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _fadeAnimation;
  // ignore: unused_field - Reserved for dismiss animation state
  bool _isDismissing = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 0.95, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutBack),
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    if (widget.animate) {
      _controller.forward();
    } else {
      _controller.value = 1.0;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  // ignore: unused_element - Reserved for swipe-to-dismiss feature
  Future<void> _handleDismiss() async {
    setState(() => _isDismissing = true);
    await _controller.reverse();
    widget.onDismiss?.call();
  }

  Color get _accentColor {
    switch (widget.notification.type) {
      case NotificationType.info:
        return AppColors.primaryBlue;
      case NotificationType.success:
        return AppColors.successGreen;
      case NotificationType.warning:
        return AppColors.warningOrange;
      case NotificationType.error:
        return AppColors.dangerRed;
      case NotificationType.priceAlert:
        return AppColors.warningOrange;
      case NotificationType.signal:
        return AppColors.primaryBlue;
    }
  }

  IconData get _icon {
    switch (widget.notification.type) {
      case NotificationType.info:
        return Icons.info_outline;
      case NotificationType.success:
        return Icons.check_circle_outline;
      case NotificationType.warning:
        return Icons.warning_amber_outlined;
      case NotificationType.error:
        return Icons.error_outline;
      case NotificationType.priceAlert:
        return Icons.notifications_active_outlined;
      case NotificationType.signal:
        return Icons.trending_up;
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return FadeTransition(
          opacity: _fadeAnimation,
          child: Transform.scale(
            scale: _scaleAnimation.value,
            child: child,
          ),
        );
      },
      child: Dismissible(
        key: Key(widget.notification.id),
        direction: widget.showDismiss
            ? DismissDirection.endToStart
            : DismissDirection.none,
        onDismissed: (_) => widget.onDismiss?.call(),
        background: _buildDismissBackground(),
        child: GestureDetector(
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
                  Colors.white.withValues(alpha: widget.notification.isRead ? 0.04 : 0.08),
                  Colors.white.withValues(alpha: widget.notification.isRead ? 0.02 : 0.04),
                ],
              ),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: widget.notification.isRead
                    ? Colors.white.withValues(alpha: 0.06)
                    : _accentColor.withValues(alpha: 0.3),
              ),
              boxShadow: widget.notification.isRead
                  ? null
                  : [
                      BoxShadow(
                        color: _accentColor.withValues(alpha: 0.1),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Icon
                      _buildIcon(),
                      const SizedBox(width: 14),
                      // Content
                      Expanded(child: _buildContent()),
                      // Unread indicator
                      if (!widget.notification.isRead)
                        _buildUnreadIndicator(),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildDismissBackground() {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            AppColors.dangerRed.withValues(alpha: 0.3),
            AppColors.dangerRed.withValues(alpha: 0.1),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      alignment: Alignment.centerRight,
      padding: const EdgeInsets.only(right: 20),
      child: Icon(
        Icons.delete_outline,
        color: AppColors.dangerRed,
        size: 28,
      ),
    );
  }

  Widget _buildIcon() {
    return Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            _accentColor.withValues(alpha: 0.25),
            _accentColor.withValues(alpha: 0.1),
          ],
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _accentColor.withValues(alpha: 0.3)),
      ),
      child: Icon(
        _icon,
        color: _accentColor,
        size: 22,
      ),
    );
  }

  Widget _buildContent() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Title row
        Row(
          children: [
            Expanded(
              child: Text(
                widget.notification.title,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: widget.notification.isRead ? FontWeight.w500 : FontWeight.w700,
                ),
              ),
            ),
            if (widget.notification.ticker != null) ...[
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: _accentColor.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: _accentColor.withValues(alpha: 0.3)),
                ),
                child: Text(
                  widget.notification.ticker!,
                  style: TextStyle(
                    color: _accentColor,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ],
        ),
        const SizedBox(height: 6),
        // Message
        Text(
          widget.notification.message,
          style: TextStyle(
            color: Colors.white.withValues(alpha: widget.notification.isRead ? 0.5 : 0.7),
            fontSize: 13,
            height: 1.4,
          ),
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        ),
        const SizedBox(height: 10),
        // Footer
        Row(
          children: [
            // Timestamp
            Icon(
              Icons.access_time,
              size: 12,
              color: Colors.white.withValues(alpha: 0.4),
            ),
            const SizedBox(width: 4),
            Text(
              _formatTime(widget.notification.timestamp),
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.4),
                fontSize: 11,
              ),
            ),
            const Spacer(),
            // Action button
            if (widget.notification.actionLabel != null)
              _buildActionButton(),
          ],
        ),
      ],
    );
  }

  Widget _buildActionButton() {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        widget.notification.onAction?.call();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              _accentColor.withValues(alpha: 0.2),
              _accentColor.withValues(alpha: 0.1),
            ],
          ),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: _accentColor.withValues(alpha: 0.3)),
        ),
        child: Text(
          widget.notification.actionLabel!,
          style: TextStyle(
            color: _accentColor,
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }

  Widget _buildUnreadIndicator() {
    return Container(
      width: 10,
      height: 10,
      decoration: BoxDecoration(
        color: _accentColor,
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: _accentColor.withValues(alpha: 0.5),
            blurRadius: 6,
          ),
        ],
      ),
    );
  }

  String _formatTime(DateTime time) {
    final now = DateTime.now();
    final diff = now.difference(time);

    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${time.month}/${time.day}';
  }
}

// =============================================================================
// PREMIUM ALERT BANNER
// =============================================================================

/// Premium alert banner for top/bottom display.
class PremiumAlertBanner extends StatefulWidget {
  final String message;
  final String? title;
  final NotificationType type;
  final VoidCallback? onDismiss;
  final VoidCallback? onAction;
  final String? actionLabel;
  final bool showIcon;
  final bool autoDismiss;
  final Duration autoDismissDuration;

  const PremiumAlertBanner({
    super.key,
    required this.message,
    this.title,
    this.type = NotificationType.info,
    this.onDismiss,
    this.onAction,
    this.actionLabel,
    this.showIcon = true,
    this.autoDismiss = false,
    this.autoDismissDuration = const Duration(seconds: 5),
  });

  /// Show banner as overlay
  static OverlayEntry show(
    BuildContext context, {
    required String message,
    String? title,
    NotificationType type = NotificationType.info,
    VoidCallback? onAction,
    String? actionLabel,
    bool autoDismiss = true,
    Duration autoDismissDuration = const Duration(seconds: 4),
  }) {
    late OverlayEntry entry;
    entry = OverlayEntry(
      builder: (context) => Positioned(
        top: MediaQuery.of(context).padding.top + 10,
        left: 16,
        right: 16,
        child: Material(
          color: Colors.transparent,
          child: PremiumAlertBanner(
            message: message,
            title: title,
            type: type,
            onAction: onAction,
            actionLabel: actionLabel,
            autoDismiss: autoDismiss,
            autoDismissDuration: autoDismissDuration,
            onDismiss: () => entry.remove(),
          ),
        ),
      ),
    );
    Overlay.of(context).insert(entry);
    return entry;
  }

  @override
  State<PremiumAlertBanner> createState() => _PremiumAlertBannerState();
}

class _PremiumAlertBannerState extends State<PremiumAlertBanner>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _fadeAnimation;
  Timer? _dismissTimer;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );

    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, -1),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic));

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    _controller.forward();

    if (widget.autoDismiss) {
      _dismissTimer = Timer(widget.autoDismissDuration, _dismiss);
    }
  }

  @override
  void dispose() {
    _dismissTimer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  Future<void> _dismiss() async {
    await _controller.reverse();
    widget.onDismiss?.call();
  }

  Color get _accentColor {
    switch (widget.type) {
      case NotificationType.info:
        return AppColors.primaryBlue;
      case NotificationType.success:
        return AppColors.successGreen;
      case NotificationType.warning:
        return AppColors.warningOrange;
      case NotificationType.error:
        return AppColors.dangerRed;
      case NotificationType.priceAlert:
        return AppColors.warningOrange;
      case NotificationType.signal:
        return AppColors.primaryBlue;
    }
  }

  IconData get _icon {
    switch (widget.type) {
      case NotificationType.info:
        return Icons.info_outline;
      case NotificationType.success:
        return Icons.check_circle_outline;
      case NotificationType.warning:
        return Icons.warning_amber_outlined;
      case NotificationType.error:
        return Icons.error_outline;
      case NotificationType.priceAlert:
        return Icons.notifications_active_outlined;
      case NotificationType.signal:
        return Icons.trending_up;
    }
  }

  @override
  Widget build(BuildContext context) {
    return SlideTransition(
      position: _slideAnimation,
      child: FadeTransition(
        opacity: _fadeAnimation,
        child: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                _accentColor.withValues(alpha: 0.2),
                _accentColor.withValues(alpha: 0.1),
              ],
            ),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: _accentColor.withValues(alpha: 0.4)),
            boxShadow: [
              BoxShadow(
                color: _accentColor.withValues(alpha: 0.2),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.3),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    // Icon
                    if (widget.showIcon) ...[
                      Container(
                        width: 40,
                        height: 40,
                        decoration: BoxDecoration(
                          color: _accentColor.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Icon(_icon, color: _accentColor, size: 22),
                      ),
                      const SizedBox(width: 14),
                    ],
                    // Content
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          if (widget.title != null) ...[
                            Text(
                              widget.title!,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 15,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 4),
                          ],
                          Text(
                            widget.message,
                            style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.85),
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    ),
                    // Actions
                    if (widget.actionLabel != null) ...[
                      const SizedBox(width: 12),
                      GestureDetector(
                        onTap: () {
                          HapticFeedback.lightImpact();
                          widget.onAction?.call();
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 8,
                          ),
                          decoration: BoxDecoration(
                            color: _accentColor,
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Text(
                            widget.actionLabel!,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ],
                    // Close button
                    const SizedBox(width: 8),
                    GestureDetector(
                      onTap: () {
                        HapticFeedback.lightImpact();
                        _dismiss();
                      },
                      child: Container(
                        padding: const EdgeInsets.all(6),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Icon(
                          Icons.close,
                          color: Colors.white.withValues(alpha: 0.7),
                          size: 18,
                        ),
                      ),
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
}

// =============================================================================
// PREMIUM TOAST
// =============================================================================

/// Premium toast notification.
class PremiumToast extends StatefulWidget {
  final String message;
  final NotificationType type;
  final Duration duration;
  final VoidCallback? onDismiss;

  const PremiumToast({
    super.key,
    required this.message,
    this.type = NotificationType.info,
    this.duration = const Duration(seconds: 3),
    this.onDismiss,
  });

  /// Show toast
  static OverlayEntry show(
    BuildContext context, {
    required String message,
    NotificationType type = NotificationType.info,
    Duration duration = const Duration(seconds: 3),
  }) {
    late OverlayEntry entry;
    entry = OverlayEntry(
      builder: (context) => Positioned(
        bottom: MediaQuery.of(context).padding.bottom + 80,
        left: 40,
        right: 40,
        child: Material(
          color: Colors.transparent,
          child: PremiumToast(
            message: message,
            type: type,
            duration: duration,
            onDismiss: () => entry.remove(),
          ),
        ),
      ),
    );
    Overlay.of(context).insert(entry);
    return entry;
  }

  @override
  State<PremiumToast> createState() => _PremiumToastState();
}

class _PremiumToastState extends State<PremiumToast>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _fadeAnimation;

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

    _controller.forward();

    Future.delayed(widget.duration, () {
      if (mounted) {
        _controller.reverse().then((_) {
          widget.onDismiss?.call();
        });
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Color get _accentColor {
    switch (widget.type) {
      case NotificationType.info:
        return AppColors.primaryBlue;
      case NotificationType.success:
        return AppColors.successGreen;
      case NotificationType.warning:
        return AppColors.warningOrange;
      case NotificationType.error:
        return AppColors.dangerRed;
      default:
        return AppColors.primaryBlue;
    }
  }

  IconData get _icon {
    switch (widget.type) {
      case NotificationType.info:
        return Icons.info_outline;
      case NotificationType.success:
        return Icons.check_circle_outline;
      case NotificationType.warning:
        return Icons.warning_amber_outlined;
      case NotificationType.error:
        return Icons.error_outline;
      default:
        return Icons.info_outline;
    }
  }

  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _scaleAnimation,
      child: FadeTransition(
        opacity: _fadeAnimation,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                _accentColor.withValues(alpha: 0.9),
                _accentColor.withValues(alpha: 0.7),
              ],
            ),
            borderRadius: BorderRadius.circular(30),
            boxShadow: [
              BoxShadow(
                color: _accentColor.withValues(alpha: 0.4),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(_icon, color: Colors.white, size: 20),
              const SizedBox(width: 10),
              Flexible(
                child: Text(
                  widget.message,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM BADGE
// =============================================================================

/// Premium animated badge for icons.
class PremiumBadge extends StatefulWidget {
  final Widget child;
  final int? count;
  final bool showBadge;
  final Color? color;
  final bool animate;
  final double size;
  final Offset offset;

  const PremiumBadge({
    super.key,
    required this.child,
    this.count,
    this.showBadge = true,
    this.color,
    this.animate = true,
    this.size = 18,
    this.offset = const Offset(0, 0),
  });

  @override
  State<PremiumBadge> createState() => _PremiumBadgeState();
}

class _PremiumBadgeState extends State<PremiumBadge>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.0, 0.3, curve: Curves.elasticOut),
      ),
    );

    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.3, 1.0, curve: Curves.easeInOut),
      ),
    );

    if (widget.animate && widget.showBadge) {
      _controller.forward();
      _controller.addStatusListener((status) {
        if (status == AnimationStatus.completed) {
          _controller.reverse();
        } else if (status == AnimationStatus.dismissed) {
          Future.delayed(const Duration(seconds: 2), () {
            if (mounted && widget.showBadge) {
              _controller.forward();
            }
          });
        }
      });
    }
  }

  @override
  void didUpdateWidget(PremiumBadge oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.showBadge && !oldWidget.showBadge) {
      _controller.forward(from: 0);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final badgeColor = widget.color ?? AppColors.dangerRed;

    return Stack(
      clipBehavior: Clip.none,
      children: [
        widget.child,
        if (widget.showBadge)
          Positioned(
            right: -4 + widget.offset.dx,
            top: -4 + widget.offset.dy,
            child: AnimatedBuilder(
              animation: _controller,
              builder: (context, child) {
                return Transform.scale(
                  scale: widget.animate
                      ? _scaleAnimation.value * _pulseAnimation.value
                      : 1.0,
                  child: child,
                );
              },
              child: Container(
                constraints: BoxConstraints(
                  minWidth: widget.size,
                  minHeight: widget.size,
                ),
                padding: EdgeInsets.symmetric(
                  horizontal: widget.count != null ? 5 : 0,
                ),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      badgeColor,
                      badgeColor.withValues(alpha: 0.8),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(widget.size / 2),
                  border: Border.all(
                    color: AppColors.darkBackground,
                    width: 2,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: badgeColor.withValues(alpha: 0.5),
                      blurRadius: 8,
                    ),
                  ],
                ),
                child: widget.count != null
                    ? Center(
                        child: Text(
                          widget.count! > 99 ? '99+' : widget.count.toString(),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      )
                    : null,
              ),
            ),
          ),
      ],
    );
  }
}

// =============================================================================
// PREMIUM NOTIFICATION BELL
// =============================================================================

/// Premium animated notification bell.
class PremiumNotificationBell extends StatefulWidget {
  final int unreadCount;
  final VoidCallback onTap;
  final double size;
  final bool animate;

  const PremiumNotificationBell({
    super.key,
    this.unreadCount = 0,
    required this.onTap,
    this.size = 24,
    this.animate = true,
  });

  @override
  State<PremiumNotificationBell> createState() => _PremiumNotificationBellState();
}

class _PremiumNotificationBellState extends State<PremiumNotificationBell>
    with SingleTickerProviderStateMixin {
  late AnimationController _shakeController;
  late Animation<double> _shakeAnimation;

  @override
  void initState() {
    super.initState();
    _shakeController = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );

    _shakeAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _shakeController, curve: Curves.elasticIn),
    );

    if (widget.animate && widget.unreadCount > 0) {
      _startShakeAnimation();
    }
  }

  void _startShakeAnimation() {
    Future.delayed(const Duration(seconds: 3), () {
      if (mounted && widget.unreadCount > 0) {
        _shakeController.forward(from: 0).then((_) {
          _startShakeAnimation();
        });
      }
    });
  }

  @override
  void didUpdateWidget(PremiumNotificationBell oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.unreadCount > 0 && oldWidget.unreadCount == 0) {
      _shakeController.forward(from: 0);
      _startShakeAnimation();
    }
  }

  @override
  void dispose() {
    _shakeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        widget.onTap();
      },
      child: PremiumBadge(
        count: widget.unreadCount > 0 ? widget.unreadCount : null,
        showBadge: widget.unreadCount > 0,
        child: AnimatedBuilder(
          animation: _shakeAnimation,
          builder: (context, child) {
            return Transform.rotate(
              angle: math.sin(_shakeAnimation.value * math.pi * 4) * 0.15,
              child: child,
            );
          },
          child: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.06),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
            ),
            child: Icon(
              widget.unreadCount > 0
                  ? Icons.notifications_active
                  : Icons.notifications_outlined,
              color: widget.unreadCount > 0
                  ? AppColors.warningOrange
                  : Colors.white.withValues(alpha: 0.7),
              size: widget.size,
            ),
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM ALERT DIALOG
// =============================================================================

/// Premium alert dialog with glass morphism.
class PremiumAlertDialog extends StatelessWidget {
  final String title;
  final String message;
  final NotificationType type;
  final String? confirmLabel;
  final String? cancelLabel;
  final VoidCallback? onConfirm;
  final VoidCallback? onCancel;
  final bool showIcon;
  final Widget? customContent;

  const PremiumAlertDialog({
    super.key,
    required this.title,
    required this.message,
    this.type = NotificationType.info,
    this.confirmLabel,
    this.cancelLabel,
    this.onConfirm,
    this.onCancel,
    this.showIcon = true,
    this.customContent,
  });

  /// Show dialog
  static Future<bool?> show(
    BuildContext context, {
    required String title,
    required String message,
    NotificationType type = NotificationType.info,
    String? confirmLabel,
    String? cancelLabel,
    bool showIcon = true,
    Widget? customContent,
  }) {
    return showDialog<bool>(
      context: context,
      barrierColor: Colors.black54,
      builder: (context) => PremiumAlertDialog(
        title: title,
        message: message,
        type: type,
        confirmLabel: confirmLabel,
        cancelLabel: cancelLabel,
        showIcon: showIcon,
        customContent: customContent,
        onConfirm: () => Navigator.of(context).pop(true),
        onCancel: () => Navigator.of(context).pop(false),
      ),
    );
  }

  Color get _accentColor {
    switch (type) {
      case NotificationType.info:
        return AppColors.primaryBlue;
      case NotificationType.success:
        return AppColors.successGreen;
      case NotificationType.warning:
        return AppColors.warningOrange;
      case NotificationType.error:
        return AppColors.dangerRed;
      default:
        return AppColors.primaryBlue;
    }
  }

  IconData get _icon {
    switch (type) {
      case NotificationType.info:
        return Icons.info_outline;
      case NotificationType.success:
        return Icons.check_circle_outline;
      case NotificationType.warning:
        return Icons.warning_amber_outlined;
      case NotificationType.error:
        return Icons.error_outline;
      default:
        return Icons.info_outline;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.transparent,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 340),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Colors.white.withValues(alpha: 0.12),
              Colors.white.withValues(alpha: 0.06),
            ],
          ),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: Colors.white.withValues(alpha: 0.15)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.3),
              blurRadius: 30,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(24),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Icon
                  if (showIcon)
                    Container(
                      width: 64,
                      height: 64,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            _accentColor.withValues(alpha: 0.25),
                            _accentColor.withValues(alpha: 0.1),
                          ],
                        ),
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: _accentColor.withValues(alpha: 0.3),
                        ),
                      ),
                      child: Icon(_icon, color: _accentColor, size: 32),
                    ),
                  if (showIcon) const SizedBox(height: 20),
                  // Title
                  Text(
                    title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 12),
                  // Message
                  Text(
                    message,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.7),
                      fontSize: 14,
                      height: 1.5,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  // Custom content
                  if (customContent != null) ...[
                    const SizedBox(height: 20),
                    customContent!,
                  ],
                  const SizedBox(height: 24),
                  // Buttons
                  Row(
                    children: [
                      if (cancelLabel != null)
                        Expanded(
                          child: _buildButton(
                            label: cancelLabel!,
                            onTap: onCancel,
                            isPrimary: false,
                          ),
                        ),
                      if (cancelLabel != null && confirmLabel != null)
                        const SizedBox(width: 12),
                      if (confirmLabel != null)
                        Expanded(
                          child: _buildButton(
                            label: confirmLabel!,
                            onTap: onConfirm,
                            isPrimary: true,
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

  Widget _buildButton({
    required String label,
    VoidCallback? onTap,
    required bool isPrimary,
  }) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap?.call();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14),
        decoration: BoxDecoration(
          gradient: isPrimary
              ? LinearGradient(
                  colors: [
                    _accentColor,
                    _accentColor.withValues(alpha: 0.8),
                  ],
                )
              : null,
          color: isPrimary ? null : Colors.white.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(14),
          border: isPrimary
              ? null
              : Border.all(color: Colors.white.withValues(alpha: 0.15)),
          boxShadow: isPrimary
              ? [
                  BoxShadow(
                    color: _accentColor.withValues(alpha: 0.3),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ]
              : null,
        ),
        child: Text(
          label,
          textAlign: TextAlign.center,
          style: TextStyle(
            color: isPrimary ? Colors.white : Colors.white.withValues(alpha: 0.8),
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM PRICE ALERT CARD
// =============================================================================

/// Premium price alert card.
class PremiumPriceAlertCard extends StatefulWidget {
  final String ticker;
  final String alertType;
  final double targetValue;
  final bool isActive;
  final bool isTriggered;
  final DateTime? triggeredAt;
  final String? note;
  final VoidCallback? onToggle;
  final VoidCallback? onDelete;
  final VoidCallback? onEdit;

  const PremiumPriceAlertCard({
    super.key,
    required this.ticker,
    required this.alertType,
    required this.targetValue,
    this.isActive = true,
    this.isTriggered = false,
    this.triggeredAt,
    this.note,
    this.onToggle,
    this.onDelete,
    this.onEdit,
  });

  @override
  State<PremiumPriceAlertCard> createState() => _PremiumPriceAlertCardState();
}

class _PremiumPriceAlertCardState extends State<PremiumPriceAlertCard>
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

  Color get _statusColor {
    if (widget.isTriggered) return AppColors.successGreen;
    if (!widget.isActive) return Colors.white.withValues(alpha: 0.3);
    return AppColors.warningOrange;
  }

  IconData get _alertIcon {
    if (widget.alertType.contains('Above')) return Icons.trending_up;
    if (widget.alertType.contains('Below')) return Icons.trending_down;
    return Icons.percent;
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _scaleAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: _scaleAnimation.value,
          child: child,
        );
      },
      child: GestureDetector(
        onTapDown: (_) => _controller.forward(),
        onTapUp: (_) => _controller.reverse(),
        onTapCancel: () => _controller.reverse(),
        child: Container(
          margin: const EdgeInsets.only(bottom: 12),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withValues(alpha: widget.isActive ? 0.08 : 0.04),
                Colors.white.withValues(alpha: widget.isActive ? 0.04 : 0.02),
              ],
            ),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: _statusColor.withValues(alpha: 0.3),
            ),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Header
                    Row(
                      children: [
                        // Alert icon
                        Container(
                          width: 44,
                          height: 44,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                _statusColor.withValues(alpha: 0.25),
                                _statusColor.withValues(alpha: 0.1),
                              ],
                            ),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: _statusColor.withValues(alpha: 0.3),
                            ),
                          ),
                          child: Icon(
                            _alertIcon,
                            color: _statusColor,
                            size: 22,
                          ),
                        ),
                        const SizedBox(width: 14),
                        // Ticker & Type
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Text(
                                    widget.ticker,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 16,
                                      fontWeight: FontWeight.w800,
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  if (widget.isTriggered)
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 8,
                                        vertical: 4,
                                      ),
                                      decoration: BoxDecoration(
                                        color: AppColors.successGreen.withValues(alpha: 0.2),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: const Text(
                                        'TRIGGERED',
                                        style: TextStyle(
                                          color: AppColors.successGreen,
                                          fontSize: 10,
                                          fontWeight: FontWeight.w700,
                                        ),
                                      ),
                                    ),
                                ],
                              ),
                              const SizedBox(height: 4),
                              Text(
                                widget.alertType,
                                style: TextStyle(
                                  color: Colors.white.withValues(alpha: 0.6),
                                  fontSize: 13,
                                ),
                              ),
                            ],
                          ),
                        ),
                        // Target Value
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              widget.alertType.contains('Percent')
                                  ? '${widget.targetValue.toStringAsFixed(1)}%'
                                  : '\$${widget.targetValue.toStringAsFixed(2)}',
                              style: TextStyle(
                                color: _statusColor,
                                fontSize: 18,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            Text(
                              'Target',
                              style: TextStyle(
                                color: Colors.white.withValues(alpha: 0.4),
                                fontSize: 11,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                    // Note
                    if (widget.note != null) ...[
                      const SizedBox(height: 12),
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.04),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.note_outlined,
                              size: 14,
                              color: Colors.white.withValues(alpha: 0.4),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                widget.note!,
                                style: TextStyle(
                                  color: Colors.white.withValues(alpha: 0.6),
                                  fontSize: 12,
                                  fontStyle: FontStyle.italic,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                    // Actions
                    const SizedBox(height: 14),
                    Row(
                      children: [
                        // Toggle
                        if (widget.onToggle != null && !widget.isTriggered)
                          _buildActionButton(
                            icon: widget.isActive
                                ? Icons.notifications_active
                                : Icons.notifications_off_outlined,
                            label: widget.isActive ? 'Active' : 'Paused',
                            color: widget.isActive
                                ? AppColors.successGreen
                                : Colors.white.withValues(alpha: 0.4),
                            onTap: widget.onToggle!,
                          ),
                        const Spacer(),
                        // Edit
                        if (widget.onEdit != null)
                          _buildIconButton(
                            icon: Icons.edit_outlined,
                            onTap: widget.onEdit!,
                          ),
                        if (widget.onEdit != null && widget.onDelete != null)
                          const SizedBox(width: 8),
                        // Delete
                        if (widget.onDelete != null)
                          _buildIconButton(
                            icon: Icons.delete_outline,
                            onTap: widget.onDelete!,
                            color: AppColors.dangerRed,
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

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: color.withValues(alpha: 0.3)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 16, color: color),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: color,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildIconButton({
    required IconData icon,
    required VoidCallback onTap,
    Color? color,
  }) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap();
      },
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: (color ?? Colors.white).withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: (color ?? Colors.white).withValues(alpha: 0.15),
          ),
        ),
        child: Icon(
          icon,
          size: 18,
          color: color ?? Colors.white.withValues(alpha: 0.6),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM NOTIFICATION EMPTY STATE
// =============================================================================

/// Premium empty state for notifications.
class PremiumNotificationEmptyState extends StatelessWidget {
  final VoidCallback? onEnableNotifications;

  const PremiumNotificationEmptyState({
    super.key,
    this.onEnableNotifications,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    AppColors.primaryBlue.withValues(alpha: 0.2),
                    AppColors.primaryBlue.withValues(alpha: 0.1),
                  ],
                ),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.notifications_none,
                size: 48,
                color: AppColors.primaryBlue.withValues(alpha: 0.6),
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'No Notifications',
              style: TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'You\'re all caught up! We\'ll notify you when there\'s something new.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.6),
                fontSize: 14,
                height: 1.5,
              ),
            ),
            if (onEnableNotifications != null) ...[
              const SizedBox(height: 24),
              GestureDetector(
                onTap: () {
                  HapticFeedback.lightImpact();
                  onEnableNotifications?.call();
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 14,
                  ),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        AppColors.primaryBlue,
                        AppColors.primaryBlue.withValues(alpha: 0.8),
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
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.notifications_active, color: Colors.white, size: 20),
                      SizedBox(width: 8),
                      Text(
                        'Enable Notifications',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
