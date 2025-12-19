/// Premium Modals & Sheets Widgets
///
/// A collection of premium modal and sheet components with glass morphism,
/// smooth animations, and professional styling.
///
/// Components:
/// - PremiumBottomSheet: Base bottom sheet with glass morphism
/// - PremiumDialog: Alert dialog with premium styling
/// - PremiumActionSheet: iOS-style action sheet
/// - PremiumConfirmationSheet: Confirmation with destructive option
/// - PremiumInputDialog: Dialog with text input
/// - PremiumMenuSheet: Menu-style sheet with options
/// - PremiumFullScreenModal: Full screen modal overlay
/// - PremiumImagePreview: Image preview modal
/// - PremiumShareSheet: Share options sheet
/// - PremiumPickerSheet: Selection picker sheet
library;

import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_colors.dart';
import '../utils/helpers.dart';

// =============================================================================
// PREMIUM BOTTOM SHEET
// =============================================================================

/// Premium bottom sheet with glass morphism
class PremiumBottomSheet extends StatefulWidget {
  final Widget child;
  final String? title;
  final IconData? icon;
  final Color? iconColor;
  final bool showDragHandle;
  final bool showCloseButton;
  final VoidCallback? onClose;
  final double? maxHeight;
  final EdgeInsets? padding;

  const PremiumBottomSheet({
    super.key,
    required this.child,
    this.title,
    this.icon,
    this.iconColor,
    this.showDragHandle = true,
    this.showCloseButton = true,
    this.onClose,
    this.maxHeight,
    this.padding,
  });

  /// Show premium bottom sheet
  static Future<T?> show<T>({
    required BuildContext context,
    required Widget child,
    String? title,
    IconData? icon,
    Color? iconColor,
    bool showDragHandle = true,
    bool showCloseButton = true,
    double? maxHeight,
    EdgeInsets? padding,
    bool isDismissible = true,
    bool enableDrag = true,
  }) {
    return showModalBottomSheet<T>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      isDismissible: isDismissible,
      enableDrag: enableDrag,
      builder: (context) => PremiumBottomSheet(
        title: title,
        icon: icon,
        iconColor: iconColor,
        showDragHandle: showDragHandle,
        showCloseButton: showCloseButton,
        maxHeight: maxHeight,
        padding: padding,
        onClose: () => Navigator.pop(context),
        child: child,
      ),
    );
  }

  @override
  State<PremiumBottomSheet> createState() => _PremiumBottomSheetState();
}

class _PremiumBottomSheetState extends State<PremiumBottomSheet>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _slideAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _slideAnimation = Tween<double>(begin: 0.1, end: 0.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic),
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
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
    final maxHeight = widget.maxHeight ?? MediaQuery.of(context).size.height * 0.85;

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return FadeTransition(
          opacity: _fadeAnimation,
          child: SlideTransition(
            position: Tween<Offset>(
              begin: Offset(0, _slideAnimation.value),
              end: Offset.zero,
            ).animate(_controller),
            child: Container(
              constraints: BoxConstraints(maxHeight: maxHeight),
              decoration: BoxDecoration(
                borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.3),
                    blurRadius: 20,
                    offset: const Offset(0, -5),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                  child: Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [
                          tone(AppColors.darkBackground, 0.95),
                          tone(AppColors.darkBackground, 0.98),
                        ],
                      ),
                      borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.1),
                        width: 1,
                      ),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        // Drag handle
                        if (widget.showDragHandle)
                          Padding(
                            padding: const EdgeInsets.only(top: 12),
                            child: Container(
                              width: 40,
                              height: 4,
                              decoration: BoxDecoration(
                                color: Colors.white.withValues(alpha: 0.3),
                                borderRadius: BorderRadius.circular(2),
                              ),
                            ),
                          ),

                        // Header
                        if (widget.title != null || widget.showCloseButton)
                          _buildHeader(),

                        // Content
                        Flexible(
                          child: SingleChildScrollView(
                            padding: widget.padding ?? const EdgeInsets.all(20),
                            child: widget.child,
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
      },
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: EdgeInsets.fromLTRB(
        20,
        widget.showDragHandle ? 16 : 20,
        widget.showCloseButton ? 8 : 20,
        16,
      ),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withValues(alpha: 0.1),
            width: 1,
          ),
        ),
      ),
      child: Row(
        children: [
          if (widget.icon != null) ...[
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    (widget.iconColor ?? AppColors.primaryBlue),
                    (widget.iconColor ?? AppColors.primaryBlue).withValues(alpha: 0.6),
                  ],
                ),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                widget.icon,
                size: 20,
                color: Colors.white,
              ),
            ),
            const SizedBox(width: 14),
          ],
          if (widget.title != null)
            Expanded(
              child: Text(
                widget.title!,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                ),
              ),
            ),
          if (widget.showCloseButton)
            IconButton(
              onPressed: () {
                HapticFeedback.lightImpact();
                widget.onClose?.call();
              },
              icon: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.close,
                  size: 18,
                  color: Colors.white70,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// =============================================================================
// PREMIUM DIALOG
// =============================================================================

/// Dialog type for styling
enum DialogType {
  info,
  success,
  warning,
  error,
}

/// Premium dialog with glass morphism
class PremiumDialog extends StatefulWidget {
  final String title;
  final String? message;
  final Widget? content;
  final DialogType type;
  final IconData? icon;
  final String? confirmLabel;
  final String? cancelLabel;
  final VoidCallback? onConfirm;
  final VoidCallback? onCancel;
  final bool isDangerous;
  final bool showCancel;

  const PremiumDialog({
    super.key,
    required this.title,
    this.message,
    this.content,
    this.type = DialogType.info,
    this.icon,
    this.confirmLabel,
    this.cancelLabel,
    this.onConfirm,
    this.onCancel,
    this.isDangerous = false,
    this.showCancel = true,
  });

  /// Show premium dialog
  static Future<bool?> show({
    required BuildContext context,
    required String title,
    String? message,
    Widget? content,
    DialogType type = DialogType.info,
    IconData? icon,
    String? confirmLabel,
    String? cancelLabel,
    VoidCallback? onConfirm,
    VoidCallback? onCancel,
    bool isDangerous = false,
    bool showCancel = true,
    bool barrierDismissible = true,
  }) {
    return showDialog<bool>(
      context: context,
      barrierDismissible: barrierDismissible,
      barrierColor: Colors.black.withValues(alpha: 0.7),
      builder: (context) => PremiumDialog(
        title: title,
        message: message,
        content: content,
        type: type,
        icon: icon,
        confirmLabel: confirmLabel,
        cancelLabel: cancelLabel,
        onConfirm: onConfirm,
        onCancel: onCancel,
        isDangerous: isDangerous,
        showCancel: showCancel,
      ),
    );
  }

  @override
  State<PremiumDialog> createState() => _PremiumDialogState();
}

class _PremiumDialogState extends State<PremiumDialog>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 250),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 0.9, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutBack),
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Color _getTypeColor() {
    switch (widget.type) {
      case DialogType.info:
        return AppColors.primaryBlue;
      case DialogType.success:
        return AppColors.successGreen;
      case DialogType.warning:
        return AppColors.warningOrange;
      case DialogType.error:
        return AppColors.dangerRed;
    }
  }

  IconData _getTypeIcon() {
    if (widget.icon != null) return widget.icon!;
    switch (widget.type) {
      case DialogType.info:
        return Icons.info_outline;
      case DialogType.success:
        return Icons.check_circle_outline;
      case DialogType.warning:
        return Icons.warning_amber_outlined;
      case DialogType.error:
        return Icons.error_outline;
    }
  }

  @override
  Widget build(BuildContext context) {
    final typeColor = _getTypeColor();
    final buttonColor = widget.isDangerous ? AppColors.dangerRed : typeColor;

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return FadeTransition(
          opacity: _fadeAnimation,
          child: ScaleTransition(
            scale: _scaleAnimation,
            child: Dialog(
              backgroundColor: Colors.transparent,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                  child: Container(
                    constraints: const BoxConstraints(maxWidth: 340),
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [
                          tone(AppColors.darkBackground, 0.95),
                          tone(AppColors.darkBackground, 0.98),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.1),
                      ),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        // Icon
                        Container(
                          width: 64,
                          height: 64,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                typeColor.withValues(alpha: 0.3),
                                typeColor.withValues(alpha: 0.1),
                              ],
                            ),
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: typeColor.withValues(alpha: 0.3),
                              width: 2,
                            ),
                          ),
                          child: Icon(
                            _getTypeIcon(),
                            size: 32,
                            color: typeColor,
                          ),
                        ),
                        const SizedBox(height: 20),

                        // Title
                        Text(
                          widget.title,
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                          textAlign: TextAlign.center,
                        ),

                        // Message
                        if (widget.message != null) ...[
                          const SizedBox(height: 12),
                          Text(
                            widget.message!,
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w400,
                              color: Colors.white.withValues(alpha: 0.7),
                              height: 1.5,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ],

                        // Custom content
                        if (widget.content != null) ...[
                          const SizedBox(height: 20),
                          widget.content!,
                        ],

                        const SizedBox(height: 24),

                        // Buttons
                        Row(
                          children: [
                            if (widget.showCancel) ...[
                              Expanded(
                                child: _PremiumDialogButton(
                                  label: widget.cancelLabel ?? 'Cancel',
                                  onTap: () {
                                    widget.onCancel?.call();
                                    Navigator.pop(context, false);
                                  },
                                  isPrimary: false,
                                ),
                              ),
                              const SizedBox(width: 12),
                            ],
                            Expanded(
                              child: _PremiumDialogButton(
                                label: widget.confirmLabel ?? 'OK',
                                onTap: () {
                                  widget.onConfirm?.call();
                                  Navigator.pop(context, true);
                                },
                                isPrimary: true,
                                color: buttonColor,
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
          ),
        );
      },
    );
  }
}

/// Dialog button widget
class _PremiumDialogButton extends StatefulWidget {
  final String label;
  final VoidCallback onTap;
  final bool isPrimary;
  final Color? color;

  const _PremiumDialogButton({
    required this.label,
    required this.onTap,
    this.isPrimary = false,
    this.color,
  });

  @override
  State<_PremiumDialogButton> createState() => _PremiumDialogButtonState();
}

class _PremiumDialogButtonState extends State<_PremiumDialogButton>
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
    _scaleAnimation = Tween<double>(begin: 1.0, end: 0.95).animate(
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
    final color = widget.color ?? AppColors.primaryBlue;

    return GestureDetector(
      onTapDown: (_) => _controller.forward(),
      onTapUp: (_) {
        _controller.reverse();
        HapticFeedback.lightImpact();
        widget.onTap();
      },
      onTapCancel: () => _controller.reverse(),
      child: AnimatedBuilder(
        animation: _scaleAnimation,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(
                gradient: widget.isPrimary
                    ? LinearGradient(
                        colors: [color, color.withValues(alpha: 0.8)],
                      )
                    : null,
                color: widget.isPrimary
                    ? null
                    : Colors.white.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(14),
                border: widget.isPrimary
                    ? null
                    : Border.all(
                        color: Colors.white.withValues(alpha: 0.15),
                      ),
                boxShadow: widget.isPrimary
                    ? [
                        BoxShadow(
                          color: color.withValues(alpha: 0.3),
                          blurRadius: 12,
                          offset: const Offset(0, 4),
                        ),
                      ]
                    : null,
              ),
              child: Center(
                child: Text(
                  widget.label,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: widget.isPrimary ? Colors.white : Colors.white70,
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
// PREMIUM ACTION SHEET
// =============================================================================

/// Action sheet item data
class ActionSheetItem {
  final String label;
  final IconData icon;
  final VoidCallback onTap;
  final Color? color;
  final bool isDestructive;
  final bool isDisabled;

  const ActionSheetItem({
    required this.label,
    required this.icon,
    required this.onTap,
    this.color,
    this.isDestructive = false,
    this.isDisabled = false,
  });
}

/// Premium iOS-style action sheet
class PremiumActionSheet extends StatefulWidget {
  final String? title;
  final String? message;
  final List<ActionSheetItem> actions;
  final String cancelLabel;
  final VoidCallback? onCancel;

  const PremiumActionSheet({
    super.key,
    this.title,
    this.message,
    required this.actions,
    this.cancelLabel = 'Cancel',
    this.onCancel,
  });

  /// Show premium action sheet
  static Future<void> show({
    required BuildContext context,
    String? title,
    String? message,
    required List<ActionSheetItem> actions,
    String cancelLabel = 'Cancel',
    VoidCallback? onCancel,
  }) {
    return showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => PremiumActionSheet(
        title: title,
        message: message,
        actions: actions,
        cancelLabel: cancelLabel,
        onCancel: onCancel,
      ),
    );
  }

  @override
  State<PremiumActionSheet> createState() => _PremiumActionSheetState();
}

class _PremiumActionSheetState extends State<PremiumActionSheet>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _slideAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _slideAnimation = Tween<double>(begin: 1.0, end: 0.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic),
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
    final bottomPadding = MediaQuery.of(context).padding.bottom;

    return AnimatedBuilder(
      animation: _slideAnimation,
      builder: (context, child) {
        return Transform.translate(
          offset: Offset(0, _slideAnimation.value * 200),
          child: Padding(
            padding: EdgeInsets.fromLTRB(12, 0, 12, 12 + bottomPadding),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Actions container
                ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: BackdropFilter(
                    filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                    child: Container(
                      decoration: BoxDecoration(
                        color: tone(AppColors.darkBackground, 0.95),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(
                          color: Colors.white.withValues(alpha: 0.1),
                        ),
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          // Title/Message
                          if (widget.title != null || widget.message != null)
                            Container(
                              padding: const EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                border: Border(
                                  bottom: BorderSide(
                                    color: Colors.white.withValues(alpha: 0.1),
                                  ),
                                ),
                              ),
                              child: Column(
                                children: [
                                  if (widget.title != null)
                                    Text(
                                      widget.title!,
                                      style: TextStyle(
                                        fontSize: 13,
                                        fontWeight: FontWeight.w600,
                                        color: Colors.white.withValues(alpha: 0.6),
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                  if (widget.message != null) ...[
                                    const SizedBox(height: 4),
                                    Text(
                                      widget.message!,
                                      style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w400,
                                        color: Colors.white.withValues(alpha: 0.4),
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                  ],
                                ],
                              ),
                            ),

                          // Actions
                          ...widget.actions.asMap().entries.map((entry) {
                            final index = entry.key;
                            final action = entry.value;
                            final isLast = index == widget.actions.length - 1;

                            return _buildActionItem(action, isLast);
                          }),
                        ],
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 8),

                // Cancel button
                ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: BackdropFilter(
                    filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                    child: GestureDetector(
                      onTap: () {
                        HapticFeedback.lightImpact();
                        widget.onCancel?.call();
                        Navigator.pop(context);
                      },
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        decoration: BoxDecoration(
                          color: tone(AppColors.darkBackground, 0.95),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.1),
                          ),
                        ),
                        child: Text(
                          widget.cancelLabel,
                          style: TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.w600,
                            color: AppColors.primaryBlue,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildActionItem(ActionSheetItem action, bool isLast) {
    final color = action.isDestructive
        ? AppColors.dangerRed
        : action.color ?? Colors.white;

    return GestureDetector(
      onTap: action.isDisabled
          ? null
          : () {
              HapticFeedback.lightImpact();
              Navigator.pop(context);
              action.onTap();
            },
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
        decoration: BoxDecoration(
          border: isLast
              ? null
              : Border(
                  bottom: BorderSide(
                    color: Colors.white.withValues(alpha: 0.1),
                  ),
                ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              action.icon,
              size: 22,
              color: action.isDisabled
                  ? Colors.white.withValues(alpha: 0.3)
                  : color,
            ),
            const SizedBox(width: 12),
            Text(
              action.label,
              style: TextStyle(
                fontSize: 17,
                fontWeight: FontWeight.w500,
                color: action.isDisabled
                    ? Colors.white.withValues(alpha: 0.3)
                    : color,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM CONFIRMATION SHEET
// =============================================================================

/// Premium confirmation bottom sheet
class PremiumConfirmationSheet extends StatelessWidget {
  final String title;
  final String message;
  final String confirmLabel;
  final String cancelLabel;
  final IconData icon;
  final bool isDestructive;
  final VoidCallback? onConfirm;
  final VoidCallback? onCancel;

  const PremiumConfirmationSheet({
    super.key,
    required this.title,
    required this.message,
    this.confirmLabel = 'Confirm',
    this.cancelLabel = 'Cancel',
    this.icon = Icons.help_outline,
    this.isDestructive = false,
    this.onConfirm,
    this.onCancel,
  });

  /// Show premium confirmation sheet
  static Future<bool?> show({
    required BuildContext context,
    required String title,
    required String message,
    String confirmLabel = 'Confirm',
    String cancelLabel = 'Cancel',
    IconData icon = Icons.help_outline,
    bool isDestructive = false,
    VoidCallback? onConfirm,
    VoidCallback? onCancel,
  }) {
    return showModalBottomSheet<bool>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => PremiumConfirmationSheet(
        title: title,
        message: message,
        confirmLabel: confirmLabel,
        cancelLabel: cancelLabel,
        icon: icon,
        isDestructive: isDestructive,
        onConfirm: onConfirm,
        onCancel: onCancel,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final color = isDestructive ? AppColors.dangerRed : AppColors.primaryBlue;

    return Container(
      margin: EdgeInsets.only(
        bottom: MediaQuery.of(context).padding.bottom,
      ),
      child: ClipRRect(
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  tone(AppColors.darkBackground, 0.95),
                  tone(AppColors.darkBackground, 0.98),
                ],
              ),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Drag handle
                Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.3),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: 24),

                // Icon
                Container(
                  width: 72,
                  height: 72,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        color.withValues(alpha: 0.25),
                        color.withValues(alpha: 0.1),
                      ],
                    ),
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: color.withValues(alpha: 0.3),
                      width: 2,
                    ),
                  ),
                  child: Icon(
                    icon,
                    size: 36,
                    color: color,
                  ),
                ),
                const SizedBox(height: 20),

                // Title
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 12),

                // Message
                Text(
                  message,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w400,
                    color: Colors.white.withValues(alpha: 0.7),
                    height: 1.5,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 28),

                // Buttons
                Row(
                  children: [
                    // Cancel
                    Expanded(
                      child: GestureDetector(
                        onTap: () {
                          HapticFeedback.lightImpact();
                          onCancel?.call();
                          Navigator.pop(context, false);
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.08),
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(
                              color: Colors.white.withValues(alpha: 0.15),
                            ),
                          ),
                          child: Text(
                            cancelLabel,
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: Colors.white70,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),

                    // Confirm
                    Expanded(
                      child: GestureDetector(
                        onTap: () {
                          HapticFeedback.lightImpact();
                          onConfirm?.call();
                          Navigator.pop(context, true);
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [color, color.withValues(alpha: 0.8)],
                            ),
                            borderRadius: BorderRadius.circular(14),
                            boxShadow: [
                              BoxShadow(
                                color: color.withValues(alpha: 0.3),
                                blurRadius: 12,
                                offset: const Offset(0, 4),
                              ),
                            ],
                          ),
                          child: Text(
                            confirmLabel,
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM INPUT DIALOG
// =============================================================================

/// Premium dialog with text input
class PremiumInputDialog extends StatefulWidget {
  final String title;
  final String? message;
  final String? initialValue;
  final String? hint;
  final String confirmLabel;
  final String cancelLabel;
  final IconData? icon;
  final int maxLines;
  final int? maxLength;
  final TextInputType keyboardType;
  final String? Function(String?)? validator;

  const PremiumInputDialog({
    super.key,
    required this.title,
    this.message,
    this.initialValue,
    this.hint,
    this.confirmLabel = 'Save',
    this.cancelLabel = 'Cancel',
    this.icon,
    this.maxLines = 1,
    this.maxLength,
    this.keyboardType = TextInputType.text,
    this.validator,
  });

  /// Show premium input dialog
  static Future<String?> show({
    required BuildContext context,
    required String title,
    String? message,
    String? initialValue,
    String? hint,
    String confirmLabel = 'Save',
    String cancelLabel = 'Cancel',
    IconData? icon,
    int maxLines = 1,
    int? maxLength,
    TextInputType keyboardType = TextInputType.text,
    String? Function(String?)? validator,
  }) {
    return showDialog<String>(
      context: context,
      barrierColor: Colors.black.withValues(alpha: 0.7),
      builder: (context) => PremiumInputDialog(
        title: title,
        message: message,
        initialValue: initialValue,
        hint: hint,
        confirmLabel: confirmLabel,
        cancelLabel: cancelLabel,
        icon: icon,
        maxLines: maxLines,
        maxLength: maxLength,
        keyboardType: keyboardType,
        validator: validator,
      ),
    );
  }

  @override
  State<PremiumInputDialog> createState() => _PremiumInputDialogState();
}

class _PremiumInputDialogState extends State<PremiumInputDialog> {
  late TextEditingController _controller;
  String? _errorText;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialValue);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _handleSave() {
    if (widget.validator != null) {
      final error = widget.validator!(_controller.text);
      if (error != null) {
        setState(() {
          _errorText = error;
        });
        return;
      }
    }
    Navigator.pop(context, _controller.text);
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.transparent,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: Container(
            constraints: const BoxConstraints(maxWidth: 340),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  tone(AppColors.darkBackground, 0.95),
                  tone(AppColors.darkBackground, 0.98),
                ],
              ),
              borderRadius: BorderRadius.circular(24),
              border: Border.all(
                color: Colors.white.withValues(alpha: 0.1),
              ),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Icon
                if (widget.icon != null) ...[
                  Container(
                    width: 56,
                    height: 56,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          AppColors.primaryBlue.withValues(alpha: 0.3),
                          AppColors.primaryBlue.withValues(alpha: 0.1),
                        ],
                      ),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      widget.icon,
                      size: 28,
                      color: AppColors.primaryBlue,
                    ),
                  ),
                  const SizedBox(height: 16),
                ],

                // Title
                Text(
                  widget.title,
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                  textAlign: TextAlign.center,
                ),

                // Message
                if (widget.message != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    widget.message!,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w400,
                      color: Colors.white.withValues(alpha: 0.6),
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],

                const SizedBox(height: 20),

                // Input field
                TextField(
                  controller: _controller,
                  maxLines: widget.maxLines,
                  maxLength: widget.maxLength,
                  keyboardType: widget.keyboardType,
                  style: const TextStyle(
                    fontSize: 15,
                    color: Colors.white,
                  ),
                  decoration: InputDecoration(
                    hintText: widget.hint,
                    hintStyle: TextStyle(
                      color: Colors.white.withValues(alpha: 0.4),
                    ),
                    filled: true,
                    fillColor: Colors.white.withValues(alpha: 0.06),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(14),
                      borderSide: BorderSide.none,
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(14),
                      borderSide: BorderSide(
                        color: AppColors.primaryBlue,
                        width: 2,
                      ),
                    ),
                    errorBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(14),
                      borderSide: BorderSide(
                        color: AppColors.dangerRed,
                        width: 2,
                      ),
                    ),
                    errorText: _errorText,
                    counterStyle: TextStyle(
                      color: Colors.white.withValues(alpha: 0.4),
                    ),
                  ),
                  onChanged: (_) {
                    if (_errorText != null) {
                      setState(() {
                        _errorText = null;
                      });
                    }
                  },
                ),

                const SizedBox(height: 20),

                // Buttons
                Row(
                  children: [
                    Expanded(
                      child: _PremiumDialogButton(
                        label: widget.cancelLabel,
                        onTap: () => Navigator.pop(context),
                        isPrimary: false,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _PremiumDialogButton(
                        label: widget.confirmLabel,
                        onTap: _handleSave,
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
    );
  }
}

// =============================================================================
// PREMIUM MENU SHEET
// =============================================================================

/// Menu item data
class MenuSheetItem {
  final String label;
  final IconData icon;
  final VoidCallback onTap;
  final String? subtitle;
  final Color? color;
  final bool showArrow;
  final Widget? trailing;

  const MenuSheetItem({
    required this.label,
    required this.icon,
    required this.onTap,
    this.subtitle,
    this.color,
    this.showArrow = false,
    this.trailing,
  });
}

/// Premium menu-style bottom sheet
class PremiumMenuSheet extends StatelessWidget {
  final String? title;
  final List<MenuSheetItem> items;
  final bool showDragHandle;

  const PremiumMenuSheet({
    super.key,
    this.title,
    required this.items,
    this.showDragHandle = true,
  });

  /// Show premium menu sheet
  static Future<void> show({
    required BuildContext context,
    String? title,
    required List<MenuSheetItem> items,
    bool showDragHandle = true,
  }) {
    return showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => PremiumMenuSheet(
        title: title,
        items: items,
        showDragHandle: showDragHandle,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.only(
        bottom: MediaQuery.of(context).padding.bottom,
      ),
      child: ClipRRect(
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  tone(AppColors.darkBackground, 0.95),
                  tone(AppColors.darkBackground, 0.98),
                ],
              ),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Drag handle
                if (showDragHandle)
                  Padding(
                    padding: const EdgeInsets.only(top: 12),
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.3),
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),

                // Title
                if (title != null)
                  Padding(
                    padding: EdgeInsets.fromLTRB(
                      20,
                      showDragHandle ? 16 : 20,
                      20,
                      12,
                    ),
                    child: Text(
                      title!,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                      ),
                    ),
                  ),

                // Items
                ...items.asMap().entries.map((entry) {
                  final index = entry.key;
                  final item = entry.value;
                  final isLast = index == items.length - 1;

                  return _buildMenuItem(context, item, isLast);
                }),

                const SizedBox(height: 8),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMenuItem(BuildContext context, MenuSheetItem item, bool isLast) {
    final color = item.color ?? Colors.white;

    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        Navigator.pop(context);
        item.onTap();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        decoration: BoxDecoration(
          border: isLast
              ? null
              : Border(
                  bottom: BorderSide(
                    color: Colors.white.withValues(alpha: 0.08),
                  ),
                ),
        ),
        child: Row(
          children: [
            // Icon
            Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                item.icon,
                size: 22,
                color: color == Colors.white ? Colors.white70 : color,
              ),
            ),
            const SizedBox(width: 14),

            // Label & subtitle
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.label,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: color == Colors.white ? Colors.white : color,
                    ),
                  ),
                  if (item.subtitle != null) ...[
                    const SizedBox(height: 2),
                    Text(
                      item.subtitle!,
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w400,
                        color: Colors.white.withValues(alpha: 0.5),
                      ),
                    ),
                  ],
                ],
              ),
            ),

            // Trailing
            if (item.trailing != null)
              item.trailing!
            else if (item.showArrow)
              Icon(
                Icons.chevron_right,
                size: 22,
                color: Colors.white.withValues(alpha: 0.4),
              ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM PICKER SHEET
// =============================================================================

/// Picker item data
class PickerItem<T> {
  final T value;
  final String label;
  final String? subtitle;
  final IconData? icon;

  const PickerItem({
    required this.value,
    required this.label,
    this.subtitle,
    this.icon,
  });
}

/// Premium selection picker sheet
class PremiumPickerSheet<T> extends StatelessWidget {
  final String title;
  final List<PickerItem<T>> items;
  final T? selectedValue;
  final bool showCheckmark;

  const PremiumPickerSheet({
    super.key,
    required this.title,
    required this.items,
    this.selectedValue,
    this.showCheckmark = true,
  });

  /// Show premium picker sheet
  static Future<T?> show<T>({
    required BuildContext context,
    required String title,
    required List<PickerItem<T>> items,
    T? selectedValue,
    bool showCheckmark = true,
  }) {
    return showModalBottomSheet<T>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => PremiumPickerSheet<T>(
        title: title,
        items: items,
        selectedValue: selectedValue,
        showCheckmark: showCheckmark,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.7,
      ),
      margin: EdgeInsets.only(
        bottom: MediaQuery.of(context).padding.bottom,
      ),
      child: ClipRRect(
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  tone(AppColors.darkBackground, 0.95),
                  tone(AppColors.darkBackground, 0.98),
                ],
              ),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Drag handle
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.3),
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),

                // Title
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        title,
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                          color: Colors.white,
                        ),
                      ),
                      IconButton(
                        onPressed: () => Navigator.pop(context),
                        icon: Container(
                          padding: const EdgeInsets.all(6),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Icon(
                            Icons.close,
                            size: 18,
                            color: Colors.white70,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                // Divider
                Divider(
                  height: 1,
                  color: Colors.white.withValues(alpha: 0.1),
                ),

                // Items
                Flexible(
                  child: ListView.builder(
                    shrinkWrap: true,
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: items.length,
                    itemBuilder: (context, index) {
                      final item = items[index];
                      final isSelected = item.value == selectedValue;

                      return _buildPickerItem(context, item, isSelected);
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPickerItem(
      BuildContext context, PickerItem<T> item, bool isSelected) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        Navigator.pop(context, item.value);
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        color: isSelected
            ? AppColors.primaryBlue.withValues(alpha: 0.1)
            : Colors.transparent,
        child: Row(
          children: [
            // Icon
            if (item.icon != null) ...[
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: isSelected
                      ? AppColors.primaryBlue.withValues(alpha: 0.2)
                      : Colors.white.withValues(alpha: 0.08),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(
                  item.icon,
                  size: 20,
                  color: isSelected ? AppColors.primaryBlue : Colors.white60,
                ),
              ),
              const SizedBox(width: 14),
            ],

            // Label & subtitle
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.label,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
                      color: isSelected ? AppColors.primaryBlue : Colors.white,
                    ),
                  ),
                  if (item.subtitle != null) ...[
                    const SizedBox(height: 2),
                    Text(
                      item.subtitle!,
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.white.withValues(alpha: 0.5),
                      ),
                    ),
                  ],
                ],
              ),
            ),

            // Checkmark
            if (showCheckmark && isSelected)
              Icon(
                Icons.check_circle,
                size: 24,
                color: AppColors.primaryBlue,
              ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM LOADING OVERLAY
// =============================================================================

/// Premium loading overlay
class PremiumLoadingOverlay extends StatelessWidget {
  final String? message;

  const PremiumLoadingOverlay({
    super.key,
    this.message,
  });

  /// Show loading overlay
  static Future<T> show<T>({
    required BuildContext context,
    required Future<T> Function() task,
    String? message,
  }) async {
    final overlay = OverlayEntry(
      builder: (context) => PremiumLoadingOverlay(message: message),
    );

    Overlay.of(context).insert(overlay);

    try {
      final result = await task();
      overlay.remove();
      return result;
    } catch (e) {
      overlay.remove();
      rethrow;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.black.withValues(alpha: 0.6),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 5, sigmaY: 5),
        child: Center(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
              child: Container(
                padding: const EdgeInsets.all(28),
                decoration: BoxDecoration(
                  color: tone(AppColors.darkBackground, 0.95),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.1),
                  ),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SizedBox(
                      width: 48,
                      height: 48,
                      child: CircularProgressIndicator(
                        strokeWidth: 3,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          AppColors.primaryBlue,
                        ),
                      ),
                    ),
                    if (message != null) ...[
                      const SizedBox(height: 20),
                      Text(
                        message!,
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w500,
                          color: Colors.white.withValues(alpha: 0.7),
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
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
// PREMIUM SUCCESS OVERLAY
// =============================================================================

/// Premium success overlay with animation
class PremiumSuccessOverlay extends StatefulWidget {
  final String? message;
  final VoidCallback? onComplete;
  final Duration duration;

  const PremiumSuccessOverlay({
    super.key,
    this.message,
    this.onComplete,
    this.duration = const Duration(milliseconds: 1500),
  });

  /// Show success overlay
  static void show({
    required BuildContext context,
    String? message,
    Duration duration = const Duration(milliseconds: 1500),
  }) {
    late OverlayEntry overlay;

    overlay = OverlayEntry(
      builder: (context) => PremiumSuccessOverlay(
        message: message,
        duration: duration,
        onComplete: () => overlay.remove(),
      ),
    );

    Overlay.of(context).insert(overlay);
  }

  @override
  State<PremiumSuccessOverlay> createState() => _PremiumSuccessOverlayState();
}

class _PremiumSuccessOverlayState extends State<PremiumSuccessOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutBack),
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    _controller.forward();
    HapticFeedback.mediumImpact();

    Future.delayed(widget.duration, () {
      if (mounted) {
        _controller.reverse().then((_) {
          widget.onComplete?.call();
        });
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
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return FadeTransition(
          opacity: _fadeAnimation,
          child: Material(
            color: Colors.black.withValues(alpha: 0.5),
            child: Center(
              child: ScaleTransition(
                scale: _scaleAnimation,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: BackdropFilter(
                    filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                    child: Container(
                      padding: const EdgeInsets.all(32),
                      decoration: BoxDecoration(
                        color: tone(AppColors.darkBackground, 0.95),
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(
                          color: AppColors.successGreen.withValues(alpha: 0.3),
                        ),
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          // Checkmark
                          Container(
                            width: 72,
                            height: 72,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  AppColors.successGreen,
                                  AppColors.successGreen.withValues(alpha: 0.7),
                                ],
                              ),
                              shape: BoxShape.circle,
                              boxShadow: [
                                BoxShadow(
                                  color:
                                      AppColors.successGreen.withValues(alpha: 0.4),
                                  blurRadius: 20,
                                ),
                              ],
                            ),
                            child: const Icon(
                              Icons.check,
                              size: 40,
                              color: Colors.white,
                            ),
                          ),
                          if (widget.message != null) ...[
                            const SizedBox(height: 20),
                            Text(
                              widget.message!,
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                                color: Colors.white,
                              ),
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ],
                      ),
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
}
