/// Premium Copilot Widgets
///
/// Premium AI chat interface components with glass morphism design,
/// smooth animations, and professional styling.
library;

import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../../models/copilot_message.dart';
import '../../../theme/app_colors.dart';

// =============================================================================
// PREMIUM CHAT BUBBLE
// =============================================================================

/// Premium chat bubble with glass morphism and animations.
class PremiumChatBubble extends StatefulWidget {
  final CopilotMessage message;
  final bool showAvatar;
  final bool animate;
  final VoidCallback? onCopy;
  final VoidCallback? onShare;
  final VoidCallback? onRetry;

  const PremiumChatBubble({
    super.key,
    required this.message,
    this.showAvatar = true,
    this.animate = true,
    this.onCopy,
    this.onShare,
    this.onRetry,
  });

  @override
  State<PremiumChatBubble> createState() => _PremiumChatBubbleState();
}

class _PremiumChatBubbleState extends State<PremiumChatBubble>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _scaleAnimation;
  bool _isHovered = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    _slideAnimation = Tween<Offset>(
      begin: Offset(widget.message.isUser ? 0.3 : -0.3, 0),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic));

    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutBack),
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

  @override
  Widget build(BuildContext context) {
    final isUser = widget.message.isUser;

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return FadeTransition(
          opacity: _fadeAnimation,
          child: SlideTransition(
            position: _slideAnimation,
            child: ScaleTransition(
              scale: _scaleAnimation,
              child: child,
            ),
          ),
        );
      },
      child: Padding(
        padding: const EdgeInsets.only(bottom: 16),
        child: Row(
          mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (!isUser && widget.showAvatar) ...[
              _buildAssistantAvatar(),
              const SizedBox(width: 12),
            ],
            Flexible(
              child: MouseRegion(
                onEnter: (_) => setState(() => _isHovered = true),
                onExit: (_) => setState(() => _isHovered = false),
                child: GestureDetector(
                  onLongPress: () {
                    HapticFeedback.mediumImpact();
                    _showMessageActions(context);
                  },
                  child: _buildBubble(isUser),
                ),
              ),
            ),
            if (isUser && widget.showAvatar) ...[
              const SizedBox(width: 12),
              _buildUserAvatar(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildAssistantAvatar() {
    return Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            AppColors.primaryBlue.withOpacity(0.3),
            AppColors.primaryBlue.withOpacity(0.1),
          ],
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: AppColors.primaryBlue.withOpacity(0.4),
        ),
        boxShadow: [
          BoxShadow(
            color: AppColors.primaryBlue.withOpacity(0.2),
            blurRadius: 12,
            spreadRadius: 0,
          ),
        ],
      ),
      child: const Icon(
        Icons.auto_awesome,
        size: 20,
        color: Colors.white,
      ),
    );
  }

  Widget _buildUserAvatar() {
    return Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.white.withOpacity(0.12),
            Colors.white.withOpacity(0.04),
          ],
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Colors.white.withOpacity(0.15),
        ),
      ),
      child: Icon(
        Icons.person_outline,
        size: 20,
        color: Colors.white.withOpacity(0.8),
      ),
    );
  }

  Widget _buildBubble(bool isUser) {
    return Container(
      constraints: const BoxConstraints(maxWidth: 320),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: isUser
              ? [
                  AppColors.primaryBlue.withOpacity(0.25),
                  AppColors.primaryBlue.withOpacity(0.15),
                ]
              : [
                  Colors.white.withOpacity(0.08),
                  Colors.white.withOpacity(0.03),
                ],
        ),
        borderRadius: BorderRadius.only(
          topLeft: const Radius.circular(20),
          topRight: const Radius.circular(20),
          bottomLeft: Radius.circular(isUser ? 20 : 4),
          bottomRight: Radius.circular(isUser ? 4 : 20),
        ),
        border: Border.all(
          color: isUser
              ? AppColors.primaryBlue.withOpacity(0.4)
              : Colors.white.withOpacity(0.1),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.15),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.only(
          topLeft: const Radius.circular(20),
          topRight: const Radius.circular(20),
          bottomLeft: Radius.circular(isUser ? 20 : 4),
          bottomRight: Radius.circular(isUser ? 4 : 20),
        ),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildMessageContent(),
                if (_isHovered && !isUser) ...[
                  const SizedBox(height: 12),
                  _buildActionRow(),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMessageContent() {
    return Text(
      widget.message.body,
      style: const TextStyle(
        color: Colors.white,
        fontSize: 15,
        height: 1.5,
        fontWeight: FontWeight.w400,
      ),
    );
  }

  Widget _buildActionRow() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _buildActionButton(Icons.copy_outlined, 'Copy', widget.onCopy),
        const SizedBox(width: 8),
        _buildActionButton(Icons.share_outlined, 'Share', widget.onShare),
        if (widget.onRetry != null) ...[
          const SizedBox(width: 8),
          _buildActionButton(Icons.refresh, 'Retry', widget.onRetry),
        ],
      ],
    );
  }

  Widget _buildActionButton(IconData icon, String tooltip, VoidCallback? onTap) {
    return Tooltip(
      message: tooltip,
      child: InkWell(
        onTap: () {
          HapticFeedback.lightImpact();
          onTap?.call();
        },
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.06),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.white.withOpacity(0.1)),
          ),
          child: Icon(
            icon,
            size: 16,
            color: Colors.white.withOpacity(0.7),
          ),
        ),
      ),
    );
  }

  void _showMessageActions(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => _MessageActionsSheet(
        onCopy: widget.onCopy,
        onShare: widget.onShare,
        onRetry: widget.onRetry,
        message: widget.message.body,
      ),
    );
  }
}

class _MessageActionsSheet extends StatelessWidget {
  final VoidCallback? onCopy;
  final VoidCallback? onShare;
  final VoidCallback? onRetry;
  final String message;

  const _MessageActionsSheet({
    this.onCopy,
    this.onShare,
    this.onRetry,
    required this.message,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.white.withOpacity(0.1),
            Colors.white.withOpacity(0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(height: 12),
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 20),
              _buildActionTile(
                context,
                Icons.copy_outlined,
                'Copy message',
                () {
                  Clipboard.setData(ClipboardData(text: message));
                  Navigator.pop(context);
                  onCopy?.call();
                },
              ),
              _buildActionTile(
                context,
                Icons.share_outlined,
                'Share',
                () {
                  Navigator.pop(context);
                  onShare?.call();
                },
              ),
              if (onRetry != null)
                _buildActionTile(
                  context,
                  Icons.refresh,
                  'Regenerate response',
                  () {
                    Navigator.pop(context);
                    onRetry?.call();
                  },
                ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildActionTile(
    BuildContext context,
    IconData icon,
    String label,
    VoidCallback onTap,
  ) {
    return ListTile(
      leading: Icon(icon, color: Colors.white.withOpacity(0.8)),
      title: Text(
        label,
        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w500),
      ),
      onTap: () {
        HapticFeedback.lightImpact();
        onTap();
      },
    );
  }
}

// =============================================================================
// TYPING INDICATOR
// =============================================================================

/// Animated typing indicator with bouncing dots.
class PremiumTypingIndicator extends StatefulWidget {
  final Color? color;
  final double dotSize;
  final Duration duration;

  const PremiumTypingIndicator({
    super.key,
    this.color,
    this.dotSize = 8,
    this.duration = const Duration(milliseconds: 1200),
  });

  @override
  State<PremiumTypingIndicator> createState() => _PremiumTypingIndicatorState();
}

class _PremiumTypingIndicatorState extends State<PremiumTypingIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.duration,
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
    return Row(
      mainAxisAlignment: MainAxisAlignment.start,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // AI Avatar
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                AppColors.primaryBlue.withOpacity(0.3),
                AppColors.primaryBlue.withOpacity(0.1),
              ],
            ),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: AppColors.primaryBlue.withOpacity(0.4),
            ),
          ),
          child: const Icon(
            Icons.auto_awesome,
            size: 20,
            color: Colors.white,
          ),
        ),
        const SizedBox(width: 12),
        // Bubble
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withOpacity(0.08),
                Colors.white.withOpacity(0.03),
              ],
            ),
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(20),
              topRight: Radius.circular(20),
              bottomLeft: Radius.circular(4),
              bottomRight: Radius.circular(20),
            ),
            border: Border.all(color: Colors.white.withOpacity(0.1)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: List.generate(3, (index) {
              return AnimatedBuilder(
                animation: _controller,
                builder: (context, child) {
                  final delay = index * 0.2;
                  final value = math.sin(
                    (_controller.value * 2 * math.pi) - (delay * math.pi),
                  );
                  final offset = value * 4;
                  final opacity = 0.4 + (value + 1) * 0.3;

                  return Container(
                    margin: EdgeInsets.only(right: index < 2 ? 6 : 0),
                    child: Transform.translate(
                      offset: Offset(0, -offset),
                      child: Container(
                        width: widget.dotSize,
                        height: widget.dotSize,
                        decoration: BoxDecoration(
                          color: (widget.color ?? AppColors.primaryBlue)
                              .withOpacity(opacity),
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color: (widget.color ?? AppColors.primaryBlue)
                                  .withOpacity(0.3),
                              blurRadius: 6,
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              );
            }),
          ),
        ),
      ],
    );
  }
}

// =============================================================================
// SUGGESTED PROMPTS
// =============================================================================

/// Premium suggested prompts with animations.
class PremiumSuggestedPrompts extends StatefulWidget {
  final List<String> prompts;
  final Function(String) onPromptTap;
  final bool animate;

  const PremiumSuggestedPrompts({
    super.key,
    required this.prompts,
    required this.onPromptTap,
    this.animate = true,
  });

  @override
  State<PremiumSuggestedPrompts> createState() => _PremiumSuggestedPromptsState();
}

class _PremiumSuggestedPromptsState extends State<PremiumSuggestedPrompts>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
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

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 12),
          child: Row(
            children: [
              Icon(
                Icons.lightbulb_outline,
                size: 16,
                color: Colors.white.withOpacity(0.6),
              ),
              const SizedBox(width: 8),
              Text(
                'Suggested prompts',
                style: TextStyle(
                  color: Colors.white.withOpacity(0.6),
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: List.generate(widget.prompts.length, (index) {
            return AnimatedBuilder(
              animation: _controller,
              builder: (context, child) {
                final delay = index * 0.15;
                final animValue = Curves.easeOutBack.transform(
                  ((_controller.value - delay) / (1 - delay)).clamp(0.0, 1.0),
                );

                return Transform.scale(
                  scale: animValue,
                  child: Opacity(
                    opacity: animValue,
                    child: child,
                  ),
                );
              },
              child: _PremiumPromptChip(
                label: widget.prompts[index],
                onTap: () => widget.onPromptTap(widget.prompts[index]),
              ),
            );
          }),
        ),
      ],
    );
  }
}

class _PremiumPromptChip extends StatefulWidget {
  final String label;
  final VoidCallback onTap;

  const _PremiumPromptChip({
    required this.label,
    required this.onTap,
  });

  @override
  State<_PremiumPromptChip> createState() => _PremiumPromptChipState();
}

class _PremiumPromptChipState extends State<_PremiumPromptChip>
    with SingleTickerProviderStateMixin {
  late AnimationController _pressController;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _pressController = AnimationController(
      duration: const Duration(milliseconds: 100),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 0.95).animate(
      CurvedAnimation(parent: _pressController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pressController.dispose();
    super.dispose();
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
        onTapDown: (_) => _pressController.forward(),
        onTapUp: (_) {
          _pressController.reverse();
          HapticFeedback.lightImpact();
          widget.onTap();
        },
        onTapCancel: () => _pressController.reverse(),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                AppColors.primaryBlue.withOpacity(0.15),
                AppColors.primaryBlue.withOpacity(0.05),
              ],
            ),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: AppColors.primaryBlue.withOpacity(0.3),
            ),
            boxShadow: [
              BoxShadow(
                color: AppColors.primaryBlue.withOpacity(0.1),
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.arrow_forward_rounded,
                size: 14,
                color: AppColors.primaryBlue.withOpacity(0.8),
              ),
              const SizedBox(width: 8),
              Text(
                widget.label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
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
// RESPONSE CARDS
// =============================================================================

/// Premium response card for structured AI responses.
class PremiumResponseCard extends StatefulWidget {
  final String title;
  final String? subtitle;
  final IconData icon;
  final Color accentColor;
  final Widget child;
  final bool collapsible;
  final bool initiallyExpanded;
  final VoidCallback? onTap;

  const PremiumResponseCard({
    super.key,
    required this.title,
    this.subtitle,
    required this.icon,
    this.accentColor = AppColors.primaryBlue,
    required this.child,
    this.collapsible = false,
    this.initiallyExpanded = true,
    this.onTap,
  });

  /// Creates a trade idea card
  factory PremiumResponseCard.tradeIdea({
    required String ticker,
    required String signal,
    required Widget child,
    VoidCallback? onTap,
  }) {
    return PremiumResponseCard(
      title: ticker,
      subtitle: signal,
      icon: Icons.trending_up,
      accentColor: AppColors.successGreen,
      onTap: onTap,
      child: child,
    );
  }

  /// Creates an analysis card
  factory PremiumResponseCard.analysis({
    required String title,
    required Widget child,
  }) {
    return PremiumResponseCard(
      title: title,
      icon: Icons.analytics_outlined,
      accentColor: AppColors.primaryBlue,
      child: child,
    );
  }

  /// Creates a risk assessment card
  factory PremiumResponseCard.risk({
    required String riskLevel,
    required Widget child,
  }) {
    final color = riskLevel.toLowerCase() == 'high'
        ? AppColors.dangerRed
        : riskLevel.toLowerCase() == 'low'
            ? AppColors.successGreen
            : AppColors.warningOrange;

    return PremiumResponseCard(
      title: 'Risk Assessment',
      subtitle: riskLevel,
      icon: Icons.shield_outlined,
      accentColor: color,
      child: child,
    );
  }

  @override
  State<PremiumResponseCard> createState() => _PremiumResponseCardState();
}

class _PremiumResponseCardState extends State<PremiumResponseCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _expandController;
  late Animation<double> _expandAnimation;
  late bool _isExpanded;

  @override
  void initState() {
    super.initState();
    _isExpanded = widget.initiallyExpanded;
    _expandController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
      value: _isExpanded ? 1.0 : 0.0,
    );
    _expandAnimation = CurvedAnimation(
      parent: _expandController,
      curve: Curves.easeOutCubic,
    );
  }

  @override
  void dispose() {
    _expandController.dispose();
    super.dispose();
  }

  void _toggleExpand() {
    setState(() {
      _isExpanded = !_isExpanded;
      if (_isExpanded) {
        _expandController.forward();
      } else {
        _expandController.reverse();
      }
    });
    HapticFeedback.lightImpact();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.white.withOpacity(0.08),
            Colors.white.withOpacity(0.03),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.15),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              InkWell(
                onTap: widget.collapsible ? _toggleExpand : widget.onTap,
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(20),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      // Icon
                      Container(
                        width: 44,
                        height: 44,
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              widget.accentColor.withOpacity(0.25),
                              widget.accentColor.withOpacity(0.1),
                            ],
                          ),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: widget.accentColor.withOpacity(0.3),
                          ),
                        ),
                        child: Icon(
                          widget.icon,
                          size: 22,
                          color: widget.accentColor,
                        ),
                      ),
                      const SizedBox(width: 14),
                      // Title & Subtitle
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              widget.title,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            if (widget.subtitle != null) ...[
                              const SizedBox(height: 2),
                              Text(
                                widget.subtitle!,
                                style: TextStyle(
                                  color: widget.accentColor,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                      // Expand/Collapse Icon
                      if (widget.collapsible)
                        AnimatedBuilder(
                          animation: _expandAnimation,
                          builder: (context, child) {
                            return Transform.rotate(
                              angle: _expandAnimation.value * math.pi,
                              child: Icon(
                                Icons.keyboard_arrow_down,
                                color: Colors.white.withOpacity(0.5),
                              ),
                            );
                          },
                        ),
                    ],
                  ),
                ),
              ),
              // Content
              SizeTransition(
                sizeFactor: widget.collapsible
                    ? _expandAnimation
                    : const AlwaysStoppedAnimation(1.0),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                  child: widget.child,
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
// CODE BLOCK
// =============================================================================

/// Premium code block with syntax styling.
class PremiumCodeBlock extends StatefulWidget {
  final String code;
  final String? language;
  final bool showLineNumbers;
  final bool copyable;

  const PremiumCodeBlock({
    super.key,
    required this.code,
    this.language,
    this.showLineNumbers = true,
    this.copyable = true,
  });

  @override
  State<PremiumCodeBlock> createState() => _PremiumCodeBlockState();
}

class _PremiumCodeBlockState extends State<PremiumCodeBlock> {
  bool _copied = false;

  void _copyCode() {
    Clipboard.setData(ClipboardData(text: widget.code));
    HapticFeedback.lightImpact();
    setState(() => _copied = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _copied = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    final lines = widget.code.split('\n');

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF0D1117),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.04),
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(12),
              ),
            ),
            child: Row(
              children: [
                // Language badge
                if (widget.language != null)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.primaryBlue.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: AppColors.primaryBlue.withOpacity(0.3),
                      ),
                    ),
                    child: Text(
                      widget.language!,
                      style: TextStyle(
                        color: AppColors.primaryBlue,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                const Spacer(),
                // Copy button
                if (widget.copyable)
                  InkWell(
                    onTap: _copyCode,
                    borderRadius: BorderRadius.circular(8),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.06),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: Colors.white.withOpacity(0.1),
                        ),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            _copied ? Icons.check : Icons.copy_outlined,
                            size: 14,
                            color: _copied
                                ? AppColors.successGreen
                                : Colors.white.withOpacity(0.7),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            _copied ? 'Copied!' : 'Copy',
                            style: TextStyle(
                              color: _copied
                                  ? AppColors.successGreen
                                  : Colors.white.withOpacity(0.7),
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
          // Code
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.all(16),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Line numbers
                if (widget.showLineNumbers)
                  Padding(
                    padding: const EdgeInsets.only(right: 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: List.generate(lines.length, (index) {
                        return Text(
                          '${index + 1}',
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.3),
                            fontSize: 13,
                            fontFamily: 'monospace',
                            height: 1.5,
                          ),
                        );
                      }),
                    ),
                  ),
                // Code content
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: lines.map((line) {
                    return Text(
                      line,
                      style: const TextStyle(
                        color: Color(0xFFE6EDF3),
                        fontSize: 13,
                        fontFamily: 'monospace',
                        height: 1.5,
                      ),
                    );
                  }).toList(),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// =============================================================================
// PREMIUM INPUT FIELD
// =============================================================================

/// Premium chat input field with glass morphism.
class PremiumChatInput extends StatefulWidget {
  final TextEditingController controller;
  final bool isSending;
  final VoidCallback onSend;
  final VoidCallback? onVoice;
  final String hintText;

  const PremiumChatInput({
    super.key,
    required this.controller,
    required this.isSending,
    required this.onSend,
    this.onVoice,
    this.hintText = 'Ask Copilot anything...',
  });

  @override
  State<PremiumChatInput> createState() => _PremiumChatInputState();
}

class _PremiumChatInputState extends State<PremiumChatInput>
    with SingleTickerProviderStateMixin {
  late AnimationController _sendController;
  late Animation<double> _sendAnimation;
  bool _hasFocus = false;

  @override
  void initState() {
    super.initState();
    _sendController = AnimationController(
      duration: const Duration(milliseconds: 200),
      vsync: this,
    );
    _sendAnimation = Tween<double>(begin: 1.0, end: 0.9).animate(
      CurvedAnimation(parent: _sendController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _sendController.dispose();
    super.dispose();
  }

  void _handleSend() {
    if (widget.controller.text.trim().isEmpty || widget.isSending) return;

    _sendController.forward().then((_) => _sendController.reverse());
    HapticFeedback.mediumImpact();
    widget.onSend();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.white.withOpacity(_hasFocus ? 0.1 : 0.06),
            Colors.white.withOpacity(_hasFocus ? 0.05 : 0.02),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: _hasFocus
              ? AppColors.primaryBlue.withOpacity(0.5)
              : Colors.white.withOpacity(0.1),
          width: _hasFocus ? 1.5 : 1,
        ),
        boxShadow: _hasFocus
            ? [
                BoxShadow(
                  color: AppColors.primaryBlue.withOpacity(0.15),
                  blurRadius: 12,
                  spreadRadius: 0,
                ),
              ]
            : null,
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              children: [
                // Text Field
                Focus(
                  onFocusChange: (focused) {
                    setState(() => _hasFocus = focused);
                  },
                  child: TextField(
                    controller: widget.controller,
                    maxLines: 4,
                    minLines: 1,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 15,
                    ),
                    decoration: InputDecoration(
                      hintText: widget.hintText,
                      hintStyle: TextStyle(
                        color: Colors.white.withOpacity(0.4),
                        fontSize: 15,
                      ),
                      border: InputBorder.none,
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                    ),
                    onSubmitted: (_) => _handleSend(),
                  ),
                ),
                const SizedBox(height: 12),
                // Actions Row
                Row(
                  children: [
                    // Voice button
                    if (widget.onVoice != null)
                      _buildIconButton(
                        icon: Icons.mic_none,
                        onTap: widget.onVoice!,
                      ),
                    const Spacer(),
                    // Send button
                    AnimatedBuilder(
                      animation: _sendAnimation,
                      builder: (context, child) {
                        return Transform.scale(
                          scale: _sendAnimation.value,
                          child: child,
                        );
                      },
                      child: GestureDetector(
                        onTap: _handleSend,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 20,
                            vertical: 12,
                          ),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                              colors: widget.isSending
                                  ? [
                                      Colors.white.withOpacity(0.1),
                                      Colors.white.withOpacity(0.05),
                                    ]
                                  : [
                                      AppColors.primaryBlue,
                                      AppColors.primaryBlue.withOpacity(0.8),
                                    ],
                            ),
                            borderRadius: BorderRadius.circular(14),
                            boxShadow: widget.isSending
                                ? null
                                : [
                                    BoxShadow(
                                      color: AppColors.primaryBlue.withOpacity(0.4),
                                      blurRadius: 12,
                                      offset: const Offset(0, 4),
                                    ),
                                  ],
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              if (widget.isSending) ...[
                                SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white.withOpacity(0.7),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Text(
                                  'Sending...',
                                  style: TextStyle(
                                    color: Colors.white.withOpacity(0.7),
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ] else ...[
                                const Icon(
                                  Icons.send_rounded,
                                  size: 18,
                                  color: Colors.white,
                                ),
                                const SizedBox(width: 8),
                                const Text(
                                  'Send',
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ],
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

  Widget _buildIconButton({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap();
      },
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.06),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.white.withOpacity(0.1)),
        ),
        child: Icon(
          icon,
          size: 20,
          color: Colors.white.withOpacity(0.7),
        ),
      ),
    );
  }
}

// =============================================================================
// COPILOT HEADER
// =============================================================================

/// Premium Copilot header with branding and status.
class PremiumCopilotHeader extends StatefulWidget {
  final bool isOnline;
  final VoidCallback? onVoice;
  final VoidCallback? onNotes;

  const PremiumCopilotHeader({
    super.key,
    this.isOnline = true,
    this.onVoice,
    this.onNotes,
  });

  @override
  State<PremiumCopilotHeader> createState() => _PremiumCopilotHeaderState();
}

class _PremiumCopilotHeaderState extends State<PremiumCopilotHeader>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 2000),
      vsync: this,
    );
    _pulseAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    if (widget.isOnline) {
      _pulseController.repeat(reverse: true);
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
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
            Colors.white.withOpacity(0.08),
            Colors.white.withOpacity(0.03),
          ],
        ),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.2),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Top Row
              Row(
                children: [
                  // Status Badge
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Colors.white.withOpacity(0.1),
                          Colors.white.withOpacity(0.05),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.15),
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        AnimatedBuilder(
                          animation: _pulseAnimation,
                          builder: (context, child) {
                            return Transform.scale(
                              scale: _pulseAnimation.value,
                              child: Container(
                                width: 8,
                                height: 8,
                                decoration: BoxDecoration(
                                  color: widget.isOnline
                                      ? AppColors.successGreen
                                      : AppColors.dangerRed,
                                  shape: BoxShape.circle,
                                  boxShadow: widget.isOnline
                                      ? [
                                          BoxShadow(
                                            color: AppColors.successGreen
                                                .withOpacity(0.5),
                                            blurRadius: 6,
                                          ),
                                        ]
                                      : null,
                                ),
                              ),
                            );
                          },
                        ),
                        const SizedBox(width: 8),
                        Text(
                          widget.isOnline ? 'Online' : 'Offline',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const Spacer(),
                  // Action Buttons
                  if (widget.onVoice != null)
                    _buildHeaderAction(
                      icon: Icons.mic_none,
                      label: 'Voice',
                      onTap: widget.onVoice!,
                    ),
                  if (widget.onNotes != null) ...[
                    const SizedBox(width: 8),
                    _buildHeaderAction(
                      icon: Icons.note_alt_outlined,
                      label: 'Notes',
                      onTap: widget.onNotes!,
                      isOutlined: true,
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 16),
              // Title
              Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [
                          AppColors.primaryBlue.withOpacity(0.3),
                          AppColors.primaryBlue.withOpacity(0.1),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color: AppColors.primaryBlue.withOpacity(0.4),
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.primaryBlue.withOpacity(0.2),
                          blurRadius: 12,
                        ),
                      ],
                    ),
                    child: const Icon(
                      Icons.auto_awesome,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                  const SizedBox(width: 14),
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Quant Copilot',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      SizedBox(height: 2),
                      Text(
                        'Context-aware chat with structured answers',
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeaderAction({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool isOutlined = false,
  }) {
    return InkWell(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap();
      },
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          gradient: isOutlined
              ? null
              : LinearGradient(
                  colors: [
                    Colors.white.withOpacity(0.1),
                    Colors.white.withOpacity(0.05),
                  ],
                ),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: Colors.white.withOpacity(isOutlined ? 0.2 : 0.1),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 18,
              color: Colors.white.withOpacity(0.8),
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                color: Colors.white.withOpacity(0.9),
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// CONTEXT CARD
// =============================================================================

/// Premium context card showing current stock context.
class PremiumContextCard extends StatelessWidget {
  final String ticker;
  final String signal;
  final String? playStyle;
  final double? icsScore;
  final String? icsTier;
  final double? winProb;
  final double? qualityScore;
  final double? atrPct;
  final VoidCallback onClear;

  const PremiumContextCard({
    super.key,
    required this.ticker,
    required this.signal,
    this.playStyle,
    this.icsScore,
    this.icsTier,
    this.winProb,
    this.qualityScore,
    this.atrPct,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            AppColors.primaryBlue.withOpacity(0.15),
            AppColors.primaryBlue.withOpacity(0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppColors.primaryBlue.withOpacity(0.3),
        ),
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
                // Header
                Row(
                  children: [
                    // Ticker Badge
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            AppColors.primaryBlue.withOpacity(0.3),
                            AppColors.primaryBlue.withOpacity(0.15),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: AppColors.primaryBlue.withOpacity(0.4),
                        ),
                      ),
                      child: Text(
                        ticker,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    // Signal & Style
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            signal,
                            style: TextStyle(
                              color: AppColors.primaryBlue,
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          if (playStyle != null)
                            Text(
                              playStyle!,
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.6),
                                fontSize: 12,
                              ),
                            ),
                        ],
                      ),
                    ),
                    // Clear Button
                    IconButton(
                      onPressed: () {
                        HapticFeedback.lightImpact();
                        onClear();
                      },
                      icon: Icon(
                        Icons.close,
                        color: Colors.white.withOpacity(0.5),
                        size: 20,
                      ),
                      style: IconButton.styleFrom(
                        backgroundColor: Colors.white.withOpacity(0.06),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                    ),
                  ],
                ),
                // Metrics Row
                if (icsScore != null || winProb != null || qualityScore != null) ...[
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.04),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.06),
                      ),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        if (icsScore != null)
                          _buildMetric(
                            'ICS',
                            '${icsScore!.toStringAsFixed(0)}/100',
                            icsTier,
                          ),
                        if (winProb != null)
                          _buildMetric(
                            'Win Prob',
                            '${(winProb! * 100).toStringAsFixed(0)}%',
                            null,
                          ),
                        if (qualityScore != null)
                          _buildMetric(
                            'Quality',
                            qualityScore!.toStringAsFixed(1),
                            null,
                          ),
                        if (atrPct != null)
                          _buildMetric(
                            'ATR',
                            '${(atrPct! * 100).toStringAsFixed(1)}%',
                            null,
                          ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMetric(String label, String value, String? tier) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 15,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          tier != null ? '$label ($tier)' : label,
          style: TextStyle(
            color: Colors.white.withOpacity(0.5),
            fontSize: 11,
          ),
        ),
      ],
    );
  }
}
