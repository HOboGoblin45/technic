/// Premium Social & Sharing Widgets
///
/// A collection of premium social and sharing components with glass morphism,
/// smooth animations, and professional styling.
///
/// Components:
/// - PremiumShareCard: Shareable stock/analysis card
/// - PremiumSocialButton: Social media buttons
/// - PremiumShareSheet: Share options bottom sheet
/// - PremiumExportOptions: Export format selector
/// - PremiumInviteCard: Invite friends card
/// - PremiumReferralBanner: Referral program banner
/// - PremiumSharePreview: Preview of shared content
/// - PremiumCopyLink: Copy link button
/// - PremiumQRCode: QR code display
/// - PremiumTestimonialCard: User testimonial card
library;

import 'dart:ui';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_colors.dart';
import '../utils/helpers.dart';

// =============================================================================
// SOCIAL PLATFORM ENUM
// =============================================================================

/// Social platform types
enum SocialPlatform {
  twitter,
  facebook,
  linkedin,
  whatsapp,
  telegram,
  email,
  sms,
  copy,
  more,
}

/// Social platform data
extension SocialPlatformData on SocialPlatform {
  String get label {
    switch (this) {
      case SocialPlatform.twitter:
        return 'Twitter';
      case SocialPlatform.facebook:
        return 'Facebook';
      case SocialPlatform.linkedin:
        return 'LinkedIn';
      case SocialPlatform.whatsapp:
        return 'WhatsApp';
      case SocialPlatform.telegram:
        return 'Telegram';
      case SocialPlatform.email:
        return 'Email';
      case SocialPlatform.sms:
        return 'Message';
      case SocialPlatform.copy:
        return 'Copy';
      case SocialPlatform.more:
        return 'More';
    }
  }

  IconData get icon {
    switch (this) {
      case SocialPlatform.twitter:
        return Icons.alternate_email;
      case SocialPlatform.facebook:
        return Icons.facebook;
      case SocialPlatform.linkedin:
        return Icons.business;
      case SocialPlatform.whatsapp:
        return Icons.chat;
      case SocialPlatform.telegram:
        return Icons.send;
      case SocialPlatform.email:
        return Icons.email_outlined;
      case SocialPlatform.sms:
        return Icons.sms_outlined;
      case SocialPlatform.copy:
        return Icons.copy;
      case SocialPlatform.more:
        return Icons.more_horiz;
    }
  }

  Color get color {
    switch (this) {
      case SocialPlatform.twitter:
        return const Color(0xFF1DA1F2);
      case SocialPlatform.facebook:
        return const Color(0xFF4267B2);
      case SocialPlatform.linkedin:
        return const Color(0xFF0077B5);
      case SocialPlatform.whatsapp:
        return const Color(0xFF25D366);
      case SocialPlatform.telegram:
        return const Color(0xFF0088CC);
      case SocialPlatform.email:
        return const Color(0xFFEA4335);
      case SocialPlatform.sms:
        return const Color(0xFF34C759);
      case SocialPlatform.copy:
        return AppColors.primaryBlue;
      case SocialPlatform.more:
        return Colors.grey;
    }
  }
}

// =============================================================================
// PREMIUM SHARE CARD
// =============================================================================

/// Premium shareable stock card with branding
class PremiumShareCard extends StatelessWidget {
  final String ticker;
  final String companyName;
  final double price;
  final double change;
  final double changePercent;
  final double? meritScore;
  final String? signal;
  final bool showBranding;
  final Color? accentColor;

  const PremiumShareCard({
    super.key,
    required this.ticker,
    required this.companyName,
    required this.price,
    required this.change,
    required this.changePercent,
    this.meritScore,
    this.signal,
    this.showBranding = true,
    this.accentColor,
  });

  @override
  Widget build(BuildContext context) {
    final isPositive = change >= 0;
    final changeColor = isPositive ? AppColors.successGreen : AppColors.dangerRed;
    final accent = accentColor ?? AppColors.primaryBlue;

    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                tone(AppColors.darkBackground, 0.95),
                tone(AppColors.darkBackground, 0.98),
              ],
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header with ticker and signal
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      // Ticker badge
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [accent, accent.withValues(alpha: 0.7)],
                          ),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Text(
                          ticker,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      // Company name
                      Flexible(
                        child: Text(
                          companyName,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                            color: Colors.white.withValues(alpha: 0.7),
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                  // Signal badge
                  if (signal != null)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: changeColor.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: changeColor.withValues(alpha: 0.3),
                        ),
                      ),
                      child: Text(
                        signal!,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: changeColor,
                          letterSpacing: 0.5,
                        ),
                      ),
                    ),
                ],
              ),

              const SizedBox(height: 20),

              // Price section
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '\$${price.toStringAsFixed(2)}',
                    style: const TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      letterSpacing: -0.5,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Row(
                      children: [
                        Icon(
                          isPositive
                              ? Icons.arrow_upward
                              : Icons.arrow_downward,
                          size: 18,
                          color: changeColor,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          '${isPositive ? '+' : ''}\$${change.toStringAsFixed(2)} (${changePercent.toStringAsFixed(2)}%)',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                            color: changeColor,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),

              // MERIT score
              if (meritScore != null) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.08),
                    ),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Icon(
                            Icons.verified,
                            size: 20,
                            color: accent,
                          ),
                          const SizedBox(width: 8),
                          const Text(
                            'MERIT Score',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: Colors.white70,
                            ),
                          ),
                        ],
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [accent, accent.withValues(alpha: 0.7)],
                          ),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          meritScore!.toStringAsFixed(1),
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],

              // Branding footer
              if (showBranding) ...[
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      width: 24,
                      height: 24,
                      decoration: BoxDecoration(
                        color: accent,
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: const Center(
                        child: Text(
                          'T',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'technic',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w300,
                        color: Colors.white.withValues(alpha: 0.5),
                        letterSpacing: 1.5,
                      ),
                    ),
                  ],
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
// PREMIUM SOCIAL BUTTON
// =============================================================================

/// Premium social media button
class PremiumSocialButton extends StatefulWidget {
  final SocialPlatform platform;
  final VoidCallback? onTap;
  final bool showLabel;
  final double size;

  const PremiumSocialButton({
    super.key,
    required this.platform,
    this.onTap,
    this.showLabel = true,
    this.size = 56,
  });

  @override
  State<PremiumSocialButton> createState() => _PremiumSocialButtonState();
}

class _PremiumSocialButtonState extends State<PremiumSocialButton>
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
        animation: _scaleAnimation,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Icon container
                Container(
                  width: widget.size,
                  height: widget.size,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        widget.platform.color,
                        widget.platform.color.withValues(alpha: 0.7),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(widget.size * 0.28),
                    boxShadow: [
                      BoxShadow(
                        color: widget.platform.color.withValues(alpha: 0.3),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Icon(
                    widget.platform.icon,
                    size: widget.size * 0.45,
                    color: Colors.white,
                  ),
                ),
                // Label
                if (widget.showLabel) ...[
                  const SizedBox(height: 8),
                  Text(
                    widget.platform.label,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      color: Colors.white.withValues(alpha: 0.7),
                    ),
                  ),
                ],
              ],
            ),
          );
        },
      ),
    );
  }
}

// =============================================================================
// PREMIUM SHARE SHEET
// =============================================================================

/// Premium share options bottom sheet
class PremiumShareSheet extends StatelessWidget {
  final String? title;
  final String? message;
  final String? url;
  final Widget? preview;
  final List<SocialPlatform> platforms;
  final ValueChanged<SocialPlatform>? onPlatformTap;
  final VoidCallback? onCopyLink;

  const PremiumShareSheet({
    super.key,
    this.title,
    this.message,
    this.url,
    this.preview,
    this.platforms = const [
      SocialPlatform.twitter,
      SocialPlatform.facebook,
      SocialPlatform.linkedin,
      SocialPlatform.whatsapp,
      SocialPlatform.telegram,
      SocialPlatform.email,
      SocialPlatform.sms,
      SocialPlatform.more,
    ],
    this.onPlatformTap,
    this.onCopyLink,
  });

  /// Show premium share sheet
  static Future<void> show({
    required BuildContext context,
    String? title,
    String? message,
    String? url,
    Widget? preview,
    List<SocialPlatform>? platforms,
    ValueChanged<SocialPlatform>? onPlatformTap,
    VoidCallback? onCopyLink,
  }) {
    return showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => PremiumShareSheet(
        title: title,
        message: message,
        url: url,
        preview: preview,
        platforms: platforms ?? const [
          SocialPlatform.twitter,
          SocialPlatform.facebook,
          SocialPlatform.linkedin,
          SocialPlatform.whatsapp,
          SocialPlatform.telegram,
          SocialPlatform.email,
          SocialPlatform.sms,
          SocialPlatform.more,
        ],
        onPlatformTap: onPlatformTap,
        onCopyLink: onCopyLink,
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
            padding: const EdgeInsets.all(20),
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
                const SizedBox(height: 20),

                // Title
                Text(
                  title ?? 'Share',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),

                // Preview
                if (preview != null) ...[
                  const SizedBox(height: 20),
                  preview!,
                ],

                // Copy link section
                if (url != null) ...[
                  const SizedBox(height: 20),
                  PremiumCopyLink(
                    url: url!,
                    onCopy: onCopyLink,
                  ),
                ],

                const SizedBox(height: 24),

                // Social platforms grid
                GridView.count(
                  crossAxisCount: 4,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  mainAxisSpacing: 16,
                  crossAxisSpacing: 16,
                  childAspectRatio: 0.85,
                  children: platforms.map((platform) {
                    return PremiumSocialButton(
                      platform: platform,
                      onTap: () {
                        onPlatformTap?.call(platform);
                        if (platform != SocialPlatform.copy) {
                          Navigator.pop(context);
                        }
                      },
                    );
                  }).toList(),
                ),

                const SizedBox(height: 8),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM COPY LINK
// =============================================================================

/// Premium copy link button with feedback
class PremiumCopyLink extends StatefulWidget {
  final String url;
  final VoidCallback? onCopy;

  const PremiumCopyLink({
    super.key,
    required this.url,
    this.onCopy,
  });

  @override
  State<PremiumCopyLink> createState() => _PremiumCopyLinkState();
}

class _PremiumCopyLinkState extends State<PremiumCopyLink> {
  bool _copied = false;

  void _copyToClipboard() async {
    await Clipboard.setData(ClipboardData(text: widget.url));
    HapticFeedback.mediumImpact();
    setState(() {
      _copied = true;
    });
    widget.onCopy?.call();

    // Reset after 2 seconds
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) {
        setState(() {
          _copied = false;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: _copyToClipboard,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: _copied
              ? AppColors.successGreen.withValues(alpha: 0.15)
              : Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: _copied
                ? AppColors.successGreen.withValues(alpha: 0.3)
                : Colors.white.withValues(alpha: 0.1),
          ),
        ),
        child: Row(
          children: [
            // Link icon
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: _copied
                    ? AppColors.successGreen.withValues(alpha: 0.2)
                    : Colors.white.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(
                _copied ? Icons.check : Icons.link,
                size: 20,
                color: _copied ? AppColors.successGreen : Colors.white60,
              ),
            ),
            const SizedBox(width: 12),

            // URL text
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _copied ? 'Link copied!' : 'Copy link',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: _copied ? AppColors.successGreen : Colors.white,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    widget.url,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white.withValues(alpha: 0.5),
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),

            // Copy icon
            Icon(
              _copied ? Icons.check_circle : Icons.copy,
              size: 22,
              color: _copied
                  ? AppColors.successGreen
                  : AppColors.primaryBlue,
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// EXPORT FORMAT ENUM
// =============================================================================

/// Export format types
enum ExportFormat {
  pdf,
  csv,
  excel,
  image,
  json,
}

/// Export format data
extension ExportFormatData on ExportFormat {
  String get label {
    switch (this) {
      case ExportFormat.pdf:
        return 'PDF';
      case ExportFormat.csv:
        return 'CSV';
      case ExportFormat.excel:
        return 'Excel';
      case ExportFormat.image:
        return 'Image';
      case ExportFormat.json:
        return 'JSON';
    }
  }

  String get description {
    switch (this) {
      case ExportFormat.pdf:
        return 'Export as PDF document';
      case ExportFormat.csv:
        return 'Export as comma-separated values';
      case ExportFormat.excel:
        return 'Export as Excel spreadsheet';
      case ExportFormat.image:
        return 'Export as PNG image';
      case ExportFormat.json:
        return 'Export as JSON data';
    }
  }

  IconData get icon {
    switch (this) {
      case ExportFormat.pdf:
        return Icons.picture_as_pdf;
      case ExportFormat.csv:
        return Icons.table_chart;
      case ExportFormat.excel:
        return Icons.grid_on;
      case ExportFormat.image:
        return Icons.image;
      case ExportFormat.json:
        return Icons.code;
    }
  }

  Color get color {
    switch (this) {
      case ExportFormat.pdf:
        return const Color(0xFFE53935);
      case ExportFormat.csv:
        return const Color(0xFF43A047);
      case ExportFormat.excel:
        return const Color(0xFF1E7145);
      case ExportFormat.image:
        return const Color(0xFF1976D2);
      case ExportFormat.json:
        return const Color(0xFFFFA000);
    }
  }
}

// =============================================================================
// PREMIUM EXPORT OPTIONS
// =============================================================================

/// Premium export options selector
class PremiumExportOptions extends StatelessWidget {
  final List<ExportFormat> formats;
  final ExportFormat? selectedFormat;
  final ValueChanged<ExportFormat>? onFormatSelected;
  final String? title;

  const PremiumExportOptions({
    super.key,
    this.formats = const [
      ExportFormat.pdf,
      ExportFormat.csv,
      ExportFormat.excel,
      ExportFormat.image,
    ],
    this.selectedFormat,
    this.onFormatSelected,
    this.title,
  });

  /// Show export options sheet
  static Future<ExportFormat?> show({
    required BuildContext context,
    List<ExportFormat>? formats,
    ExportFormat? selectedFormat,
    String? title,
  }) {
    return showModalBottomSheet<ExportFormat>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => _PremiumExportSheet(
        formats: formats ?? const [
          ExportFormat.pdf,
          ExportFormat.csv,
          ExportFormat.excel,
          ExportFormat.image,
        ],
        selectedFormat: selectedFormat,
        title: title,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (title != null) ...[
          Text(
            title!,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 16),
        ],
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: formats.map((format) {
            final isSelected = format == selectedFormat;
            return _buildFormatChip(format, isSelected);
          }).toList(),
        ),
      ],
    );
  }

  Widget _buildFormatChip(ExportFormat format, bool isSelected) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onFormatSelected?.call(format);
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          gradient: isSelected
              ? LinearGradient(
                  colors: [format.color, format.color.withValues(alpha: 0.7)],
                )
              : null,
          color: isSelected ? null : Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected
                ? format.color
                : Colors.white.withValues(alpha: 0.1),
          ),
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: format.color.withValues(alpha: 0.3),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ]
              : null,
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              format.icon,
              size: 20,
              color: isSelected ? Colors.white : format.color,
            ),
            const SizedBox(width: 8),
            Text(
              format.label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
                color: isSelected ? Colors.white : Colors.white70,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Export options bottom sheet
class _PremiumExportSheet extends StatelessWidget {
  final List<ExportFormat> formats;
  final ExportFormat? selectedFormat;
  final String? title;

  const _PremiumExportSheet({
    required this.formats,
    this.selectedFormat,
    this.title,
  });

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
            padding: const EdgeInsets.all(20),
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
                const SizedBox(height: 20),

                // Title
                Text(
                  title ?? 'Export As',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),

                // Format options
                ...formats.map((format) {
                  final isSelected = format == selectedFormat;
                  return _buildFormatOption(context, format, isSelected);
                }),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildFormatOption(
      BuildContext context, ExportFormat format, bool isSelected) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        Navigator.pop(context, format);
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isSelected
              ? format.color.withValues(alpha: 0.15)
              : Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: isSelected
                ? format.color
                : Colors.white.withValues(alpha: 0.1),
          ),
        ),
        child: Row(
          children: [
            // Icon
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: format.color.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                format.icon,
                size: 24,
                color: format.color,
              ),
            ),
            const SizedBox(width: 14),

            // Label and description
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    format.label,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
                      color: isSelected ? format.color : Colors.white,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    format.description,
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.white.withValues(alpha: 0.5),
                    ),
                  ),
                ],
              ),
            ),

            // Checkmark
            if (isSelected)
              Icon(
                Icons.check_circle,
                size: 24,
                color: format.color,
              ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM INVITE CARD
// =============================================================================

/// Premium invite friends card
class PremiumInviteCard extends StatelessWidget {
  final String? referralCode;
  final String? referralLink;
  final int? inviteCount;
  final VoidCallback? onShare;
  final VoidCallback? onCopyCode;

  const PremiumInviteCard({
    super.key,
    this.referralCode,
    this.referralLink,
    this.inviteCount,
    this.onShare,
    this.onCopyCode,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                AppColors.primaryBlue.withValues(alpha: 0.2),
                AppColors.primaryBlue.withValues(alpha: 0.1),
              ],
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: AppColors.primaryBlue.withValues(alpha: 0.3),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          AppColors.primaryBlue,
                          AppColors.primaryBlue.withValues(alpha: 0.7),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: const Icon(
                      Icons.card_giftcard,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Invite Friends',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                        ),
                        Text(
                          'Share Technic with friends',
                          style: TextStyle(
                            fontSize: 13,
                            color: Colors.white.withValues(alpha: 0.7),
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (inviteCount != null)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: AppColors.successGreen.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        '$inviteCount invited',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: AppColors.successGreen,
                        ),
                      ),
                    ),
                ],
              ),

              // Referral code
              if (referralCode != null) ...[
                const SizedBox(height: 20),
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.1),
                    ),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Your referral code',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.white.withValues(alpha: 0.6),
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            referralCode!,
                            style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w800,
                              color: Colors.white,
                              letterSpacing: 2,
                            ),
                          ),
                        ],
                      ),
                      GestureDetector(
                        onTap: () {
                          HapticFeedback.lightImpact();
                          onCopyCode?.call();
                        },
                        child: Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: AppColors.primaryBlue.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Icon(
                            Icons.copy,
                            size: 20,
                            color: AppColors.primaryBlue,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],

              // Share button
              const SizedBox(height: 16),
              GestureDetector(
                onTap: () {
                  HapticFeedback.lightImpact();
                  onShare?.call();
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        AppColors.primaryBlue,
                        AppColors.primaryBlue.withValues(alpha: 0.8),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: AppColors.primaryBlue.withValues(alpha: 0.3),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: const Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.share,
                        color: Colors.white,
                        size: 20,
                      ),
                      SizedBox(width: 8),
                      Text(
                        'Share Invite Link',
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
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
// PREMIUM REFERRAL BANNER
// =============================================================================

/// Premium referral program banner
class PremiumReferralBanner extends StatelessWidget {
  final String title;
  final String subtitle;
  final String? reward;
  final VoidCallback? onTap;
  final VoidCallback? onDismiss;

  const PremiumReferralBanner({
    super.key,
    this.title = 'Refer & Earn',
    this.subtitle = 'Invite friends and earn rewards',
    this.reward,
    this.onTap,
    this.onDismiss,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap?.call();
      },
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  AppColors.successGreen.withValues(alpha: 0.2),
                  AppColors.successGreen.withValues(alpha: 0.1),
                ],
              ),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: AppColors.successGreen.withValues(alpha: 0.3),
              ),
            ),
            child: Row(
              children: [
                // Icon
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        AppColors.successGreen,
                        AppColors.successGreen.withValues(alpha: 0.7),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.card_giftcard,
                    color: Colors.white,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 14),

                // Content
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Text(
                            title,
                            style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                            ),
                          ),
                          if (reward != null) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 2,
                              ),
                              decoration: BoxDecoration(
                                color: AppColors.successGreen,
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: Text(
                                reward!,
                                style: const TextStyle(
                                  fontSize: 10,
                                  fontWeight: FontWeight.w700,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                          ],
                        ],
                      ),
                      const SizedBox(height: 2),
                      Text(
                        subtitle,
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.white.withValues(alpha: 0.7),
                        ),
                      ),
                    ],
                  ),
                ),

                // Arrow or dismiss
                if (onDismiss != null)
                  GestureDetector(
                    onTap: () {
                      HapticFeedback.lightImpact();
                      onDismiss?.call();
                    },
                    child: Icon(
                      Icons.close,
                      size: 20,
                      color: Colors.white.withValues(alpha: 0.5),
                    ),
                  )
                else
                  Icon(
                    Icons.chevron_right,
                    size: 24,
                    color: Colors.white.withValues(alpha: 0.5),
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
// PREMIUM QR CODE
// =============================================================================

/// Premium QR code display
class PremiumQRCode extends StatelessWidget {
  final String data;
  final double size;
  final String? label;
  final Color? foregroundColor;
  final Color? backgroundColor;

  const PremiumQRCode({
    super.key,
    required this.data,
    this.size = 200,
    this.label,
    this.foregroundColor,
    this.backgroundColor,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // QR code container
        ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(
              width: size + 32,
              height: size + 32,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: backgroundColor ?? Colors.white,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.2),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: CustomPaint(
                size: Size(size, size),
                painter: _QRCodePainter(
                  data: data,
                  foregroundColor: foregroundColor ?? Colors.black,
                ),
              ),
            ),
          ),
        ),

        // Label
        if (label != null) ...[
          const SizedBox(height: 16),
          Text(
            label!,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: Colors.white.withValues(alpha: 0.7),
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ],
    );
  }
}

/// Simple QR code painter (placeholder pattern)
class _QRCodePainter extends CustomPainter {
  final String data;
  final Color foregroundColor;

  _QRCodePainter({
    required this.data,
    required this.foregroundColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = foregroundColor
      ..style = PaintingStyle.fill;

    final moduleCount = 25;
    final moduleSize = size.width / moduleCount;

    // Generate pattern based on data hash
    final hash = data.hashCode;
    final random = math.Random(hash);

    // Draw finder patterns (corners)
    _drawFinderPattern(canvas, paint, 0, 0, moduleSize);
    _drawFinderPattern(canvas, paint, (moduleCount - 7) * moduleSize, 0, moduleSize);
    _drawFinderPattern(canvas, paint, 0, (moduleCount - 7) * moduleSize, moduleSize);

    // Draw random modules
    for (int row = 0; row < moduleCount; row++) {
      for (int col = 0; col < moduleCount; col++) {
        // Skip finder pattern areas
        if ((row < 8 && col < 8) ||
            (row < 8 && col > moduleCount - 9) ||
            (row > moduleCount - 9 && col < 8)) {
          continue;
        }

        if (random.nextBool()) {
          canvas.drawRect(
            Rect.fromLTWH(
              col * moduleSize,
              row * moduleSize,
              moduleSize - 1,
              moduleSize - 1,
            ),
            paint,
          );
        }
      }
    }
  }

  void _drawFinderPattern(
      Canvas canvas, Paint paint, double x, double y, double moduleSize) {
    // Outer square
    canvas.drawRect(
      Rect.fromLTWH(x, y, moduleSize * 7, moduleSize * 7),
      paint,
    );

    // White inner
    final whitePaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;
    canvas.drawRect(
      Rect.fromLTWH(
        x + moduleSize,
        y + moduleSize,
        moduleSize * 5,
        moduleSize * 5,
      ),
      whitePaint,
    );

    // Center square
    canvas.drawRect(
      Rect.fromLTWH(
        x + moduleSize * 2,
        y + moduleSize * 2,
        moduleSize * 3,
        moduleSize * 3,
      ),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant _QRCodePainter oldDelegate) {
    return data != oldDelegate.data ||
        foregroundColor != oldDelegate.foregroundColor;
  }
}

// =============================================================================
// PREMIUM TESTIMONIAL CARD
// =============================================================================

/// Premium testimonial/review card
class PremiumTestimonialCard extends StatelessWidget {
  final String quote;
  final String authorName;
  final String? authorTitle;
  final String? avatarUrl;
  final int? rating;
  final Color? accentColor;

  const PremiumTestimonialCard({
    super.key,
    required this.quote,
    required this.authorName,
    this.authorTitle,
    this.avatarUrl,
    this.rating,
    this.accentColor,
  });

  @override
  Widget build(BuildContext context) {
    final accent = accentColor ?? AppColors.primaryBlue;

    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withValues(alpha: 0.08),
                Colors.white.withValues(alpha: 0.04),
              ],
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Quote icon
              Icon(
                Icons.format_quote,
                size: 32,
                color: accent.withValues(alpha: 0.5),
              ),
              const SizedBox(height: 12),

              // Quote text
              Text(
                quote,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                  color: Colors.white.withValues(alpha: 0.9),
                  height: 1.5,
                  fontStyle: FontStyle.italic,
                ),
              ),

              // Rating
              if (rating != null) ...[
                const SizedBox(height: 16),
                Row(
                  children: List.generate(5, (index) {
                    return Icon(
                      index < rating! ? Icons.star : Icons.star_border,
                      size: 18,
                      color: index < rating!
                          ? AppColors.warningOrange
                          : Colors.white.withValues(alpha: 0.3),
                    );
                  }),
                ),
              ],

              const SizedBox(height: 16),

              // Author info
              Row(
                children: [
                  // Avatar
                  Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [accent, accent.withValues(alpha: 0.7)],
                      ),
                      shape: BoxShape.circle,
                    ),
                    child: avatarUrl != null
                        ? ClipOval(
                            child: Image.network(
                              avatarUrl!,
                              fit: BoxFit.cover,
                              errorBuilder: (_, __, ___) => Center(
                                child: Text(
                                  authorName[0].toUpperCase(),
                                  style: const TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.w700,
                                    color: Colors.white,
                                  ),
                                ),
                              ),
                            ),
                          )
                        : Center(
                            child: Text(
                              authorName[0].toUpperCase(),
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w700,
                                color: Colors.white,
                              ),
                            ),
                          ),
                  ),
                  const SizedBox(width: 12),

                  // Name and title
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        authorName,
                        style: const TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                        ),
                      ),
                      if (authorTitle != null) ...[
                        const SizedBox(height: 2),
                        Text(
                          authorTitle!,
                          style: TextStyle(
                            fontSize: 13,
                            color: Colors.white.withValues(alpha: 0.6),
                          ),
                        ),
                      ],
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
}

// =============================================================================
// PREMIUM SHARE PREVIEW
// =============================================================================

/// Premium share preview card
class PremiumSharePreview extends StatelessWidget {
  final String title;
  final String? description;
  final String? imageUrl;
  final String? url;

  const PremiumSharePreview({
    super.key,
    required this.title,
    this.description,
    this.imageUrl,
    this.url,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(14),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.1),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Image
            if (imageUrl != null)
              Container(
                height: 120,
                width: double.infinity,
                color: Colors.white.withValues(alpha: 0.08),
                child: Image.network(
                  imageUrl!,
                  fit: BoxFit.cover,
                  errorBuilder: (_, __, ___) => Center(
                    child: Icon(
                      Icons.image,
                      size: 40,
                      color: Colors.white.withValues(alpha: 0.3),
                    ),
                  ),
                ),
              ),

            // Content
            Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  if (description != null) ...[
                    const SizedBox(height: 6),
                    Text(
                      description!,
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.white.withValues(alpha: 0.6),
                        height: 1.4,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                  if (url != null) ...[
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(
                          Icons.link,
                          size: 14,
                          color: AppColors.primaryBlue,
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            url!,
                            style: TextStyle(
                              fontSize: 12,
                              color: AppColors.primaryBlue,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
