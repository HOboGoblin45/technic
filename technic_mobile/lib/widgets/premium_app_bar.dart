/// Premium App Bar Widget
///
/// Professional app bar with:
/// - Glass morphism background
/// - Gradient overlay
/// - Animated search bar
/// - Premium action buttons
/// - Smooth transitions
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'dart:ui';

import '../theme/app_colors.dart';

/// Premium app bar with glass morphism and animations
class PremiumAppBar extends StatefulWidget implements PreferredSizeWidget {
  final String title;
  final String? logoAsset;
  final bool showSearch;
  final bool showNotifications;
  final int notificationCount;
  final VoidCallback? onSearchTap;
  final VoidCallback? onNotificationTap;
  final VoidCallback? onProfileTap;
  final ValueChanged<String>? onSearchChanged;
  final VoidCallback? onSearchSubmitted;
  final String? searchHint;
  final List<Widget>? actions;

  const PremiumAppBar({
    super.key,
    this.title = 'technic',
    this.logoAsset,
    this.showSearch = true,
    this.showNotifications = false,
    this.notificationCount = 0,
    this.onSearchTap,
    this.onNotificationTap,
    this.onProfileTap,
    this.onSearchChanged,
    this.onSearchSubmitted,
    this.searchHint = 'Search stocks...',
    this.actions,
  });

  @override
  Size get preferredSize => const Size.fromHeight(70);

  @override
  State<PremiumAppBar> createState() => _PremiumAppBarState();
}

class _PremiumAppBarState extends State<PremiumAppBar>
    with SingleTickerProviderStateMixin {
  bool _isSearchExpanded = false;
  late AnimationController _searchController;
  late Animation<double> _searchAnimation;
  late Animation<double> _fadeAnimation;
  final TextEditingController _searchTextController = TextEditingController();
  final FocusNode _searchFocusNode = FocusNode();

  @override
  void initState() {
    super.initState();

    _searchController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _searchAnimation = CurvedAnimation(
      parent: _searchController,
      curve: Curves.easeOutCubic,
    );

    _fadeAnimation = Tween<double>(begin: 1.0, end: 0.0).animate(
      CurvedAnimation(
        parent: _searchController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeOut),
      ),
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    _searchTextController.dispose();
    _searchFocusNode.dispose();
    super.dispose();
  }

  void _toggleSearch() {
    HapticFeedback.lightImpact();
    setState(() {
      _isSearchExpanded = !_isSearchExpanded;
    });

    if (_isSearchExpanded) {
      _searchController.forward();
      Future.delayed(const Duration(milliseconds: 200), () {
        _searchFocusNode.requestFocus();
      });
    } else {
      _searchController.reverse();
      _searchTextController.clear();
      _searchFocusNode.unfocus();
      widget.onSearchChanged?.call('');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        // Glass morphism gradient
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            const Color(0xFF0A0E27).withOpacity(0.98),
            const Color(0xFF0A0E27).withOpacity(0.95),
          ],
        ),
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ClipRRect(
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: SafeArea(
            bottom: false,
            child: Container(
              height: 70,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  // Logo and Title (animated out when search expanded)
                  AnimatedBuilder(
                    animation: _fadeAnimation,
                    builder: (context, child) {
                      return Opacity(
                        opacity: _fadeAnimation.value,
                        child: Transform.translate(
                          offset: Offset(-20 * _searchAnimation.value, 0),
                          child: _buildLogoSection(),
                        ),
                      );
                    },
                  ),

                  const Spacer(),

                  // Search bar (expandable)
                  if (widget.showSearch) _buildSearchSection(),

                  // Action buttons
                  if (!_isSearchExpanded) ...[
                    if (widget.showNotifications) _buildNotificationButton(),
                    if (widget.actions != null) ...widget.actions!,
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLogoSection() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Logo container with glow
        Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                AppColors.primaryBlue,
                AppColors.primaryBlue.withOpacity(0.8),
              ],
            ),
            borderRadius: BorderRadius.circular(10),
            boxShadow: [
              BoxShadow(
                color: AppColors.primaryBlue.withOpacity(0.3),
                blurRadius: 12,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          alignment: Alignment.center,
          child: widget.logoAsset != null
              ? SvgPicture.asset(
                  widget.logoAsset!,
                  width: 20,
                  height: 20,
                  colorFilter: const ColorFilter.mode(
                    Colors.white,
                    BlendMode.srcIn,
                  ),
                )
              : const Text(
                  'tQ',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w900,
                    color: Colors.white,
                    letterSpacing: -0.5,
                  ),
                ),
        ),
        const SizedBox(width: 12),

        // Title with gradient text effect
        ShaderMask(
          shaderCallback: (bounds) => LinearGradient(
            colors: [
              Colors.white,
              Colors.white.withOpacity(0.8),
            ],
          ).createShader(bounds),
          child: Text(
            widget.title,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w300,
              letterSpacing: 2.0,
              color: Colors.white,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSearchSection() {
    return AnimatedBuilder(
      animation: _searchAnimation,
      builder: (context, child) {
        final expandedWidth = MediaQuery.of(context).size.width - 80;
        final collapsedWidth = 44.0;
        final currentWidth =
            collapsedWidth + (expandedWidth - collapsedWidth) * _searchAnimation.value;

        return Container(
          width: currentWidth,
          height: 40,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withOpacity(0.08 + 0.04 * _searchAnimation.value),
                Colors.white.withOpacity(0.04 + 0.02 * _searchAnimation.value),
              ],
            ),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: Colors.white.withOpacity(0.1 + 0.1 * _searchAnimation.value),
              width: 1,
            ),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: BackdropFilter(
              filter: ImageFilter.blur(
                sigmaX: 10 * _searchAnimation.value,
                sigmaY: 10 * _searchAnimation.value,
              ),
              child: Row(
                children: [
                  // Search icon / button
                  GestureDetector(
                    onTap: _isSearchExpanded ? null : _toggleSearch,
                    child: Container(
                      width: 44,
                      height: 40,
                      alignment: Alignment.center,
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 200),
                        child: Icon(
                          _isSearchExpanded ? Icons.search : Icons.search,
                          key: ValueKey(_isSearchExpanded),
                          size: 20,
                          color: _isSearchExpanded
                              ? AppColors.primaryBlue
                              : Colors.white.withOpacity(0.7),
                        ),
                      ),
                    ),
                  ),

                  // Search text field (visible when expanded)
                  if (_searchAnimation.value > 0.3)
                    Expanded(
                      child: Opacity(
                        opacity: ((_searchAnimation.value - 0.3) / 0.7).clamp(0.0, 1.0),
                        child: TextField(
                          controller: _searchTextController,
                          focusNode: _searchFocusNode,
                          style: const TextStyle(
                            fontSize: 14,
                            color: Colors.white,
                            fontWeight: FontWeight.w500,
                          ),
                          decoration: InputDecoration(
                            hintText: widget.searchHint,
                            hintStyle: TextStyle(
                              fontSize: 14,
                              color: Colors.white.withOpacity(0.4),
                              fontWeight: FontWeight.w400,
                            ),
                            border: InputBorder.none,
                            contentPadding: EdgeInsets.zero,
                            isDense: true,
                          ),
                          onChanged: widget.onSearchChanged,
                          onSubmitted: (_) => widget.onSearchSubmitted?.call(),
                          textInputAction: TextInputAction.search,
                        ),
                      ),
                    ),

                  // Close button (visible when expanded)
                  if (_searchAnimation.value > 0.5)
                    Opacity(
                      opacity: ((_searchAnimation.value - 0.5) / 0.5).clamp(0.0, 1.0),
                      child: GestureDetector(
                        onTap: _toggleSearch,
                        child: Container(
                          width: 36,
                          height: 40,
                          alignment: Alignment.center,
                          child: Icon(
                            Icons.close,
                            size: 18,
                            color: Colors.white.withOpacity(0.6),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildNotificationButton() {
    return Padding(
      padding: const EdgeInsets.only(left: 8),
      child: GestureDetector(
        onTap: () {
          HapticFeedback.lightImpact();
          widget.onNotificationTap?.call();
        },
        child: Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withOpacity(0.08),
                Colors.white.withOpacity(0.04),
              ],
            ),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: Colors.white.withOpacity(0.1),
              width: 1,
            ),
          ),
          child: Stack(
            alignment: Alignment.center,
            children: [
              Icon(
                Icons.notifications_outlined,
                size: 20,
                color: Colors.white.withOpacity(0.8),
              ),
              if (widget.notificationCount > 0)
                Positioned(
                  right: 6,
                  top: 6,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                    constraints: const BoxConstraints(minWidth: 16, minHeight: 14),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFFFF3B5C), Color(0xFFE6284A)],
                      ),
                      borderRadius: BorderRadius.circular(7),
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.dangerRed.withOpacity(0.4),
                          blurRadius: 4,
                          offset: const Offset(0, 1),
                        ),
                      ],
                    ),
                    child: Text(
                      widget.notificationCount > 9 ? '9+' : widget.notificationCount.toString(),
                      style: const TextStyle(
                        fontSize: 9,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                      ),
                      textAlign: TextAlign.center,
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

/// Premium action button for app bar
class PremiumAppBarAction extends StatelessWidget {
  final IconData icon;
  final VoidCallback? onTap;
  final int? badgeCount;
  final bool showBadge;

  const PremiumAppBarAction({
    super.key,
    required this.icon,
    this.onTap,
    this.badgeCount,
    this.showBadge = false,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 8),
      child: GestureDetector(
        onTap: () {
          HapticFeedback.lightImpact();
          onTap?.call();
        },
        child: Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withOpacity(0.08),
                Colors.white.withOpacity(0.04),
              ],
            ),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: Colors.white.withOpacity(0.1),
              width: 1,
            ),
          ),
          child: Stack(
            alignment: Alignment.center,
            children: [
              Icon(
                icon,
                size: 20,
                color: Colors.white.withOpacity(0.8),
              ),
              if (showBadge || (badgeCount != null && badgeCount! > 0))
                Positioned(
                  right: 6,
                  top: 6,
                  child: Container(
                    padding: EdgeInsets.symmetric(
                      horizontal: badgeCount != null ? 4 : 0,
                      vertical: badgeCount != null ? 1 : 0,
                    ),
                    constraints: BoxConstraints(
                      minWidth: badgeCount != null ? 16 : 8,
                      minHeight: badgeCount != null ? 14 : 8,
                    ),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFFFF3B5C), Color(0xFFE6284A)],
                      ),
                      borderRadius: BorderRadius.circular(badgeCount != null ? 7 : 4),
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.dangerRed.withOpacity(0.4),
                          blurRadius: 4,
                          offset: const Offset(0, 1),
                        ),
                      ],
                    ),
                    child: badgeCount != null
                        ? Text(
                            badgeCount! > 9 ? '9+' : badgeCount.toString(),
                            style: const TextStyle(
                              fontSize: 9,
                              fontWeight: FontWeight.w800,
                              color: Colors.white,
                            ),
                            textAlign: TextAlign.center,
                          )
                        : null,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Simple premium app bar without search (for inner pages)
class SimplePremiumAppBar extends StatelessWidget implements PreferredSizeWidget {
  final String title;
  final bool showBackButton;
  final VoidCallback? onBackTap;
  final List<Widget>? actions;

  const SimplePremiumAppBar({
    super.key,
    required this.title,
    this.showBackButton = true,
    this.onBackTap,
    this.actions,
  });

  @override
  Size get preferredSize => const Size.fromHeight(60);

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            const Color(0xFF0A0E27).withOpacity(0.98),
            const Color(0xFF0A0E27).withOpacity(0.95),
          ],
        ),
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
      ),
      child: ClipRRect(
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: SafeArea(
            bottom: false,
            child: Container(
              height: 60,
              padding: const EdgeInsets.symmetric(horizontal: 8),
              child: Row(
                children: [
                  // Back button
                  if (showBackButton)
                    GestureDetector(
                      onTap: () {
                        HapticFeedback.lightImpact();
                        if (onBackTap != null) {
                          onBackTap!();
                        } else {
                          Navigator.of(context).pop();
                        }
                      },
                      child: Container(
                        width: 40,
                        height: 40,
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              Colors.white.withOpacity(0.08),
                              Colors.white.withOpacity(0.04),
                            ],
                          ),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: Colors.white.withOpacity(0.1),
                            width: 1,
                          ),
                        ),
                        child: Icon(
                          Icons.arrow_back_ios_new,
                          size: 18,
                          color: Colors.white.withOpacity(0.8),
                        ),
                      ),
                    ),

                  // Title
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      child: Text(
                        title,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                          letterSpacing: -0.3,
                        ),
                        textAlign: showBackButton ? TextAlign.left : TextAlign.center,
                      ),
                    ),
                  ),

                  // Actions
                  if (actions != null) ...actions!,
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
