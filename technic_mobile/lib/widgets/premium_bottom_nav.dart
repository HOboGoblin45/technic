/// Premium Bottom Navigation Widget
///
/// Professional bottom navigation with:
/// - Glass morphism background
/// - Animated icon transitions
/// - Active indicator with gradient
/// - Haptic feedback
/// - Badge notifications
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui';

import '../theme/app_colors.dart';

/// Navigation item data model
class NavItem {
  final IconData icon;
  final IconData activeIcon;
  final String label;
  final int? badgeCount;
  final bool showBadge;

  const NavItem({
    required this.icon,
    required this.activeIcon,
    required this.label,
    this.badgeCount,
    this.showBadge = false,
  });
}

/// Premium bottom navigation with glass morphism
class PremiumBottomNav extends StatefulWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;
  final List<NavItem> items;
  final bool enableHaptics;

  const PremiumBottomNav({
    super.key,
    required this.currentIndex,
    required this.onTap,
    required this.items,
    this.enableHaptics = true,
  });

  @override
  State<PremiumBottomNav> createState() => _PremiumBottomNavState();
}

class _PremiumBottomNavState extends State<PremiumBottomNav>
    with TickerProviderStateMixin {
  late List<AnimationController> _scaleControllers;
  late List<Animation<double>> _scaleAnimations;
  late AnimationController _indicatorController;
  late Animation<double> _indicatorAnimation;

  int _previousIndex = 0;

  @override
  void initState() {
    super.initState();
    _previousIndex = widget.currentIndex;

    // Scale animations for each nav item
    _scaleControllers = List.generate(
      widget.items.length,
      (index) => AnimationController(
        duration: const Duration(milliseconds: 200),
        vsync: this,
      ),
    );

    _scaleAnimations = _scaleControllers.map((controller) {
      return Tween<double>(begin: 1.0, end: 0.85).animate(
        CurvedAnimation(parent: controller, curve: Curves.easeInOut),
      );
    }).toList();

    // Indicator slide animation
    _indicatorController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _indicatorAnimation = CurvedAnimation(
      parent: _indicatorController,
      curve: Curves.easeOutCubic,
    );
  }

  @override
  void didUpdateWidget(PremiumBottomNav oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.currentIndex != widget.currentIndex) {
      _previousIndex = oldWidget.currentIndex;
      _indicatorController.forward(from: 0);
    }
  }

  @override
  void dispose() {
    for (var controller in _scaleControllers) {
      controller.dispose();
    }
    _indicatorController.dispose();
    super.dispose();
  }

  void _onItemTap(int index) {
    if (widget.enableHaptics) {
      HapticFeedback.lightImpact();
    }

    // Trigger scale animation
    _scaleControllers[index].forward().then((_) {
      _scaleControllers[index].reverse();
    });

    widget.onTap(index);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        // Glass morphism gradient background
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            const Color(0xFF0A0E27).withOpacity(0.95),
            const Color(0xFF0A0E27),
          ],
        ),
        border: Border(
          top: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.4),
            blurRadius: 20,
            offset: const Offset(0, -8),
          ),
        ],
      ),
      child: ClipRRect(
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: SafeArea(
            top: false,
            child: Container(
              height: 75,
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: List.generate(widget.items.length, (index) {
                  return Expanded(
                    child: _buildNavItem(index),
                  );
                }),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(int index) {
    final item = widget.items[index];
    final isSelected = widget.currentIndex == index;

    return AnimatedBuilder(
      animation: _scaleAnimations[index],
      builder: (context, child) {
        return Transform.scale(
          scale: _scaleAnimations[index].value,
          child: GestureDetector(
            onTap: () => _onItemTap(index),
            behavior: HitTestBehavior.opaque,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Icon with indicator
                  Stack(
                    clipBehavior: Clip.none,
                    children: [
                      // Active indicator background
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 250),
                        curve: Curves.easeOutCubic,
                        width: isSelected ? 56 : 40,
                        height: 32,
                        decoration: BoxDecoration(
                          gradient: isSelected
                              ? LinearGradient(
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                  colors: [
                                    AppColors.primaryBlue.withOpacity(0.25),
                                    AppColors.primaryBlue.withOpacity(0.1),
                                  ],
                                )
                              : null,
                          borderRadius: BorderRadius.circular(16),
                          border: isSelected
                              ? Border.all(
                                  color: AppColors.primaryBlue.withOpacity(0.3),
                                  width: 1,
                                )
                              : null,
                        ),
                        child: Center(
                          child: _buildAnimatedIcon(item, isSelected),
                        ),
                      ),

                      // Badge
                      if (item.showBadge || (item.badgeCount != null && item.badgeCount! > 0))
                        Positioned(
                          right: isSelected ? 4 : 0,
                          top: -4,
                          child: _buildBadge(item.badgeCount),
                        ),
                    ],
                  ),

                  const SizedBox(height: 4),

                  // Label
                  AnimatedDefaultTextStyle(
                    duration: const Duration(milliseconds: 200),
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
                      color: isSelected
                          ? AppColors.primaryBlue
                          : Colors.white.withOpacity(0.5),
                      letterSpacing: isSelected ? 0.3 : 0,
                    ),
                    child: Text(item.label),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildAnimatedIcon(NavItem item, bool isSelected) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 250),
      switchInCurve: Curves.easeOutCubic,
      switchOutCurve: Curves.easeInCubic,
      transitionBuilder: (child, animation) {
        return ScaleTransition(
          scale: animation,
          child: FadeTransition(
            opacity: animation,
            child: child,
          ),
        );
      },
      child: Icon(
        isSelected ? item.activeIcon : item.icon,
        key: ValueKey(isSelected),
        size: isSelected ? 24 : 22,
        color: isSelected
            ? AppColors.primaryBlue
            : Colors.white.withOpacity(0.6),
      ),
    );
  }

  Widget _buildBadge(int? count) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: const Duration(milliseconds: 300),
      curve: Curves.elasticOut,
      builder: (context, value, child) {
        return Transform.scale(
          scale: value,
          child: Container(
            padding: EdgeInsets.symmetric(
              horizontal: count != null && count > 9 ? 5 : 4,
              vertical: 2,
            ),
            constraints: const BoxConstraints(
              minWidth: 16,
              minHeight: 16,
            ),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Color(0xFFFF3B5C),
                  Color(0xFFE6284A),
                ],
              ),
              borderRadius: BorderRadius.circular(10),
              boxShadow: [
                BoxShadow(
                  color: AppColors.dangerRed.withOpacity(0.4),
                  blurRadius: 6,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: count != null
                ? Text(
                    count > 99 ? '99+' : count.toString(),
                    style: const TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                    ),
                    textAlign: TextAlign.center,
                  )
                : const SizedBox(width: 6, height: 6),
          ),
        );
      },
    );
  }
}

/// Convenience method to create standard Technic nav items
List<NavItem> createTechnicNavItems({
  int? ideasBadge,
  int? copilotBadge,
  bool showWatchlistBadge = false,
}) {
  return [
    const NavItem(
      icon: Icons.assessment_outlined,
      activeIcon: Icons.assessment,
      label: 'Scan',
    ),
    NavItem(
      icon: Icons.lightbulb_outline,
      activeIcon: Icons.lightbulb,
      label: 'Ideas',
      badgeCount: ideasBadge,
    ),
    NavItem(
      icon: Icons.chat_bubble_outline,
      activeIcon: Icons.chat_bubble,
      label: 'Copilot',
      badgeCount: copilotBadge,
    ),
    NavItem(
      icon: Icons.bookmark_outline,
      activeIcon: Icons.bookmark,
      label: 'Watchlist',
      showBadge: showWatchlistBadge,
    ),
    const NavItem(
      icon: Icons.settings_outlined,
      activeIcon: Icons.settings,
      label: 'Settings',
    ),
  ];
}
