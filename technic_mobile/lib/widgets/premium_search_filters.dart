/// Premium Search & Filters Widgets
///
/// A collection of premium search and filter components with glass morphism,
/// smooth animations, and professional styling.
///
/// Components:
/// - PremiumSearchBar: Animated search input
/// - PremiumFilterChip: Single/multi-select chips
/// - PremiumQuickFilters: Horizontal scrolling filters
/// - PremiumSortSelector: Sort options dropdown
/// - PremiumRangeSlider: Dual-thumb range slider
/// - PremiumSearchSuggestions: Autocomplete suggestions
/// - PremiumActiveFilters: Active filter tags
/// - PremiumFilterSection: Filter group section
library;

import 'dart:async';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_colors.dart';
import '../utils/helpers.dart';

// =============================================================================
// PREMIUM SEARCH BAR
// =============================================================================

/// Animated search bar with glass morphism
class PremiumSearchBar extends StatefulWidget {
  final String? hint;
  final String? initialValue;
  final ValueChanged<String>? onChanged;
  final ValueChanged<String>? onSubmitted;
  final VoidCallback? onClear;
  final VoidCallback? onVoiceSearch;
  final VoidCallback? onFilterTap;
  final bool showFilter;
  final bool showVoice;
  final bool autofocus;
  final int? activeFilterCount;
  final FocusNode? focusNode;

  const PremiumSearchBar({
    super.key,
    this.hint = 'Search...',
    this.initialValue,
    this.onChanged,
    this.onSubmitted,
    this.onClear,
    this.onVoiceSearch,
    this.onFilterTap,
    this.showFilter = true,
    this.showVoice = false,
    this.autofocus = false,
    this.activeFilterCount,
    this.focusNode,
  });

  @override
  State<PremiumSearchBar> createState() => _PremiumSearchBarState();
}

class _PremiumSearchBarState extends State<PremiumSearchBar>
    with SingleTickerProviderStateMixin {
  late TextEditingController _controller;
  late FocusNode _focusNode;
  late AnimationController _animationController;
  late Animation<double> _scaleAnimation;
  late Animation<double> _glowAnimation;
  bool _isFocused = false;
  bool _hasText = false;

  @override
  void initState() {
    super.initState();

    _controller = TextEditingController(text: widget.initialValue);
    _focusNode = widget.focusNode ?? FocusNode();
    _hasText = _controller.text.isNotEmpty;

    _animationController = AnimationController(
      duration: const Duration(milliseconds: 200),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 1.0, end: 1.02).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeOut),
    );

    _glowAnimation = Tween<double>(begin: 0.0, end: 0.3).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeOut),
    );

    _focusNode.addListener(_onFocusChange);
    _controller.addListener(_onTextChange);

    if (widget.autofocus) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _focusNode.requestFocus();
      });
    }
  }

  void _onFocusChange() {
    setState(() {
      _isFocused = _focusNode.hasFocus;
    });

    if (_isFocused) {
      _animationController.forward();
    } else {
      _animationController.reverse();
    }
  }

  void _onTextChange() {
    final hasText = _controller.text.isNotEmpty;
    if (hasText != _hasText) {
      setState(() {
        _hasText = hasText;
      });
    }
    widget.onChanged?.call(_controller.text);
  }

  void _clearSearch() {
    HapticFeedback.lightImpact();
    _controller.clear();
    widget.onClear?.call();
  }

  @override
  void dispose() {
    _controller.dispose();
    if (widget.focusNode == null) {
      _focusNode.dispose();
    }
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animationController,
      builder: (context, child) {
        return Transform.scale(
          scale: _scaleAnimation.value,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Container(
                height: 56,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Colors.white.withValues(alpha: _isFocused ? 0.12 : 0.08),
                      Colors.white.withValues(alpha: _isFocused ? 0.06 : 0.04),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: _isFocused
                        ? AppColors.primaryBlue.withValues(alpha: 0.5)
                        : Colors.white.withValues(alpha: 0.1),
                    width: _isFocused ? 1.5 : 1,
                  ),
                  boxShadow: _isFocused
                      ? [
                          BoxShadow(
                            color: AppColors.primaryBlue
                                .withValues(alpha: _glowAnimation.value),
                            blurRadius: 20,
                            spreadRadius: 2,
                          ),
                        ]
                      : null,
                ),
                child: Row(
                  children: [
                    // Search icon
                    Padding(
                      padding: const EdgeInsets.only(left: 16),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        child: Icon(
                          Icons.search,
                          size: 22,
                          color: _isFocused
                              ? AppColors.primaryBlue
                              : Colors.white.withValues(alpha: 0.5),
                        ),
                      ),
                    ),

                    // Text field
                    Expanded(
                      child: TextField(
                        controller: _controller,
                        focusNode: _focusNode,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                          color: Colors.white,
                        ),
                        decoration: InputDecoration(
                          hintText: widget.hint,
                          hintStyle: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w400,
                            color: Colors.white.withValues(alpha: 0.4),
                          ),
                          border: InputBorder.none,
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 16,
                          ),
                        ),
                        textInputAction: TextInputAction.search,
                        onSubmitted: widget.onSubmitted,
                      ),
                    ),

                    // Clear button
                    if (_hasText)
                      GestureDetector(
                        onTap: _clearSearch,
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          child: Icon(
                            Icons.close,
                            size: 20,
                            color: Colors.white.withValues(alpha: 0.5),
                          ),
                        ),
                      ),

                    // Voice search button
                    if (widget.showVoice && !_hasText)
                      GestureDetector(
                        onTap: () {
                          HapticFeedback.lightImpact();
                          widget.onVoiceSearch?.call();
                        },
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          child: Icon(
                            Icons.mic,
                            size: 22,
                            color: Colors.white.withValues(alpha: 0.5),
                          ),
                        ),
                      ),

                    // Filter button
                    if (widget.showFilter)
                      GestureDetector(
                        onTap: () {
                          HapticFeedback.lightImpact();
                          widget.onFilterTap?.call();
                        },
                        child: Container(
                          margin: const EdgeInsets.only(right: 8),
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: widget.activeFilterCount != null &&
                                    widget.activeFilterCount! > 0
                                ? AppColors.primaryBlue.withValues(alpha: 0.2)
                                : Colors.white.withValues(alpha: 0.05),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Stack(
                            clipBehavior: Clip.none,
                            children: [
                              Icon(
                                Icons.tune,
                                size: 20,
                                color: widget.activeFilterCount != null &&
                                        widget.activeFilterCount! > 0
                                    ? AppColors.primaryBlue
                                    : Colors.white.withValues(alpha: 0.6),
                              ),
                              if (widget.activeFilterCount != null &&
                                  widget.activeFilterCount! > 0)
                                Positioned(
                                  top: -6,
                                  right: -6,
                                  child: Container(
                                    padding: const EdgeInsets.all(4),
                                    decoration: BoxDecoration(
                                      color: AppColors.primaryBlue,
                                      shape: BoxShape.circle,
                                    ),
                                    child: Text(
                                      '${widget.activeFilterCount}',
                                      style: const TextStyle(
                                        fontSize: 10,
                                        fontWeight: FontWeight.w700,
                                        color: Colors.white,
                                      ),
                                    ),
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
          ),
        );
      },
    );
  }
}

// =============================================================================
// FILTER CHIP STYLE
// =============================================================================

/// Style options for filter chips
enum FilterChipStyle {
  filled,
  outlined,
  gradient,
}

// =============================================================================
// PREMIUM FILTER CHIP
// =============================================================================

/// Premium filter chip with animations
class PremiumFilterChip extends StatefulWidget {
  final String label;
  final bool isSelected;
  final ValueChanged<bool>? onSelected;
  final IconData? icon;
  final Color? selectedColor;
  final FilterChipStyle style;
  final bool showCheckmark;

  const PremiumFilterChip({
    super.key,
    required this.label,
    this.isSelected = false,
    this.onSelected,
    this.icon,
    this.selectedColor,
    this.style = FilterChipStyle.gradient,
    this.showCheckmark = true,
  });

  @override
  State<PremiumFilterChip> createState() => _PremiumFilterChipState();
}

class _PremiumFilterChipState extends State<PremiumFilterChip>
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
    final selectedColor = widget.selectedColor ?? AppColors.primaryBlue;

    return GestureDetector(
      onTapDown: (_) => _controller.forward(),
      onTapUp: (_) {
        _controller.reverse();
        HapticFeedback.lightImpact();
        widget.onSelected?.call(!widget.isSelected);
      },
      onTapCancel: () => _controller.reverse(),
      child: AnimatedBuilder(
        animation: _scaleAnimation,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              padding: EdgeInsets.symmetric(
                horizontal: widget.icon != null ? 12 : 16,
                vertical: 10,
              ),
              decoration: _buildDecoration(selectedColor),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (widget.icon != null) ...[
                    Icon(
                      widget.icon,
                      size: 16,
                      color: widget.isSelected ? Colors.white : Colors.white70,
                    ),
                    const SizedBox(width: 6),
                  ],
                  Text(
                    widget.label,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight:
                          widget.isSelected ? FontWeight.w700 : FontWeight.w600,
                      color: widget.isSelected ? Colors.white : Colors.white70,
                    ),
                  ),
                  if (widget.showCheckmark && widget.isSelected) ...[
                    const SizedBox(width: 6),
                    const Icon(
                      Icons.check_circle,
                      size: 14,
                      color: Colors.white,
                    ),
                  ],
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  BoxDecoration _buildDecoration(Color selectedColor) {
    switch (widget.style) {
      case FilterChipStyle.filled:
        return BoxDecoration(
          color: widget.isSelected
              ? selectedColor
              : Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: widget.isSelected
                ? selectedColor
                : Colors.white.withValues(alpha: 0.1),
            width: 1,
          ),
        );

      case FilterChipStyle.outlined:
        return BoxDecoration(
          color: widget.isSelected
              ? selectedColor.withValues(alpha: 0.15)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: widget.isSelected
                ? selectedColor
                : Colors.white.withValues(alpha: 0.2),
            width: widget.isSelected ? 2 : 1,
          ),
        );

      case FilterChipStyle.gradient:
        return BoxDecoration(
          gradient: widget.isSelected
              ? LinearGradient(
                  colors: [
                    selectedColor,
                    selectedColor.withValues(alpha: 0.7),
                  ],
                )
              : null,
          color:
              widget.isSelected ? null : Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: widget.isSelected
                ? selectedColor
                : Colors.white.withValues(alpha: 0.1),
            width: 1,
          ),
          boxShadow: widget.isSelected
              ? [
                  BoxShadow(
                    color: selectedColor.withValues(alpha: 0.3),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ]
              : null,
        );
    }
  }
}

// =============================================================================
// PREMIUM QUICK FILTERS
// =============================================================================

/// Quick filter data model
class QuickFilterItem {
  final String id;
  final String label;
  final IconData? icon;
  final Color? color;

  const QuickFilterItem({
    required this.id,
    required this.label,
    this.icon,
    this.color,
  });
}

/// Horizontal scrolling quick filters
class PremiumQuickFilters extends StatelessWidget {
  final List<QuickFilterItem> filters;
  final Set<String> selectedIds;
  final ValueChanged<String>? onFilterTap;
  final bool multiSelect;
  final EdgeInsets? padding;

  const PremiumQuickFilters({
    super.key,
    required this.filters,
    required this.selectedIds,
    this.onFilterTap,
    this.multiSelect = false,
    this.padding,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: padding ?? const EdgeInsets.symmetric(horizontal: 16),
      child: Row(
        children: filters.map((filter) {
          final isSelected = selectedIds.contains(filter.id);
          return Padding(
            padding: const EdgeInsets.only(right: 8),
            child: PremiumFilterChip(
              label: filter.label,
              icon: filter.icon,
              isSelected: isSelected,
              selectedColor: filter.color,
              onSelected: (_) => onFilterTap?.call(filter.id),
            ),
          );
        }).toList(),
      ),
    );
  }
}

// =============================================================================
// PREMIUM SORT SELECTOR
// =============================================================================

/// Sort option data model
class SortOptionItem {
  final String id;
  final String label;
  final IconData icon;

  const SortOptionItem({
    required this.id,
    required this.label,
    required this.icon,
  });
}

/// Premium sort selector button
class PremiumSortSelector extends StatefulWidget {
  final List<SortOptionItem> options;
  final String selectedId;
  final bool descending;
  final ValueChanged<String>? onSortChanged;
  final ValueChanged<bool>? onDirectionChanged;

  const PremiumSortSelector({
    super.key,
    required this.options,
    required this.selectedId,
    this.descending = true,
    this.onSortChanged,
    this.onDirectionChanged,
  });

  @override
  State<PremiumSortSelector> createState() => _PremiumSortSelectorState();
}

class _PremiumSortSelectorState extends State<PremiumSortSelector> {
  void _showSortOptions() {
    HapticFeedback.lightImpact();

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => _PremiumSortSheet(
        options: widget.options,
        selectedId: widget.selectedId,
        descending: widget.descending,
        onSortChanged: widget.onSortChanged,
        onDirectionChanged: widget.onDirectionChanged,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final selectedOption = widget.options.firstWhere(
      (o) => o.id == widget.selectedId,
      orElse: () => widget.options.first,
    );

    return GestureDetector(
      onTap: _showSortOptions,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: Colors.white.withValues(alpha: 0.1),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.sort,
                  size: 18,
                  color: AppColors.primaryBlue,
                ),
                const SizedBox(width: 8),
                Text(
                  selectedOption.label,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(width: 6),
                Icon(
                  widget.descending
                      ? Icons.arrow_downward
                      : Icons.arrow_upward,
                  size: 16,
                  color: AppColors.primaryBlue,
                ),
                const SizedBox(width: 4),
                Icon(
                  Icons.keyboard_arrow_down,
                  size: 18,
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

/// Sort options bottom sheet
class _PremiumSortSheet extends StatelessWidget {
  final List<SortOptionItem> options;
  final String selectedId;
  final bool descending;
  final ValueChanged<String>? onSortChanged;
  final ValueChanged<bool>? onDirectionChanged;

  const _PremiumSortSheet({
    required this.options,
    required this.selectedId,
    required this.descending,
    this.onSortChanged,
    this.onDirectionChanged,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: tone(AppColors.darkBackground, 0.95),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
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

              // Header
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Sort By',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                    ),
                  ),
                  // Direction toggle
                  GestureDetector(
                    onTap: () {
                      HapticFeedback.lightImpact();
                      onDirectionChanged?.call(!descending);
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: AppColors.primaryBlue.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            descending
                                ? Icons.arrow_downward
                                : Icons.arrow_upward,
                            size: 18,
                            color: AppColors.primaryBlue,
                          ),
                          const SizedBox(width: 6),
                          Text(
                            descending ? 'Descending' : 'Ascending',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                              color: AppColors.primaryBlue,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),

              // Options
              ...options.map((option) {
                final isSelected = option.id == selectedId;
                return GestureDetector(
                  onTap: () {
                    HapticFeedback.lightImpact();
                    onSortChanged?.call(option.id);
                    Navigator.pop(context);
                  },
                  child: Container(
                    margin: const EdgeInsets.only(bottom: 8),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: isSelected
                          ? AppColors.primaryBlue.withValues(alpha: 0.15)
                          : Colors.white.withValues(alpha: 0.05),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color: isSelected
                            ? AppColors.primaryBlue
                            : Colors.white.withValues(alpha: 0.1),
                        width: isSelected ? 1.5 : 1,
                      ),
                    ),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: isSelected
                                ? AppColors.primaryBlue.withValues(alpha: 0.2)
                                : Colors.white.withValues(alpha: 0.05),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Icon(
                            option.icon,
                            size: 20,
                            color: isSelected
                                ? AppColors.primaryBlue
                                : Colors.white60,
                          ),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Text(
                            option.label,
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight:
                                  isSelected ? FontWeight.w700 : FontWeight.w600,
                              color:
                                  isSelected ? Colors.white : Colors.white70,
                            ),
                          ),
                        ),
                        if (isSelected)
                          Icon(
                            Icons.check_circle,
                            size: 22,
                            color: AppColors.primaryBlue,
                          ),
                      ],
                    ),
                  ),
                );
              }),

              SizedBox(height: MediaQuery.of(context).padding.bottom + 10),
            ],
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM RANGE SLIDER
// =============================================================================

/// Premium dual-thumb range slider
class PremiumRangeSlider extends StatefulWidget {
  final double min;
  final double max;
  final double startValue;
  final double endValue;
  final ValueChanged<RangeValues>? onChanged;
  final ValueChanged<RangeValues>? onChangeEnd;
  final String? label;
  final String Function(double)? formatValue;
  final Color? activeColor;
  final int? divisions;

  const PremiumRangeSlider({
    super.key,
    required this.min,
    required this.max,
    required this.startValue,
    required this.endValue,
    this.onChanged,
    this.onChangeEnd,
    this.label,
    this.formatValue,
    this.activeColor,
    this.divisions,
  });

  @override
  State<PremiumRangeSlider> createState() => _PremiumRangeSliderState();
}

class _PremiumRangeSliderState extends State<PremiumRangeSlider> {
  late RangeValues _values;

  @override
  void initState() {
    super.initState();
    _values = RangeValues(widget.startValue, widget.endValue);
  }

  @override
  void didUpdateWidget(PremiumRangeSlider oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.startValue != oldWidget.startValue ||
        widget.endValue != oldWidget.endValue) {
      _values = RangeValues(widget.startValue, widget.endValue);
    }
  }

  String _formatValue(double value) {
    if (widget.formatValue != null) {
      return widget.formatValue!(value);
    }
    return value.toStringAsFixed(0);
  }

  @override
  Widget build(BuildContext context) {
    final activeColor = widget.activeColor ?? AppColors.primaryBlue;

    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.06),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header with label and values
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  if (widget.label != null)
                    Text(
                      widget.label!,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: Colors.white70,
                      ),
                    ),
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 5,
                        ),
                        decoration: BoxDecoration(
                          color: activeColor.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          _formatValue(_values.start),
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                            color: activeColor,
                          ),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 8),
                        child: Text(
                          '–',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.white.withValues(alpha: 0.5),
                          ),
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 5,
                        ),
                        decoration: BoxDecoration(
                          color: activeColor.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          _formatValue(_values.end),
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                            color: activeColor,
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Slider
              SliderTheme(
                data: SliderThemeData(
                  activeTrackColor: activeColor,
                  inactiveTrackColor: Colors.white.withValues(alpha: 0.1),
                  thumbColor: Colors.white,
                  overlayColor: activeColor.withValues(alpha: 0.2),
                  rangeThumbShape: const RoundRangeSliderThumbShape(
                    enabledThumbRadius: 10,
                  ),
                  trackHeight: 6,
                  rangeTrackShape: const RoundedRectRangeSliderTrackShape(),
                ),
                child: RangeSlider(
                  values: _values,
                  min: widget.min,
                  max: widget.max,
                  divisions: widget.divisions,
                  onChanged: (values) {
                    HapticFeedback.selectionClick();
                    setState(() {
                      _values = values;
                    });
                    widget.onChanged?.call(values);
                  },
                  onChangeEnd: widget.onChangeEnd,
                ),
              ),

              // Min/Max labels
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    _formatValue(widget.min),
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white.withValues(alpha: 0.4),
                    ),
                  ),
                  Text(
                    _formatValue(widget.max),
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white.withValues(alpha: 0.4),
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

// =============================================================================
// PREMIUM SEARCH SUGGESTIONS
// =============================================================================

/// Search suggestion data model
class SearchSuggestion {
  final String id;
  final String text;
  final String? subtitle;
  final IconData? icon;
  final bool isRecent;

  const SearchSuggestion({
    required this.id,
    required this.text,
    this.subtitle,
    this.icon,
    this.isRecent = false,
  });
}

/// Premium search suggestions list
class PremiumSearchSuggestions extends StatelessWidget {
  final List<SearchSuggestion> suggestions;
  final ValueChanged<SearchSuggestion>? onSuggestionTap;
  final ValueChanged<String>? onRemoveRecent;
  final String? highlightText;
  final bool showRecent;

  const PremiumSearchSuggestions({
    super.key,
    required this.suggestions,
    this.onSuggestionTap,
    this.onRemoveRecent,
    this.highlightText,
    this.showRecent = true,
  });

  @override
  Widget build(BuildContext context) {
    if (suggestions.isEmpty) {
      return const SizedBox.shrink();
    }

    final recentSuggestions =
        suggestions.where((s) => s.isRecent).toList();
    final otherSuggestions =
        suggestions.where((s) => !s.isRecent).toList();

    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
        child: Container(
          decoration: BoxDecoration(
            color: tone(AppColors.darkBackground, 0.95),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Recent searches section
              if (showRecent && recentSuggestions.isNotEmpty) ...[
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                  child: Row(
                    children: [
                      Icon(
                        Icons.history,
                        size: 16,
                        color: Colors.white.withValues(alpha: 0.5),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Recent',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: Colors.white.withValues(alpha: 0.5),
                          letterSpacing: 0.5,
                        ),
                      ),
                    ],
                  ),
                ),
                ...recentSuggestions.map((s) => _buildSuggestionTile(s)),
              ],

              // Other suggestions
              if (otherSuggestions.isNotEmpty) ...[
                if (recentSuggestions.isNotEmpty)
                  Divider(
                    height: 1,
                    color: Colors.white.withValues(alpha: 0.1),
                  ),
                ...otherSuggestions.map((s) => _buildSuggestionTile(s)),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSuggestionTile(SearchSuggestion suggestion) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onSuggestionTap?.call(suggestion);
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            // Icon
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(
                suggestion.icon ??
                    (suggestion.isRecent ? Icons.history : Icons.search),
                size: 18,
                color: Colors.white.withValues(alpha: 0.6),
              ),
            ),
            const SizedBox(width: 12),

            // Text content
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHighlightedText(suggestion.text),
                  if (suggestion.subtitle != null) ...[
                    const SizedBox(height: 2),
                    Text(
                      suggestion.subtitle!,
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.white.withValues(alpha: 0.4),
                      ),
                    ),
                  ],
                ],
              ),
            ),

            // Remove button for recent
            if (suggestion.isRecent && onRemoveRecent != null)
              GestureDetector(
                onTap: () {
                  HapticFeedback.lightImpact();
                  onRemoveRecent?.call(suggestion.id);
                },
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: Icon(
                    Icons.close,
                    size: 18,
                    color: Colors.white.withValues(alpha: 0.4),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildHighlightedText(String text) {
    if (highlightText == null || highlightText!.isEmpty) {
      return Text(
        text,
        style: const TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.w500,
          color: Colors.white,
        ),
      );
    }

    final lowerText = text.toLowerCase();
    final lowerHighlight = highlightText!.toLowerCase();
    final startIndex = lowerText.indexOf(lowerHighlight);

    if (startIndex == -1) {
      return Text(
        text,
        style: const TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.w500,
          color: Colors.white,
        ),
      );
    }

    final endIndex = startIndex + highlightText!.length;

    return RichText(
      text: TextSpan(
        style: const TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.w500,
          color: Colors.white70,
        ),
        children: [
          if (startIndex > 0) TextSpan(text: text.substring(0, startIndex)),
          TextSpan(
            text: text.substring(startIndex, endIndex),
            style: TextStyle(
              color: AppColors.primaryBlue,
              fontWeight: FontWeight.w700,
            ),
          ),
          if (endIndex < text.length) TextSpan(text: text.substring(endIndex)),
        ],
      ),
    );
  }
}

// =============================================================================
// PREMIUM ACTIVE FILTERS
// =============================================================================

/// Active filter tag data
class ActiveFilterTag {
  final String id;
  final String label;
  final String? category;

  const ActiveFilterTag({
    required this.id,
    required this.label,
    this.category,
  });
}

/// Premium active filters display
class PremiumActiveFilters extends StatelessWidget {
  final List<ActiveFilterTag> filters;
  final ValueChanged<String>? onRemove;
  final VoidCallback? onClearAll;
  final EdgeInsets? padding;

  const PremiumActiveFilters({
    super.key,
    required this.filters,
    this.onRemove,
    this.onClearAll,
    this.padding,
  });

  @override
  Widget build(BuildContext context) {
    if (filters.isEmpty) {
      return const SizedBox.shrink();
    }

    return Container(
      padding: padding ?? const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.filter_list,
                    size: 16,
                    color: AppColors.primaryBlue,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    'Active Filters (${filters.length})',
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: Colors.white70,
                    ),
                  ),
                ],
              ),
              if (onClearAll != null)
                GestureDetector(
                  onTap: () {
                    HapticFeedback.lightImpact();
                    onClearAll?.call();
                  },
                  child: Text(
                    'Clear All',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: AppColors.dangerRed,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),

          // Filter tags
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: filters.map((filter) => _buildFilterTag(filter)).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildFilterTag(ActiveFilterTag filter) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(10),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 5, sigmaY: 5),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: AppColors.primaryBlue.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: AppColors.primaryBlue.withValues(alpha: 0.3),
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (filter.category != null) ...[
                Text(
                  '${filter.category}: ',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: Colors.white.withValues(alpha: 0.6),
                  ),
                ),
              ],
              Text(
                filter.label,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.primaryBlue,
                ),
              ),
              if (onRemove != null) ...[
                const SizedBox(width: 6),
                GestureDetector(
                  onTap: () {
                    HapticFeedback.lightImpact();
                    onRemove?.call(filter.id);
                  },
                  child: Icon(
                    Icons.close,
                    size: 14,
                    color: AppColors.primaryBlue.withValues(alpha: 0.8),
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
// PREMIUM FILTER SECTION
// =============================================================================

/// Premium filter section with header
class PremiumFilterSection extends StatefulWidget {
  final String title;
  final IconData? icon;
  final Widget child;
  final bool initiallyExpanded;
  final bool collapsible;

  const PremiumFilterSection({
    super.key,
    required this.title,
    this.icon,
    required this.child,
    this.initiallyExpanded = true,
    this.collapsible = true,
  });

  @override
  State<PremiumFilterSection> createState() => _PremiumFilterSectionState();
}

class _PremiumFilterSectionState extends State<PremiumFilterSection>
    with SingleTickerProviderStateMixin {
  late bool _isExpanded;
  late AnimationController _controller;
  late Animation<double> _expandAnimation;
  late Animation<double> _rotationAnimation;

  @override
  void initState() {
    super.initState();
    _isExpanded = widget.initiallyExpanded;

    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _expandAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );

    _rotationAnimation = Tween<double>(begin: 0, end: 0.5).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic),
    );

    if (_isExpanded) {
      _controller.value = 1.0;
    }
  }

  void _toggleExpanded() {
    if (!widget.collapsible) return;

    HapticFeedback.lightImpact();
    setState(() {
      _isExpanded = !_isExpanded;
    });

    if (_isExpanded) {
      _controller.forward();
    } else {
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
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        GestureDetector(
          onTap: _toggleExpanded,
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Row(
              children: [
                if (widget.icon != null) ...[
                  Icon(
                    widget.icon,
                    size: 18,
                    color: AppColors.primaryBlue,
                  ),
                  const SizedBox(width: 10),
                ],
                Text(
                  widget.title,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
                if (widget.collapsible) ...[
                  const Spacer(),
                  RotationTransition(
                    turns: _rotationAnimation,
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      size: 22,
                      color: Colors.white.withValues(alpha: 0.5),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),

        // Content
        SizeTransition(
          sizeFactor: _expandAnimation,
          child: Padding(
            padding: const EdgeInsets.only(top: 12),
            child: widget.child,
          ),
        ),
      ],
    );
  }
}

// =============================================================================
// PREMIUM DATE RANGE PICKER
// =============================================================================

/// Premium date range selector
class PremiumDateRangePicker extends StatelessWidget {
  final DateTimeRange? selectedRange;
  final ValueChanged<DateTimeRange?>? onChanged;
  final DateTime? firstDate;
  final DateTime? lastDate;
  final String? label;
  final List<_DatePreset>? presets;

  const PremiumDateRangePicker({
    super.key,
    this.selectedRange,
    this.onChanged,
    this.firstDate,
    this.lastDate,
    this.label,
    this.presets,
  });

  static final List<_DatePreset> defaultPresets = [
    _DatePreset('Today', () => DateTimeRange(
      start: DateTime.now(),
      end: DateTime.now(),
    )),
    _DatePreset('Last 7 Days', () => DateTimeRange(
      start: DateTime.now().subtract(const Duration(days: 7)),
      end: DateTime.now(),
    )),
    _DatePreset('Last 30 Days', () => DateTimeRange(
      start: DateTime.now().subtract(const Duration(days: 30)),
      end: DateTime.now(),
    )),
    _DatePreset('Last 90 Days', () => DateTimeRange(
      start: DateTime.now().subtract(const Duration(days: 90)),
      end: DateTime.now(),
    )),
    _DatePreset('This Year', () => DateTimeRange(
      start: DateTime(DateTime.now().year, 1, 1),
      end: DateTime.now(),
    )),
  ];

  void _showDatePicker(BuildContext context) async {
    HapticFeedback.lightImpact();

    final result = await showDateRangePicker(
      context: context,
      firstDate: firstDate ?? DateTime(2020),
      lastDate: lastDate ?? DateTime.now(),
      initialDateRange: selectedRange,
      builder: (context, child) {
        return Theme(
          data: ThemeData.dark().copyWith(
            colorScheme: ColorScheme.dark(
              primary: AppColors.primaryBlue,
              onPrimary: Colors.white,
              surface: tone(AppColors.darkBackground, 0.98),
              onSurface: Colors.white,
            ),
          ),
          child: child!,
        );
      },
    );

    if (result != null) {
      onChanged?.call(result);
    }
  }

  String _formatDate(DateTime date) {
    return '${date.month}/${date.day}/${date.year}';
  }

  @override
  Widget build(BuildContext context) {
    final effectivePresets = presets ?? defaultPresets;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (label != null) ...[
          Text(
            label!,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.white70,
            ),
          ),
          const SizedBox(height: 12),
        ],

        // Presets
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: effectivePresets.map((preset) {
            final presetRange = preset.getRange();
            final isSelected = selectedRange != null &&
                selectedRange!.start.day == presetRange.start.day &&
                selectedRange!.end.day == presetRange.end.day;

            return PremiumFilterChip(
              label: preset.label,
              isSelected: isSelected,
              onSelected: (_) {
                onChanged?.call(presetRange);
              },
              style: FilterChipStyle.outlined,
              showCheckmark: false,
            );
          }).toList(),
        ),

        const SizedBox(height: 12),

        // Custom date picker button
        GestureDetector(
          onTap: () => _showDatePicker(context),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.06),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.1),
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.calendar_today,
                      size: 20,
                      color: AppColors.primaryBlue,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        selectedRange != null
                            ? '${_formatDate(selectedRange!.start)} - ${_formatDate(selectedRange!.end)}'
                            : 'Select custom date range',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                          color: selectedRange != null
                              ? Colors.white
                              : Colors.white.withValues(alpha: 0.5),
                        ),
                      ),
                    ),
                    Icon(
                      Icons.chevron_right,
                      size: 20,
                      color: Colors.white.withValues(alpha: 0.4),
                    ),
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

/// Date preset helper class
class _DatePreset {
  final String label;
  final DateTimeRange Function() getRange;

  const _DatePreset(this.label, this.getRange);
}

// =============================================================================
// PREMIUM RESULTS COUNT
// =============================================================================

/// Premium results count display
class PremiumResultsCount extends StatelessWidget {
  final int filteredCount;
  final int totalCount;
  final String? label;

  const PremiumResultsCount({
    super.key,
    required this.filteredCount,
    required this.totalCount,
    this.label,
  });

  @override
  Widget build(BuildContext context) {
    final isFiltered = filteredCount != totalCount;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            isFiltered ? Icons.filter_alt : Icons.list,
            size: 16,
            color: isFiltered
                ? AppColors.primaryBlue
                : Colors.white.withValues(alpha: 0.5),
          ),
          const SizedBox(width: 8),
          if (isFiltered)
            RichText(
              text: TextSpan(
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                ),
                children: [
                  TextSpan(
                    text: '$filteredCount',
                    style: TextStyle(
                      color: AppColors.primaryBlue,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  TextSpan(
                    text: ' of $totalCount',
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.5),
                    ),
                  ),
                  if (label != null)
                    TextSpan(
                      text: ' $label',
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.5),
                      ),
                    ),
                ],
              ),
            )
          else
            Text(
              '$totalCount${label != null ? ' $label' : ''}',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w500,
                color: Colors.white.withValues(alpha: 0.7),
              ),
            ),
        ],
      ),
    );
  }
}
