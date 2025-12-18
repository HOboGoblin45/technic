/// Premium Data Tables - Phase 14
///
/// Professional data table components with:
/// - Glass morphism design
/// - Sortable columns with animations
/// - Expandable rows
/// - Pagination controls
/// - Smooth 60fps animations
/// - Haptic feedback
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui';
import 'dart:math' as math;

import '../theme/app_colors.dart';

// =============================================================================
// ENUMS AND MODELS
// =============================================================================

/// Sort direction for table columns
enum SortDirection { ascending, descending, none }

/// Column alignment options
enum ColumnAlignment { left, center, right }

/// Table column definition
class TableColumnDef<T> {
  final String id;
  final String label;
  final double? width;
  final double? flex;
  final bool sortable;
  final ColumnAlignment alignment;
  final Widget Function(T item)? cellBuilder;
  final Comparable Function(T item)? sortValue;
  final String? tooltip;

  const TableColumnDef({
    required this.id,
    required this.label,
    this.width,
    this.flex,
    this.sortable = true,
    this.alignment = ColumnAlignment.left,
    this.cellBuilder,
    this.sortValue,
    this.tooltip,
  });
}

/// Pagination info
class PaginationInfo {
  final int currentPage;
  final int totalPages;
  final int totalItems;
  final int itemsPerPage;

  const PaginationInfo({
    required this.currentPage,
    required this.totalPages,
    required this.totalItems,
    required this.itemsPerPage,
  });

  int get startIndex => (currentPage - 1) * itemsPerPage;
  int get endIndex => math.min(startIndex + itemsPerPage, totalItems);
  bool get hasNext => currentPage < totalPages;
  bool get hasPrevious => currentPage > 1;
}

// =============================================================================
// PREMIUM DATA TABLE
// =============================================================================

/// Premium data table with glass morphism and animations
class PremiumDataTable<T> extends StatefulWidget {
  final List<TableColumnDef<T>> columns;
  final List<T> data;
  final String? title;
  final Widget Function(T item)? rowBuilder;
  final void Function(T item)? onRowTap;
  final void Function(T item)? onRowLongPress;
  final String? sortColumnId;
  final SortDirection sortDirection;
  final void Function(String columnId, SortDirection direction)? onSort;
  final bool showHeader;
  final bool showDividers;
  final double rowHeight;
  final bool enableHover;
  final Widget? emptyWidget;
  final bool isLoading;
  final int skeletonRows;
  final PaginationInfo? pagination;
  final void Function(int page)? onPageChange;

  const PremiumDataTable({
    super.key,
    required this.columns,
    required this.data,
    this.title,
    this.rowBuilder,
    this.onRowTap,
    this.onRowLongPress,
    this.sortColumnId,
    this.sortDirection = SortDirection.none,
    this.onSort,
    this.showHeader = true,
    this.showDividers = true,
    this.rowHeight = 64,
    this.enableHover = true,
    this.emptyWidget,
    this.isLoading = false,
    this.skeletonRows = 5,
    this.pagination,
    this.onPageChange,
  });

  @override
  State<PremiumDataTable<T>> createState() => _PremiumDataTableState<T>();
}

class _PremiumDataTableState<T> extends State<PremiumDataTable<T>>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  int? _hoveredIndex;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );
    _fadeAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
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
    return FadeTransition(
      opacity: _fadeAnimation,
      child: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Colors.white.withOpacity(0.06),
              Colors.white.withOpacity(0.02),
            ],
          ),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Title
                if (widget.title != null) _buildTitle(),

                // Header
                if (widget.showHeader) _buildHeader(),

                // Body
                if (widget.isLoading)
                  _buildSkeleton()
                else if (widget.data.isEmpty)
                  _buildEmpty()
                else
                  _buildBody(),

                // Pagination
                if (widget.pagination != null) _buildPagination(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTitle() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 4,
            height: 24,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  AppColors.primaryBlue,
                  AppColors.primaryBlue.withOpacity(0.3),
                ],
              ),
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            widget.title!,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: Colors.white,
              letterSpacing: -0.3,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.03),
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
      ),
      child: Row(
        children: widget.columns.map((column) {
          return _buildHeaderCell(column);
        }).toList(),
      ),
    );
  }

  Widget _buildHeaderCell(TableColumnDef<T> column) {
    final isSorted = widget.sortColumnId == column.id;
    final direction = isSorted ? widget.sortDirection : SortDirection.none;

    Widget content = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          column.label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: isSorted
                ? AppColors.primaryBlue
                : Colors.white.withOpacity(0.6),
            letterSpacing: 0.5,
          ),
        ),
        if (column.sortable) ...[
          const SizedBox(width: 4),
          _buildSortIcon(direction, isSorted),
        ],
      ],
    );

    if (column.tooltip != null) {
      content = Tooltip(
        message: column.tooltip!,
        child: content,
      );
    }

    final Widget cell = column.sortable
        ? GestureDetector(
            onTap: () {
              HapticFeedback.lightImpact();
              _handleSort(column);
            },
            child: MouseRegion(
              cursor: SystemMouseCursors.click,
              child: content,
            ),
          )
        : content;

    return _wrapInFlex(column, cell);
  }

  Widget _buildSortIcon(SortDirection direction, bool isSorted) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      child: Icon(
        direction == SortDirection.ascending
            ? Icons.arrow_upward
            : direction == SortDirection.descending
                ? Icons.arrow_downward
                : Icons.unfold_more,
        size: 14,
        color: isSorted
            ? AppColors.primaryBlue
            : Colors.white.withOpacity(0.4),
        key: ValueKey(direction),
      ),
    );
  }

  void _handleSort(TableColumnDef<T> column) {
    if (!column.sortable || widget.onSort == null) return;

    SortDirection newDirection;
    if (widget.sortColumnId != column.id) {
      newDirection = SortDirection.ascending;
    } else if (widget.sortDirection == SortDirection.ascending) {
      newDirection = SortDirection.descending;
    } else if (widget.sortDirection == SortDirection.descending) {
      newDirection = SortDirection.none;
    } else {
      newDirection = SortDirection.ascending;
    }

    widget.onSort!(column.id, newDirection);
  }

  Widget _buildBody() {
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: widget.data.length,
      itemBuilder: (context, index) {
        return _buildRow(widget.data[index], index);
      },
    );
  }

  Widget _buildRow(T item, int index) {
    final isHovered = _hoveredIndex == index;
    final isEven = index % 2 == 0;

    return MouseRegion(
      onEnter: widget.enableHover ? (_) => setState(() => _hoveredIndex = index) : null,
      onExit: widget.enableHover ? (_) => setState(() => _hoveredIndex = null) : null,
      child: GestureDetector(
        onTap: widget.onRowTap != null
            ? () {
                HapticFeedback.lightImpact();
                widget.onRowTap!(item);
              }
            : null,
        onLongPress: widget.onRowLongPress != null
            ? () {
                HapticFeedback.mediumImpact();
                widget.onRowLongPress!(item);
              }
            : null,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          height: widget.rowHeight,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          decoration: BoxDecoration(
            color: isHovered
                ? Colors.white.withOpacity(0.06)
                : isEven
                    ? Colors.white.withOpacity(0.02)
                    : Colors.transparent,
            border: widget.showDividers
                ? Border(
                    bottom: BorderSide(
                      color: Colors.white.withOpacity(0.05),
                      width: 1,
                    ),
                  )
                : null,
          ),
          child: widget.rowBuilder != null
              ? widget.rowBuilder!(item)
              : Row(
                  children: widget.columns.map((column) {
                    return _buildCell(column, item);
                  }).toList(),
                ),
        ),
      ),
    );
  }

  Widget _buildCell(TableColumnDef<T> column, T item) {
    Widget content;

    if (column.cellBuilder != null) {
      content = column.cellBuilder!(item);
    } else {
      content = const SizedBox.shrink();
    }

    final alignment = switch (column.alignment) {
      ColumnAlignment.left => Alignment.centerLeft,
      ColumnAlignment.center => Alignment.center,
      ColumnAlignment.right => Alignment.centerRight,
    };

    return _wrapInFlex(
      column,
      Align(
        alignment: alignment,
        child: content,
      ),
    );
  }

  Widget _wrapInFlex(TableColumnDef<T> column, Widget child) {
    if (column.width != null) {
      return SizedBox(width: column.width, child: child);
    }
    return Expanded(
      flex: column.flex?.toInt() ?? 1,
      child: child,
    );
  }

  Widget _buildSkeleton() {
    return Column(
      children: List.generate(widget.skeletonRows, (index) {
        return Container(
          height: widget.rowHeight,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          decoration: BoxDecoration(
            border: widget.showDividers
                ? Border(
                    bottom: BorderSide(
                      color: Colors.white.withOpacity(0.05),
                      width: 1,
                    ),
                  )
                : null,
          ),
          child: Row(
            children: widget.columns.map((column) {
              return _wrapInFlex(
                column,
                _SkeletonCell(delay: index * 100),
              );
            }).toList(),
          ),
        );
      }),
    );
  }

  Widget _buildEmpty() {
    return widget.emptyWidget ??
        Container(
          padding: const EdgeInsets.all(40),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.inbox_outlined,
                size: 48,
                color: Colors.white.withOpacity(0.3),
              ),
              const SizedBox(height: 16),
              Text(
                'No data available',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  color: Colors.white.withOpacity(0.5),
                ),
              ),
            ],
          ),
        );
  }

  Widget _buildPagination() {
    return PremiumPagination(
      pagination: widget.pagination!,
      onPageChange: widget.onPageChange,
    );
  }
}

// =============================================================================
// SKELETON CELL
// =============================================================================

class _SkeletonCell extends StatefulWidget {
  final int delay;

  const _SkeletonCell({this.delay = 0});

  @override
  State<_SkeletonCell> createState() => _SkeletonCellState();
}

class _SkeletonCellState extends State<_SkeletonCell>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    Future.delayed(Duration(milliseconds: widget.delay), () {
      if (mounted) {
        _controller.repeat();
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
        return Container(
          height: 16,
          margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment(-1.0 + _controller.value * 2, 0),
              end: Alignment(0.0 + _controller.value * 2, 0),
              colors: [
                Colors.white.withOpacity(0.05),
                Colors.white.withOpacity(0.1),
                Colors.white.withOpacity(0.05),
              ],
            ),
            borderRadius: BorderRadius.circular(4),
          ),
        );
      },
    );
  }
}

// =============================================================================
// PREMIUM PAGINATION
// =============================================================================

/// Premium pagination controls with glass morphism
class PremiumPagination extends StatelessWidget {
  final PaginationInfo pagination;
  final void Function(int page)? onPageChange;
  final bool showPageNumbers;
  final int maxVisiblePages;

  const PremiumPagination({
    super.key,
    required this.pagination,
    this.onPageChange,
    this.showPageNumbers = true,
    this.maxVisiblePages = 5,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.03),
        border: Border(
          top: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Items info
          Text(
            'Showing ${pagination.startIndex + 1}-${pagination.endIndex} of ${pagination.totalItems}',
            style: TextStyle(
              fontSize: 13,
              color: Colors.white.withOpacity(0.5),
            ),
          ),

          // Page controls
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildNavButton(
                icon: Icons.first_page,
                onTap: pagination.hasPrevious ? () => onPageChange?.call(1) : null,
              ),
              const SizedBox(width: 4),
              _buildNavButton(
                icon: Icons.chevron_left,
                onTap: pagination.hasPrevious
                    ? () => onPageChange?.call(pagination.currentPage - 1)
                    : null,
              ),
              if (showPageNumbers) ...[
                const SizedBox(width: 8),
                ..._buildPageNumbers(),
                const SizedBox(width: 8),
              ],
              _buildNavButton(
                icon: Icons.chevron_right,
                onTap: pagination.hasNext
                    ? () => onPageChange?.call(pagination.currentPage + 1)
                    : null,
              ),
              const SizedBox(width: 4),
              _buildNavButton(
                icon: Icons.last_page,
                onTap: pagination.hasNext
                    ? () => onPageChange?.call(pagination.totalPages)
                    : null,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildNavButton({
    required IconData icon,
    VoidCallback? onTap,
  }) {
    final isEnabled = onTap != null;

    return GestureDetector(
      onTap: onTap != null
          ? () {
              HapticFeedback.lightImpact();
              onTap();
            }
          : null,
      child: Container(
        width: 32,
        height: 32,
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(isEnabled ? 0.08 : 0.03),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: Colors.white.withOpacity(isEnabled ? 0.1 : 0.05),
            width: 1,
          ),
        ),
        child: Icon(
          icon,
          size: 18,
          color: Colors.white.withOpacity(isEnabled ? 0.7 : 0.3),
        ),
      ),
    );
  }

  List<Widget> _buildPageNumbers() {
    final pages = <Widget>[];
    final totalPages = pagination.totalPages;
    final currentPage = pagination.currentPage;

    int start = math.max(1, currentPage - maxVisiblePages ~/ 2);
    int end = math.min(totalPages, start + maxVisiblePages - 1);

    if (end - start < maxVisiblePages - 1) {
      start = math.max(1, end - maxVisiblePages + 1);
    }

    if (start > 1) {
      pages.add(_buildPageButton(1));
      if (start > 2) {
        pages.add(_buildEllipsis());
      }
    }

    for (int i = start; i <= end; i++) {
      pages.add(_buildPageButton(i));
    }

    if (end < totalPages) {
      if (end < totalPages - 1) {
        pages.add(_buildEllipsis());
      }
      pages.add(_buildPageButton(totalPages));
    }

    return pages;
  }

  Widget _buildPageButton(int page) {
    final isSelected = page == pagination.currentPage;

    return GestureDetector(
      onTap: !isSelected
          ? () {
              HapticFeedback.lightImpact();
              onPageChange?.call(page);
            }
          : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 32,
        height: 32,
        margin: const EdgeInsets.symmetric(horizontal: 2),
        decoration: BoxDecoration(
          gradient: isSelected
              ? LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    AppColors.primaryBlue,
                    AppColors.primaryBlue.withOpacity(0.8),
                  ],
                )
              : null,
          color: isSelected ? null : Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected
                ? AppColors.primaryBlue.withOpacity(0.5)
                : Colors.white.withOpacity(0.1),
            width: 1,
          ),
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: AppColors.primaryBlue.withOpacity(0.3),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ]
              : null,
        ),
        child: Center(
          child: Text(
            '$page',
            style: TextStyle(
              fontSize: 13,
              fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
              color: isSelected ? Colors.white : Colors.white.withOpacity(0.6),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildEllipsis() {
    return Container(
      width: 32,
      height: 32,
      margin: const EdgeInsets.symmetric(horizontal: 2),
      child: Center(
        child: Text(
          '...',
          style: TextStyle(
            fontSize: 13,
            color: Colors.white.withOpacity(0.4),
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// PREMIUM TABLE CELL WIDGETS
// =============================================================================

/// Text cell with optional formatting
class PremiumTextCell extends StatelessWidget {
  final String text;
  final TextStyle? style;
  final int? maxLines;
  final bool mono;

  const PremiumTextCell({
    super.key,
    required this.text,
    this.style,
    this.maxLines = 1,
    this.mono = false,
  });

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: style ??
          TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w500,
            color: Colors.white.withOpacity(0.9),
            fontFamily: mono ? 'monospace' : null,
          ),
      maxLines: maxLines,
      overflow: TextOverflow.ellipsis,
    );
  }
}

/// Number cell with formatting and color coding
class PremiumNumberCell extends StatelessWidget {
  final double value;
  final String? prefix;
  final String? suffix;
  final int decimals;
  final bool colorCode;
  final bool showSign;
  final TextStyle? style;

  const PremiumNumberCell({
    super.key,
    required this.value,
    this.prefix,
    this.suffix,
    this.decimals = 2,
    this.colorCode = false,
    this.showSign = false,
    this.style,
  });

  @override
  Widget build(BuildContext context) {
    final isPositive = value >= 0;
    final color = colorCode
        ? (isPositive ? AppColors.successGreen : AppColors.dangerRed)
        : Colors.white.withOpacity(0.9);

    final sign = showSign && isPositive ? '+' : '';
    final formattedValue = value.toStringAsFixed(decimals);

    return Text(
      '${prefix ?? ''}$sign$formattedValue${suffix ?? ''}',
      style: style ??
          TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: color,
            fontFamily: 'monospace',
          ),
    );
  }
}

/// Percentage cell with color coding and badge
class PremiumPercentCell extends StatelessWidget {
  final double value;
  final bool showBadge;
  final bool showIcon;
  final int decimals;

  const PremiumPercentCell({
    super.key,
    required this.value,
    this.showBadge = true,
    this.showIcon = true,
    this.decimals = 2,
  });

  @override
  Widget build(BuildContext context) {
    final isPositive = value >= 0;
    final color = isPositive ? AppColors.successGreen : AppColors.dangerRed;
    final sign = isPositive ? '+' : '';

    final content = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (showIcon)
          Icon(
            isPositive ? Icons.arrow_upward : Icons.arrow_downward,
            size: 12,
            color: color,
          ),
        if (showIcon) const SizedBox(width: 4),
        Text(
          '$sign${value.toStringAsFixed(decimals)}%',
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
      ],
    );

    if (!showBadge) return content;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(6),
      ),
      child: content,
    );
  }
}

/// Status badge cell
class PremiumStatusCell extends StatelessWidget {
  final String status;
  final Color? color;
  final IconData? icon;
  final Map<String, Color>? statusColors;

  const PremiumStatusCell({
    super.key,
    required this.status,
    this.color,
    this.icon,
    this.statusColors,
  });

  @override
  Widget build(BuildContext context) {
    final statusColor = color ?? _getStatusColor();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: statusColor.withOpacity(0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: statusColor.withOpacity(0.3),
          width: 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 12, color: statusColor),
            const SizedBox(width: 4),
          ],
          Text(
            status,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: statusColor,
            ),
          ),
        ],
      ),
    );
  }

  Color _getStatusColor() {
    if (statusColors != null && statusColors!.containsKey(status.toLowerCase())) {
      return statusColors![status.toLowerCase()]!;
    }

    switch (status.toLowerCase()) {
      case 'active':
      case 'success':
      case 'complete':
      case 'buy':
        return AppColors.successGreen;
      case 'inactive':
      case 'error':
      case 'failed':
      case 'sell':
        return AppColors.dangerRed;
      case 'pending':
      case 'warning':
      case 'hold':
        return AppColors.warningOrange;
      default:
        return AppColors.primaryBlue;
    }
  }
}

/// Progress cell with bar visualization
class PremiumProgressCell extends StatelessWidget {
  final double value;
  final double maxValue;
  final Color? color;
  final bool showLabel;
  final double height;

  const PremiumProgressCell({
    super.key,
    required this.value,
    this.maxValue = 100,
    this.color,
    this.showLabel = true,
    this.height = 6,
  });

  @override
  Widget build(BuildContext context) {
    final progress = (value / maxValue).clamp(0.0, 1.0);
    final barColor = color ?? AppColors.primaryBlue;

    return Row(
      children: [
        Expanded(
          child: Container(
            height: height,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              borderRadius: BorderRadius.circular(height / 2),
            ),
            child: FractionallySizedBox(
              alignment: Alignment.centerLeft,
              widthFactor: progress,
              child: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [barColor, barColor.withOpacity(0.7)],
                  ),
                  borderRadius: BorderRadius.circular(height / 2),
                  boxShadow: [
                    BoxShadow(
                      color: barColor.withOpacity(0.4),
                      blurRadius: 4,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
        if (showLabel) ...[
          const SizedBox(width: 8),
          Text(
            '${(progress * 100).toInt()}%',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: Colors.white.withOpacity(0.6),
            ),
          ),
        ],
      ],
    );
  }
}

/// Sparkline cell for mini charts
class PremiumSparklineCell extends StatelessWidget {
  final List<double> data;
  final Color? color;
  final double height;

  const PremiumSparklineCell({
    super.key,
    required this.data,
    this.color,
    this.height = 24,
  });

  @override
  Widget build(BuildContext context) {
    if (data.isEmpty) return const SizedBox.shrink();

    final isPositive = data.last >= data.first;
    final lineColor = color ?? (isPositive ? AppColors.successGreen : AppColors.dangerRed);

    return SizedBox(
      height: height,
      child: CustomPaint(
        size: Size.infinite,
        painter: _SparklinePainter(data: data, color: lineColor),
      ),
    );
  }
}

class _SparklinePainter extends CustomPainter {
  final List<double> data;
  final Color color;

  _SparklinePainter({required this.data, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final minValue = data.reduce(math.min);
    final maxValue = data.reduce(math.max);
    final range = maxValue - minValue;

    final path = Path();

    for (int i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final y = range == 0
          ? size.height / 2
          : size.height - ((data[i] - minValue) / range) * size.height;

      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    final paint = Paint()
      ..color = color
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(_SparklinePainter oldDelegate) {
    return oldDelegate.data != data || oldDelegate.color != color;
  }
}

// =============================================================================
// PREMIUM EXPANDABLE TABLE ROW
// =============================================================================

/// Expandable table row with animated expansion
class PremiumExpandableRow extends StatefulWidget {
  final Widget header;
  final Widget expandedContent;
  final bool initiallyExpanded;
  final VoidCallback? onToggle;
  final Color? backgroundColor;

  const PremiumExpandableRow({
    super.key,
    required this.header,
    required this.expandedContent,
    this.initiallyExpanded = false,
    this.onToggle,
    this.backgroundColor,
  });

  @override
  State<PremiumExpandableRow> createState() => _PremiumExpandableRowState();
}

class _PremiumExpandableRowState extends State<PremiumExpandableRow>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _expandAnimation;
  late Animation<double> _rotateAnimation;
  late bool _isExpanded;

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
    _rotateAnimation = Tween<double>(begin: 0, end: 0.5).animate(_expandAnimation);

    if (_isExpanded) {
      _controller.value = 1.0;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _toggle() {
    HapticFeedback.lightImpact();
    setState(() {
      _isExpanded = !_isExpanded;
      if (_isExpanded) {
        _controller.forward();
      } else {
        _controller.reverse();
      }
    });
    widget.onToggle?.call();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: widget.backgroundColor ?? Colors.transparent,
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.05),
            width: 1,
          ),
        ),
      ),
      child: Column(
        children: [
          // Header
          GestureDetector(
            onTap: _toggle,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  Expanded(child: widget.header),
                  RotationTransition(
                    turns: _rotateAnimation,
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      color: Colors.white.withOpacity(0.5),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Expanded content
          SizeTransition(
            sizeFactor: _expandAnimation,
            child: Container(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.02),
              ),
              child: widget.expandedContent,
            ),
          ),
        ],
      ),
    );
  }
}

// =============================================================================
// PREMIUM STOCK TABLE (SPECIALIZED)
// =============================================================================

/// Stock data model for table display
class StockTableData {
  final String ticker;
  final String? name;
  final double price;
  final double change;
  final double changePercent;
  final double volume;
  final double? marketCap;
  final List<double>? sparkline;
  final String? signal;

  const StockTableData({
    required this.ticker,
    this.name,
    required this.price,
    required this.change,
    required this.changePercent,
    required this.volume,
    this.marketCap,
    this.sparkline,
    this.signal,
  });
}

/// Specialized stock table with predefined columns
class PremiumStockTable extends StatelessWidget {
  final List<StockTableData> stocks;
  final void Function(StockTableData stock)? onStockTap;
  final String? sortColumnId;
  final SortDirection sortDirection;
  final void Function(String columnId, SortDirection direction)? onSort;
  final bool showSparkline;
  final bool showMarketCap;
  final PaginationInfo? pagination;
  final void Function(int page)? onPageChange;
  final bool isLoading;

  const PremiumStockTable({
    super.key,
    required this.stocks,
    this.onStockTap,
    this.sortColumnId,
    this.sortDirection = SortDirection.none,
    this.onSort,
    this.showSparkline = true,
    this.showMarketCap = false,
    this.pagination,
    this.onPageChange,
    this.isLoading = false,
  });

  @override
  Widget build(BuildContext context) {
    final columns = <TableColumnDef<StockTableData>>[
      TableColumnDef<StockTableData>(
        id: 'ticker',
        label: 'Symbol',
        flex: 2,
        sortable: true,
        cellBuilder: (stock) => _buildTickerCell(stock),
        sortValue: (stock) => stock.ticker,
      ),
      TableColumnDef<StockTableData>(
        id: 'price',
        label: 'Price',
        flex: 1,
        alignment: ColumnAlignment.right,
        sortable: true,
        cellBuilder: (stock) => PremiumNumberCell(
          value: stock.price,
          prefix: '\$',
          decimals: 2,
        ),
        sortValue: (stock) => stock.price,
      ),
      TableColumnDef<StockTableData>(
        id: 'change',
        label: 'Change',
        flex: 1,
        alignment: ColumnAlignment.right,
        sortable: true,
        cellBuilder: (stock) => PremiumPercentCell(
          value: stock.changePercent,
          showBadge: true,
          showIcon: true,
        ),
        sortValue: (stock) => stock.changePercent,
      ),
      if (showSparkline)
        TableColumnDef<StockTableData>(
          id: 'sparkline',
          label: 'Trend',
          flex: 1,
          sortable: false,
          cellBuilder: (stock) => stock.sparkline != null
              ? PremiumSparklineCell(data: stock.sparkline!)
              : const SizedBox.shrink(),
        ),
      TableColumnDef<StockTableData>(
        id: 'volume',
        label: 'Volume',
        flex: 1,
        alignment: ColumnAlignment.right,
        sortable: true,
        cellBuilder: (stock) => PremiumTextCell(
          text: _formatVolume(stock.volume),
          style: TextStyle(
            fontSize: 13,
            color: Colors.white.withOpacity(0.7),
            fontFamily: 'monospace',
          ),
        ),
        sortValue: (stock) => stock.volume,
      ),
      if (showMarketCap)
        TableColumnDef<StockTableData>(
          id: 'marketCap',
          label: 'Market Cap',
          flex: 1,
          alignment: ColumnAlignment.right,
          sortable: true,
          cellBuilder: (stock) => PremiumTextCell(
            text: stock.marketCap != null ? _formatMarketCap(stock.marketCap!) : '-',
            style: TextStyle(
              fontSize: 13,
              color: Colors.white.withOpacity(0.7),
            ),
          ),
          sortValue: (stock) => stock.marketCap ?? 0,
        ),
    ];

    return PremiumDataTable<StockTableData>(
      columns: columns,
      data: stocks,
      title: 'Stocks',
      sortColumnId: sortColumnId,
      sortDirection: sortDirection,
      onSort: onSort,
      onRowTap: onStockTap,
      pagination: pagination,
      onPageChange: onPageChange,
      isLoading: isLoading,
      rowHeight: 72,
    );
  }

  Widget _buildTickerCell(StockTableData stock) {
    return Row(
      children: [
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                AppColors.primaryBlue.withOpacity(0.2),
                AppColors.primaryBlue.withOpacity(0.1),
              ],
            ),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: AppColors.primaryBlue.withOpacity(0.3),
              width: 1,
            ),
          ),
          child: Center(
            child: Text(
              stock.ticker.length > 2 ? stock.ticker.substring(0, 2) : stock.ticker,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w800,
                color: Colors.white,
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                stock.ticker,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              if (stock.name != null)
                Text(
                  stock.name!,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.white.withOpacity(0.5),
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              if (stock.signal != null)
                Container(
                  margin: const EdgeInsets.only(top: 4),
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: AppColors.successGreen.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    stock.signal!,
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                      color: AppColors.successGreen,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }

  String _formatVolume(double volume) {
    if (volume >= 1e9) {
      return '${(volume / 1e9).toStringAsFixed(1)}B';
    } else if (volume >= 1e6) {
      return '${(volume / 1e6).toStringAsFixed(1)}M';
    } else if (volume >= 1e3) {
      return '${(volume / 1e3).toStringAsFixed(1)}K';
    }
    return volume.toStringAsFixed(0);
  }

  String _formatMarketCap(double marketCap) {
    if (marketCap >= 1e12) {
      return '\$${(marketCap / 1e12).toStringAsFixed(1)}T';
    } else if (marketCap >= 1e9) {
      return '\$${(marketCap / 1e9).toStringAsFixed(1)}B';
    } else if (marketCap >= 1e6) {
      return '\$${(marketCap / 1e6).toStringAsFixed(1)}M';
    }
    return '\$${marketCap.toStringAsFixed(0)}';
  }
}

// =============================================================================
// PREMIUM COMPARISON TABLE
// =============================================================================

/// Row data for comparison table
class ComparisonRowData {
  final String label;
  final List<String> values;
  final List<Color>? highlights;
  final IconData? icon;
  final bool isHeader;

  const ComparisonRowData({
    required this.label,
    required this.values,
    this.highlights,
    this.icon,
    this.isHeader = false,
  });
}

/// Side-by-side comparison table
class PremiumComparisonTable extends StatelessWidget {
  final List<String> headers;
  final List<ComparisonRowData> rows;
  final String? title;
  final List<Color>? headerColors;

  const PremiumComparisonTable({
    super.key,
    required this.headers,
    required this.rows,
    this.title,
    this.headerColors,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.white.withOpacity(0.06),
            Colors.white.withOpacity(0.02),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: Colors.white.withOpacity(0.08),
          width: 1,
        ),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Title
              if (title != null) _buildTitle(),

              // Header
              _buildHeader(),

              // Rows
              ...rows.asMap().entries.map((entry) {
                return _buildRow(entry.value, entry.key);
              }),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTitle() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
      ),
      child: Text(
        title!,
        style: const TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: Colors.white,
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.03),
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
      ),
      child: Row(
        children: [
          const Expanded(flex: 2, child: SizedBox()),
          ...headers.asMap().entries.map((entry) {
            final color = headerColors?[entry.key] ?? AppColors.primaryBlue;
            return Expanded(
              flex: 2,
              child: Text(
                entry.value,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: color,
                ),
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildRow(ComparisonRowData row, int index) {
    final isEven = index % 2 == 0;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: row.isHeader
            ? Colors.white.withOpacity(0.04)
            : isEven
                ? Colors.white.withOpacity(0.02)
                : Colors.transparent,
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.05),
            width: 1,
          ),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            flex: 2,
            child: Row(
              children: [
                if (row.icon != null) ...[
                  Icon(
                    row.icon,
                    size: 16,
                    color: Colors.white.withOpacity(0.5),
                  ),
                  const SizedBox(width: 8),
                ],
                Expanded(
                  child: Text(
                    row.label,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: row.isHeader ? FontWeight.w700 : FontWeight.w500,
                      color: row.isHeader
                          ? Colors.white
                          : Colors.white.withOpacity(0.7),
                    ),
                  ),
                ),
              ],
            ),
          ),
          ...row.values.asMap().entries.map((entry) {
            final highlight = row.highlights?[entry.key];
            return Expanded(
              flex: 2,
              child: Text(
                entry.value,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: highlight ?? Colors.white.withOpacity(0.9),
                ),
              ),
            );
          }),
        ],
      ),
    );
  }
}

// =============================================================================
// PREMIUM LEADERBOARD TABLE
// =============================================================================

/// Leaderboard entry data
class LeaderboardEntry {
  final int rank;
  final String name;
  final String? avatarUrl;
  final String value;
  final double? changePercent;
  final String? badge;
  final Color? badgeColor;

  const LeaderboardEntry({
    required this.rank,
    required this.name,
    required this.value,
    this.avatarUrl,
    this.changePercent,
    this.badge,
    this.badgeColor,
  });
}

/// Leaderboard/ranking table
class PremiumLeaderboardTable extends StatelessWidget {
  final List<LeaderboardEntry> entries;
  final String? title;
  final String valueLabel;
  final void Function(LeaderboardEntry entry)? onEntryTap;
  final bool showRankBadge;
  final bool animate;

  const PremiumLeaderboardTable({
    super.key,
    required this.entries,
    this.title,
    this.valueLabel = 'Score',
    this.onEntryTap,
    this.showRankBadge = true,
    this.animate = true,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.white.withOpacity(0.06),
            Colors.white.withOpacity(0.02),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: Colors.white.withOpacity(0.08),
          width: 1,
        ),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Title
              if (title != null) _buildTitle(),

              // Entries
              ...entries.asMap().entries.map((entry) {
                return _LeaderboardRow(
                  entry: entry.value,
                  index: entry.key,
                  valueLabel: valueLabel,
                  onTap: onEntryTap,
                  showRankBadge: showRankBadge,
                  animate: animate,
                );
              }),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTitle() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  AppColors.warningOrange.withOpacity(0.2),
                  AppColors.warningOrange.withOpacity(0.1),
                ],
              ),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(
              Icons.leaderboard,
              color: AppColors.warningOrange,
              size: 20,
            ),
          ),
          const SizedBox(width: 12),
          Text(
            title!,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}

class _LeaderboardRow extends StatefulWidget {
  final LeaderboardEntry entry;
  final int index;
  final String valueLabel;
  final void Function(LeaderboardEntry)? onTap;
  final bool showRankBadge;
  final bool animate;

  const _LeaderboardRow({
    required this.entry,
    required this.index,
    required this.valueLabel,
    this.onTap,
    this.showRankBadge = true,
    this.animate = true,
  });

  @override
  State<_LeaderboardRow> createState() => _LeaderboardRowState();
}

class _LeaderboardRowState extends State<_LeaderboardRow>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _slideAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );
    _slideAnimation = Tween<double>(begin: 30, end: 0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic),
    );
    _fadeAnimation = CurvedAnimation(parent: _controller, curve: Curves.easeOut);

    if (widget.animate) {
      Future.delayed(Duration(milliseconds: widget.index * 50), () {
        if (mounted) {
          _controller.forward();
        }
      });
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
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Transform.translate(
          offset: Offset(_slideAnimation.value, 0),
          child: Opacity(
            opacity: _fadeAnimation.value,
            child: _buildContent(),
          ),
        );
      },
    );
  }

  Widget _buildContent() {
    final entry = widget.entry;
    final isTopThree = entry.rank <= 3;

    return GestureDetector(
      onTap: widget.onTap != null
          ? () {
              HapticFeedback.lightImpact();
              widget.onTap!(entry);
            }
          : null,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: isTopThree
              ? LinearGradient(
                  begin: Alignment.centerLeft,
                  end: Alignment.centerRight,
                  colors: [
                    _getRankColor(entry.rank).withOpacity(0.1),
                    Colors.transparent,
                  ],
                )
              : null,
          border: Border(
            bottom: BorderSide(
              color: Colors.white.withOpacity(0.05),
              width: 1,
            ),
          ),
        ),
        child: Row(
          children: [
            // Rank
            if (widget.showRankBadge)
              _buildRankBadge()
            else
              SizedBox(
                width: 32,
                child: Text(
                  '${entry.rank}',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Colors.white.withOpacity(0.5),
                  ),
                ),
              ),
            const SizedBox(width: 12),

            // Avatar or initials
            _buildAvatar(),
            const SizedBox(width: 12),

            // Name & badge
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    entry.name,
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                  if (entry.badge != null)
                    Container(
                      margin: const EdgeInsets.only(top: 4),
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: (entry.badgeColor ?? AppColors.primaryBlue).withOpacity(0.15),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        entry.badge!,
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                          color: entry.badgeColor ?? AppColors.primaryBlue,
                        ),
                      ),
                    ),
                ],
              ),
            ),

            // Value & change
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  entry.value,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
                if (entry.changePercent != null)
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        entry.changePercent! >= 0
                            ? Icons.arrow_upward
                            : Icons.arrow_downward,
                        size: 10,
                        color: entry.changePercent! >= 0
                            ? AppColors.successGreen
                            : AppColors.dangerRed,
                      ),
                      const SizedBox(width: 2),
                      Text(
                        '${entry.changePercent! >= 0 ? '+' : ''}${entry.changePercent!.toStringAsFixed(1)}%',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: entry.changePercent! >= 0
                              ? AppColors.successGreen
                              : AppColors.dangerRed,
                        ),
                      ),
                    ],
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRankBadge() {
    final rank = widget.entry.rank;
    final isTopThree = rank <= 3;

    if (isTopThree) {
      return Container(
        width: 32,
        height: 32,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              _getRankColor(rank),
              _getRankColor(rank).withOpacity(0.7),
            ],
          ),
          borderRadius: BorderRadius.circular(8),
          boxShadow: [
            BoxShadow(
              color: _getRankColor(rank).withOpacity(0.4),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Center(
          child: Icon(
            rank == 1
                ? Icons.emoji_events
                : rank == 2
                    ? Icons.military_tech
                    : Icons.workspace_premium,
            size: 16,
            color: Colors.white,
          ),
        ),
      );
    }

    return Container(
      width: 32,
      height: 32,
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Center(
        child: Text(
          '$rank',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w700,
            color: Colors.white.withOpacity(0.6),
          ),
        ),
      ),
    );
  }

  Widget _buildAvatar() {
    final entry = widget.entry;
    final initials = entry.name.split(' ').map((e) => e[0]).take(2).join();

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
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: AppColors.primaryBlue.withOpacity(0.3),
          width: 1,
        ),
      ),
      child: entry.avatarUrl != null
          ? ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Image.network(
                entry.avatarUrl!,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => Center(
                  child: Text(
                    initials,
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                ),
              ),
            )
          : Center(
              child: Text(
                initials,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
            ),
    );
  }

  Color _getRankColor(int rank) {
    switch (rank) {
      case 1:
        return const Color(0xFFFFD700); // Gold
      case 2:
        return const Color(0xFFC0C0C0); // Silver
      case 3:
        return const Color(0xFFCD7F32); // Bronze
      default:
        return Colors.white.withOpacity(0.5);
    }
  }
}

// =============================================================================
// PREMIUM TABLE FOOTER
// =============================================================================

/// Table footer with summary statistics
class PremiumTableFooter extends StatelessWidget {
  final List<FooterStatItem> stats;
  final Widget? trailing;

  const PremiumTableFooter({
    super.key,
    required this.stats,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.03),
        border: Border(
          top: BorderSide(
            color: Colors.white.withOpacity(0.08),
            width: 1,
          ),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: Wrap(
              spacing: 24,
              runSpacing: 12,
              children: stats.map((stat) => _buildStat(stat)).toList(),
            ),
          ),
          if (trailing != null) trailing!,
        ],
      ),
    );
  }

  Widget _buildStat(FooterStatItem stat) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (stat.icon != null) ...[
          Icon(
            stat.icon,
            size: 14,
            color: stat.color ?? Colors.white.withOpacity(0.5),
          ),
          const SizedBox(width: 6),
        ],
        Text(
          stat.label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.white.withOpacity(0.5),
          ),
        ),
        const SizedBox(width: 6),
        Text(
          stat.value,
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: stat.color ?? Colors.white,
          ),
        ),
      ],
    );
  }
}

/// Footer stat item
class FooterStatItem {
  final String label;
  final String value;
  final IconData? icon;
  final Color? color;

  const FooterStatItem({
    required this.label,
    required this.value,
    this.icon,
    this.color,
  });
}
