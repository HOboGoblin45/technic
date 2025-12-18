/// Premium Chart Section Widget
/// 
/// Professional stock chart with:
/// - Glass morphism container
/// - Timeframe selector (1D, 1W, 1M, 3M, 1Y, ALL)
/// - Interactive touch with crosshair
/// - Gradient fill under line
/// - Price tooltip
/// - Smooth animations
library;

import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'dart:ui';

import '../../../models/symbol_detail.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/formatters.dart';

/// Timeframe options for chart
enum ChartTimeframe {
  oneDay('1D'),
  oneWeek('1W'),
  oneMonth('1M'),
  threeMonths('3M'),
  oneYear('1Y'),
  all('ALL');

  final String label;
  const ChartTimeframe(this.label);
}

/// Premium chart section with glass morphism and interactions
class PremiumChartSection extends StatefulWidget {
  final List<PricePoint> history;
  final String symbol;
  final double? currentPrice;

  const PremiumChartSection({
    super.key,
    required this.history,
    required this.symbol,
    this.currentPrice,
  });

  @override
  State<PremiumChartSection> createState() => _PremiumChartSectionState();
}

class _PremiumChartSectionState extends State<PremiumChartSection>
    with SingleTickerProviderStateMixin {
  ChartTimeframe _selectedTimeframe = ChartTimeframe.oneMonth;
  int? _touchedIndex;
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    
    // Setup animations
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOut,
    ));

    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOutCubic,
    ));

    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  List<PricePoint> get _filteredHistory {
    if (widget.history.isEmpty) return [];
    
    final now = DateTime.now();
    DateTime cutoffDate;

    switch (_selectedTimeframe) {
      case ChartTimeframe.oneDay:
        cutoffDate = now.subtract(const Duration(days: 1));
        break;
      case ChartTimeframe.oneWeek:
        cutoffDate = now.subtract(const Duration(days: 7));
        break;
      case ChartTimeframe.oneMonth:
        cutoffDate = now.subtract(const Duration(days: 30));
        break;
      case ChartTimeframe.threeMonths:
        cutoffDate = now.subtract(const Duration(days: 90));
        break;
      case ChartTimeframe.oneYear:
        cutoffDate = now.subtract(const Duration(days: 365));
        break;
      case ChartTimeframe.all:
        return widget.history;
    }

    return widget.history
        .where((point) => point.date.isAfter(cutoffDate))
        .toList();
  }

  bool get _isPositive {
    final filtered = _filteredHistory;
    if (filtered.isEmpty) return true;
    return filtered.last.close >= filtered.first.close;
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: SlideTransition(
        position: _slideAnimation,
        child: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                AppColors.primaryBlue.withOpacity(0.1),
                AppColors.primaryBlue.withOpacity(0.05),
              ],
            ),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(
              color: Colors.white.withOpacity(0.1),
              width: 1,
            ),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Header with timeframe selector
                    _buildHeader(),
                    const SizedBox(height: 20),
                    
                    // Chart
                    SizedBox(
                      height: 280,
                      child: _buildChart(),
                    ),
                    
                    // Touch indicator
                    if (_touchedIndex != null) ...[
                      const SizedBox(height: 12),
                      _buildTouchIndicator(),
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

  Widget _buildHeader() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        // Title
        const Text(
          'Price Chart',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        
        // Timeframe selector
        Container(
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.05),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: Colors.white.withOpacity(0.1),
              width: 1,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: ChartTimeframe.values.map((timeframe) {
              final isSelected = timeframe == _selectedTimeframe;
              return GestureDetector(
                onTap: () {
                  setState(() {
                    _selectedTimeframe = timeframe;
                    _touchedIndex = null;
                  });
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? AppColors.primaryBlue.withOpacity(0.3)
                        : Colors.transparent,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    timeframe.label,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
                      color: isSelected
                          ? AppColors.primaryBlue
                          : Colors.white.withOpacity(0.6),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  Widget _buildChart() {
    final filtered = _filteredHistory;
    
    if (filtered.isEmpty) {
      return Center(
        child: Text(
          'No data available',
          style: TextStyle(
            fontSize: 14,
            color: Colors.white.withOpacity(0.5),
          ),
        ),
      );
    }

    final spots = filtered.asMap().entries.map((entry) {
      return FlSpot(entry.key.toDouble(), entry.value.close);
    }).toList();

    final minY = filtered.map((p) => p.close).reduce((a, b) => a < b ? a : b);
    final maxY = filtered.map((p) => p.close).reduce((a, b) => a > b ? a : b);
    final padding = (maxY - minY) * 0.1;

    return LineChart(
      LineChartData(
        minY: minY - padding,
        maxY: maxY + padding,
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            curveSmoothness: 0.3,
            color: _isPositive ? AppColors.successGreen : AppColors.dangerRed,
            barWidth: 2.5,
            isStrokeCapRound: true,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  (_isPositive ? AppColors.successGreen : AppColors.dangerRed)
                      .withOpacity(0.3),
                  (_isPositive ? AppColors.successGreen : AppColors.dangerRed)
                      .withOpacity(0.0),
                ],
              ),
            ),
          ),
        ],
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 50,
              getTitlesWidget: (value, meta) {
                return Text(
                  formatCurrency(value),
                  style: TextStyle(
                    fontSize: 11,
                    color: Colors.white.withOpacity(0.5),
                    fontWeight: FontWeight.w600,
                  ),
                );
              },
            ),
          ),
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              interval: (spots.length / 4).ceilToDouble(),
              getTitlesWidget: (value, meta) {
                final index = value.toInt();
                if (index < 0 || index >= filtered.length) {
                  return const SizedBox.shrink();
                }
                
                final date = filtered[index].date;
                String label;
                
                switch (_selectedTimeframe) {
                  case ChartTimeframe.oneDay:
                    label = '${date.hour}:${date.minute.toString().padLeft(2, '0')}';
                    break;
                  case ChartTimeframe.oneWeek:
                  case ChartTimeframe.oneMonth:
                    label = '${date.month}/${date.day}';
                    break;
                  default:
                    label = '${date.month}/${date.year.toString().substring(2)}';
                }
                
                return Text(
                  label,
                  style: TextStyle(
                    fontSize: 10,
                    color: Colors.white.withOpacity(0.5),
                    fontWeight: FontWeight.w600,
                  ),
                );
              },
            ),
          ),
        ),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: (maxY - minY) / 4,
          getDrawingHorizontalLine: (value) {
            return FlLine(
              color: Colors.white.withOpacity(0.05),
              strokeWidth: 1,
            );
          },
        ),
        borderData: FlBorderData(show: false),
        lineTouchData: LineTouchData(
          enabled: true,
          touchCallback: (FlTouchEvent event, LineTouchResponse? response) {
            if (response == null || response.lineBarSpots == null) {
              setState(() => _touchedIndex = null);
              return;
            }
            
            if (event is FlTapUpEvent || event is FlPanEndEvent) {
              setState(() => _touchedIndex = null);
              return;
            }
            
            setState(() {
              _touchedIndex = response.lineBarSpots!.first.spotIndex;
            });
          },
          touchTooltipData: LineTouchTooltipData(
            tooltipBgColor: AppColors.primaryBlue.withOpacity(0.9),
            tooltipRoundedRadius: 8,
            tooltipPadding: const EdgeInsets.symmetric(
              horizontal: 12,
              vertical: 8,
            ),
            getTooltipItems: (List<LineBarSpot> touchedSpots) {
              return touchedSpots.map((spot) {
                final index = spot.spotIndex;
                if (index < 0 || index >= filtered.length) {
                  return null;
                }
                
                final point = filtered[index];
                return LineTooltipItem(
                  '${formatCurrency(point.close)}\n${formatDate(point.date)}',
                  const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    fontSize: 12,
                  ),
                );
              }).toList();
            },
          ),
          getTouchedSpotIndicator: (LineChartBarData barData, List<int> spotIndexes) {
            return spotIndexes.map((index) {
              return TouchedSpotIndicatorData(
                FlLine(
                  color: AppColors.primaryBlue.withOpacity(0.5),
                  strokeWidth: 2,
                  dashArray: [5, 5],
                ),
                FlDotData(
                  show: true,
                  getDotPainter: (spot, percent, barData, index) {
                    return FlDotCirclePainter(
                      radius: 6,
                      color: AppColors.primaryBlue,
                      strokeWidth: 2,
                      strokeColor: Colors.white,
                    );
                  },
                ),
              );
            }).toList();
          },
        ),
      ),
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
    );
  }

  Widget _buildTouchIndicator() {
    if (_touchedIndex == null) return const SizedBox.shrink();
    
    final filtered = _filteredHistory;
    if (_touchedIndex! < 0 || _touchedIndex! >= filtered.length) {
      return const SizedBox.shrink();
    }
    
    final point = filtered[_touchedIndex!];
    final change = point.close - filtered.first.close;
    final changePct = (change / filtered.first.close) * 100;
    final isPositive = change >= 0;
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Colors.white.withOpacity(0.1),
          width: 1,
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                formatDate(point.date),
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.white.withOpacity(0.6),
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                formatCurrency(point.close),
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
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
              color: (isPositive ? AppColors.successGreen : AppColors.dangerRed)
                  .withOpacity(0.2),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: isPositive ? AppColors.successGreen : AppColors.dangerRed,
                width: 1,
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  isPositive ? Icons.arrow_upward : Icons.arrow_downward,
                  size: 14,
                  color: isPositive ? AppColors.successGreen : AppColors.dangerRed,
                ),
                const SizedBox(width: 4),
                Text(
                  '${isPositive ? '+' : ''}${changePct.toStringAsFixed(2)}%',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: isPositive ? AppColors.successGreen : AppColors.dangerRed,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
