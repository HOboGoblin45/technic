/// Price Chart Widget
/// 
/// Interactive candlestick chart with volume bars using fl_chart.
/// Features:
/// - Candlestick display
/// - Volume bars
/// - Timeframe selector
/// - Touch interactions
/// - Price labels
library;

import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../../models/symbol_detail.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';
import '../../../utils/formatters.dart';

enum ChartTimeframe {
  oneDay('1D', 1),
  oneWeek('1W', 7),
  oneMonth('1M', 30),
  threeMonths('3M', 90),
  oneYear('1Y', 365);

  final String label;
  final int days;
  const ChartTimeframe(this.label, this.days);
}

class PriceChartWidget extends StatefulWidget {
  final List<PricePoint> history;
  final String symbol;

  const PriceChartWidget({
    super.key,
    required this.history,
    required this.symbol,
  });

  @override
  State<PriceChartWidget> createState() => _PriceChartWidgetState();
}

class _PriceChartWidgetState extends State<PriceChartWidget> {
  ChartTimeframe _selectedTimeframe = ChartTimeframe.threeMonths;
  int? _touchedIndex;

  List<PricePoint> get _filteredHistory {
    if (widget.history.isEmpty) return [];
    
    final now = DateTime.now();
    final cutoff = now.subtract(Duration(days: _selectedTimeframe.days));
    
    return widget.history
        .where((point) => point.date.isAfter(cutoff))
        .toList();
  }

  @override
  Widget build(BuildContext context) {
    if (widget.history.isEmpty) {
      return _buildEmptyState();
    }

    final data = _filteredHistory;
    if (data.isEmpty) {
      return _buildEmptyState();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header with timeframe selector
        _buildHeader(),
        const SizedBox(height: 16),
        
        // Candlestick chart
        SizedBox(
          height: 250,
          child: _buildCandlestickChart(data),
        ),
        
        const SizedBox(height: 16),
        
        // Volume chart
        SizedBox(
          height: 80,
          child: _buildVolumeChart(data),
        ),
        
        // Touch info
        if (_touchedIndex != null && _touchedIndex! < data.length)
          _buildTouchInfo(data[_touchedIndex!]),
      ],
    );
  }

  Widget _buildHeader() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'PRICE CHART',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.2,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '${_filteredHistory.length} data points',
              style: const TextStyle(
                fontSize: 12,
                color: Colors.white54,
              ),
            ),
          ],
        ),
        _buildTimeframeSelector(),
      ],
    );
  }

  Widget _buildTimeframeSelector() {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.05),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: ChartTimeframe.values.map((timeframe) {
          final isSelected = _selectedTimeframe == timeframe;
          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedTimeframe = timeframe;
                _touchedIndex = null;
              });
            },
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: isSelected ? AppColors.primaryBlue : Colors.transparent,
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                timeframe.label,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: isSelected ? Colors.white : Colors.white60,
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildCandlestickChart(List<PricePoint> data) {
    final minPrice = data.map((p) => p.low).reduce((a, b) => a < b ? a : b);
    final maxPrice = data.map((p) => p.high).reduce((a, b) => a > b ? a : b);
    final priceRange = maxPrice - minPrice;
    final padding = priceRange * 0.1;

    return LineChart(
      LineChartData(
        minY: minPrice - padding,
        maxY: maxPrice + padding,
        minX: 0,
        maxX: data.length.toDouble() - 1,
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: priceRange / 4,
          getDrawingHorizontalLine: (value) {
            return FlLine(
              color: Colors.white.withValues(alpha: 0.1),
              strokeWidth: 1,
            );
          },
        ),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 50,
              getTitlesWidget: (value, meta) {
                return Text(
                  formatCurrency(value),
                  style: const TextStyle(
                    fontSize: 10,
                    color: Colors.white54,
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
              interval: (data.length / 4).ceilToDouble(),
              getTitlesWidget: (value, meta) {
                final index = value.toInt();
                if (index < 0 || index >= data.length) return const SizedBox();
                
                final date = data[index].date;
                return Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    '${date.month}/${date.day}',
                    style: const TextStyle(
                      fontSize: 10,
                      color: Colors.white54,
                    ),
                  ),
                );
              },
            ),
          ),
        ),
        borderData: FlBorderData(show: false),
        lineTouchData: LineTouchData(
          enabled: true,
          touchCallback: (event, response) {
            if (response?.lineBarSpots != null && response!.lineBarSpots!.isNotEmpty) {
              setState(() {
                _touchedIndex = response.lineBarSpots!.first.x.toInt();
              });
            }
          },
          getTouchedSpotIndicator: (barData, spotIndexes) {
            return spotIndexes.map((index) {
              return TouchedSpotIndicatorData(
                FlLine(
                  color: AppColors.primaryBlue,
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
          touchTooltipData: LineTouchTooltipData(
            tooltipBgColor: tone(Colors.black, 0.8),
            tooltipRoundedRadius: 8,
            tooltipPadding: const EdgeInsets.all(8),
            getTooltipItems: (touchedSpots) {
              return touchedSpots.map((spot) {
                final index = spot.x.toInt();
                if (index < 0 || index >= data.length) return null;
                
                final point = data[index];
                return LineTooltipItem(
                  '${formatCurrency(point.close)}\n${formatDate(point.date)}',
                  const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                  ),
                );
              }).toList();
            },
          ),
        ),
        lineBarsData: [
          // Candlestick bodies (represented as bars)
          LineChartBarData(
            spots: data.asMap().entries.map((entry) {
              return FlSpot(entry.key.toDouble(), entry.value.close);
            }).toList(),
            isCurved: false,
            color: AppColors.primaryBlue,
            barWidth: 2,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                colors: [
                  AppColors.primaryBlue.withValues(alpha: 0.3),
                  AppColors.primaryBlue.withValues(alpha: 0.0),
                ],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
          ),
        ],
        extraLinesData: ExtraLinesData(
          horizontalLines: _touchedIndex != null && _touchedIndex! < data.length
              ? [
                  HorizontalLine(
                    y: data[_touchedIndex!].close,
                    color: AppColors.primaryBlue.withValues(alpha: 0.5),
                    strokeWidth: 1,
                    dashArray: [5, 5],
                  ),
                ]
              : [],
        ),
      ),
    );
  }

  Widget _buildVolumeChart(List<PricePoint> data) {
    final maxVolume = data.map((p) => p.volume).reduce((a, b) => a > b ? a : b);

    return BarChart(
      BarChartData(
        maxY: maxVolume.toDouble(),
        minY: 0,
        barGroups: data.asMap().entries.map((entry) {
          final index = entry.key;
          final point = entry.value;
          final isUp = point.close >= point.open;
          
          return BarChartGroupData(
            x: index,
            barRods: [
              BarChartRodData(
                toY: point.volume.toDouble(),
                color: isUp
                    ? Colors.green.withValues(alpha: 0.5)
                    : Colors.red.withValues(alpha: 0.5),
                width: 2,
                borderRadius: BorderRadius.zero,
              ),
            ],
          );
        }).toList(),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: maxVolume / 2,
          getDrawingHorizontalLine: (value) {
            return FlLine(
              color: Colors.white.withValues(alpha: 0.1),
              strokeWidth: 1,
            );
          },
        ),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 50,
              getTitlesWidget: (value, meta) {
                return Text(
                  formatCompact(value),
                  style: const TextStyle(
                    fontSize: 10,
                    color: Colors.white54,
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
          bottomTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
        ),
        borderData: FlBorderData(show: false),
        barTouchData: BarTouchData(
          enabled: true,
          touchCallback: (event, response) {
            if (response?.spot != null) {
              setState(() {
                _touchedIndex = response!.spot!.touchedBarGroupIndex;
              });
            }
          },
        ),
      ),
    );
  }

  Widget _buildTouchInfo(PricePoint point) {
    final isUp = point.close >= point.open;
    final changeColor = isUp ? Colors.green : Colors.red;
    final change = point.close - point.open;
    final changePct = (change / point.open) * 100;

    return Container(
      margin: const EdgeInsets.only(top: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: tone(Colors.white, 0.1)),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                formatDate(point.date),
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70,
                ),
              ),
              Row(
                children: [
                  Icon(
                    isUp ? Icons.arrow_upward : Icons.arrow_downward,
                    size: 16,
                    color: changeColor,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    '${isUp ? '+' : ''}${changePct.toStringAsFixed(2)}%',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: changeColor,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _buildPriceInfo('Open', point.open),
              _buildPriceInfo('High', point.high, color: Colors.green),
              _buildPriceInfo('Low', point.low, color: Colors.red),
              _buildPriceInfo('Close', point.close),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Volume',
                style: TextStyle(fontSize: 12, color: Colors.white54),
              ),
              Text(
                formatCompact(point.volume.toDouble()),
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPriceInfo(String label, double value, {Color? color}) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 10,
            color: Colors.white54,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          formatCurrency(value),
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: color ?? Colors.white,
          ),
        ),
      ],
    );
  }

  Widget _buildEmptyState() {
    return const Center(
      child: Padding(
        padding: EdgeInsets.all(40),
        child: Column(
          children: [
            Icon(Icons.show_chart, size: 48, color: Colors.white38),
            SizedBox(height: 16),
            Text(
              'No price data available',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white70,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
