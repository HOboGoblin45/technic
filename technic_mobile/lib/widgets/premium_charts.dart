/// Premium Charts & Visualizations
///
/// Premium chart components with glass morphism design,
/// smooth animations, and professional styling.
library;

import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../theme/app_colors.dart';

// =============================================================================
// PREMIUM LINE CHART
// =============================================================================

/// Premium animated line chart with gradient fill.
class PremiumLineChart extends StatefulWidget {
  final List<double> data;
  final List<String>? labels;
  final Color? lineColor;
  final Color? fillColor;
  final double strokeWidth;
  final bool showDots;
  final bool showGrid;
  final bool showLabels;
  final bool animate;
  final Duration animationDuration;
  final double height;
  final ValueChanged<int>? onPointTap;

  const PremiumLineChart({
    super.key,
    required this.data,
    this.labels,
    this.lineColor,
    this.fillColor,
    this.strokeWidth = 2.5,
    this.showDots = true,
    this.showGrid = true,
    this.showLabels = true,
    this.animate = true,
    this.animationDuration = const Duration(milliseconds: 1200),
    this.height = 200,
    this.onPointTap,
  });

  @override
  State<PremiumLineChart> createState() => _PremiumLineChartState();
}

class _PremiumLineChartState extends State<PremiumLineChart>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  int? _selectedIndex;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.animationDuration,
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
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
    if (widget.data.isEmpty) {
      return SizedBox(
        height: widget.height,
        child: const Center(
          child: Text(
            'No data available',
            style: TextStyle(color: Colors.white54),
          ),
        ),
      );
    }

    final lineColor = widget.lineColor ?? AppColors.primaryBlue;
    final fillColor = widget.fillColor ?? lineColor.withValues(alpha: 0.2);

    return Column(
      children: [
        SizedBox(
          height: widget.height,
          child: GestureDetector(
            onTapDown: (details) => _handleTap(details),
            child: AnimatedBuilder(
              animation: _animation,
              builder: (context, child) {
                return CustomPaint(
                  painter: _LineChartPainter(
                    data: widget.data,
                    progress: _animation.value,
                    lineColor: lineColor,
                    fillColor: fillColor,
                    strokeWidth: widget.strokeWidth,
                    showDots: widget.showDots,
                    showGrid: widget.showGrid,
                    selectedIndex: _selectedIndex,
                  ),
                  size: Size.infinite,
                );
              },
            ),
          ),
        ),
        if (widget.showLabels && widget.labels != null) ...[
          const SizedBox(height: 8),
          _buildLabels(),
        ],
      ],
    );
  }

  void _handleTap(TapDownDetails details) {
    if (widget.data.isEmpty) return;

    final RenderBox box = context.findRenderObject() as RenderBox;
    final localPosition = details.localPosition;
    final width = box.size.width;
    final segmentWidth = width / (widget.data.length - 1);
    final index = (localPosition.dx / segmentWidth).round().clamp(0, widget.data.length - 1);

    setState(() {
      _selectedIndex = _selectedIndex == index ? null : index;
    });

    HapticFeedback.lightImpact();
    widget.onPointTap?.call(index);
  }

  Widget _buildLabels() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: widget.labels!.asMap().entries.map((entry) {
        final isSelected = _selectedIndex == entry.key;
        return Text(
          entry.value,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.white54,
            fontSize: 10,
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
          ),
        );
      }).toList(),
    );
  }
}

class _LineChartPainter extends CustomPainter {
  final List<double> data;
  final double progress;
  final Color lineColor;
  final Color fillColor;
  final double strokeWidth;
  final bool showDots;
  final bool showGrid;
  final int? selectedIndex;

  _LineChartPainter({
    required this.data,
    required this.progress,
    required this.lineColor,
    required this.fillColor,
    required this.strokeWidth,
    required this.showDots,
    required this.showGrid,
    this.selectedIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final minValue = data.reduce(math.min);
    final maxValue = data.reduce(math.max);
    final range = maxValue - minValue;
    final padding = range * 0.1;
    final effectiveMin = minValue - padding;
    final effectiveMax = maxValue + padding;
    final effectiveRange = effectiveMax - effectiveMin;

    // Draw grid
    if (showGrid) {
      _drawGrid(canvas, size);
    }

    // Calculate points
    final points = <Offset>[];
    for (var i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final normalizedY = (data[i] - effectiveMin) / effectiveRange;
      final y = size.height - (normalizedY * size.height);
      points.add(Offset(x, y));
    }

    // Animate points
    final animatedPoints = points.map((p) {
      final animatedY = size.height - ((size.height - p.dy) * progress);
      return Offset(p.dx, animatedY);
    }).toList();

    // Draw fill gradient
    _drawFill(canvas, size, animatedPoints);

    // Draw line
    _drawLine(canvas, animatedPoints);

    // Draw dots
    if (showDots) {
      _drawDots(canvas, animatedPoints);
    }

    // Draw selected indicator
    if (selectedIndex != null && selectedIndex! < animatedPoints.length) {
      _drawSelectedIndicator(canvas, size, animatedPoints[selectedIndex!], data[selectedIndex!]);
    }
  }

  void _drawGrid(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.06)
      ..strokeWidth = 1;

    // Horizontal lines
    for (var i = 0; i <= 4; i++) {
      final y = size.height * (i / 4);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }
  }

  void _drawFill(Canvas canvas, Size size, List<Offset> points) {
    if (points.isEmpty) return;

    final fillPath = Path()..moveTo(points.first.dx, size.height);
    for (final point in points) {
      fillPath.lineTo(point.dx, point.dy);
    }
    fillPath.lineTo(points.last.dx, size.height);
    fillPath.close();

    final fillPaint = Paint()
      ..shader = LinearGradient(
        colors: [
          fillColor,
          fillColor.withValues(alpha: 0),
        ],
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

    canvas.drawPath(fillPath, fillPaint);
  }

  void _drawLine(Canvas canvas, List<Offset> points) {
    if (points.isEmpty) return;

    final linePath = Path()..moveTo(points.first.dx, points.first.dy);
    for (var i = 1; i < points.length; i++) {
      linePath.lineTo(points[i].dx, points[i].dy);
    }

    // Draw glow
    final glowPaint = Paint()
      ..color = lineColor.withValues(alpha: 0.3)
      ..strokeWidth = strokeWidth + 4
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4);
    canvas.drawPath(linePath, glowPaint);

    // Draw line
    final linePaint = Paint()
      ..color = lineColor
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    canvas.drawPath(linePath, linePaint);
  }

  void _drawDots(Canvas canvas, List<Offset> points) {
    for (var i = 0; i < points.length; i++) {
      final isSelected = i == selectedIndex;
      final dotRadius = isSelected ? 6.0 : 4.0;

      // Outer glow for selected
      if (isSelected) {
        final glowPaint = Paint()
          ..color = lineColor.withValues(alpha: 0.4)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
        canvas.drawCircle(points[i], 12, glowPaint);
      }

      // Dot fill
      final fillPaint = Paint()..color = isSelected ? lineColor : AppColors.darkBackground;
      canvas.drawCircle(points[i], dotRadius, fillPaint);

      // Dot border
      final borderPaint = Paint()
        ..color = lineColor
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke;
      canvas.drawCircle(points[i], dotRadius, borderPaint);
    }
  }

  void _drawSelectedIndicator(Canvas canvas, Size size, Offset point, double value) {
    // Vertical line
    final linePaint = Paint()
      ..color = lineColor.withValues(alpha: 0.3)
      ..strokeWidth = 1;
    canvas.drawLine(Offset(point.dx, 0), Offset(point.dx, size.height), linePaint);

    // Value label
    final textPainter = TextPainter(
      text: TextSpan(
        text: value.toStringAsFixed(2),
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w700,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final labelX = (point.dx - textPainter.width / 2).clamp(0.0, size.width - textPainter.width);
    final labelY = point.dy - 24;

    // Label background
    final labelRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(labelX - 8, labelY - 4, textPainter.width + 16, textPainter.height + 8),
      const Radius.circular(6),
    );
    canvas.drawRRect(labelRect, Paint()..color = lineColor);

    textPainter.paint(canvas, Offset(labelX, labelY));
  }

  @override
  bool shouldRepaint(covariant _LineChartPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.selectedIndex != selectedIndex ||
        oldDelegate.data != data;
  }
}

// =============================================================================
// PREMIUM BAR CHART
// =============================================================================

/// Premium animated bar chart with glass morphism.
class PremiumBarChart extends StatefulWidget {
  final List<double> data;
  final List<String>? labels;
  final List<Color>? colors;
  final Color? defaultColor;
  final bool showValues;
  final bool animate;
  final Duration animationDuration;
  final double height;
  final double barWidth;
  final double spacing;
  final ValueChanged<int>? onBarTap;

  const PremiumBarChart({
    super.key,
    required this.data,
    this.labels,
    this.colors,
    this.defaultColor,
    this.showValues = true,
    this.animate = true,
    this.animationDuration = const Duration(milliseconds: 800),
    this.height = 200,
    this.barWidth = 24,
    this.spacing = 12,
    this.onBarTap,
  });

  @override
  State<PremiumBarChart> createState() => _PremiumBarChartState();
}

class _PremiumBarChartState extends State<PremiumBarChart>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  int? _selectedIndex;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.animationDuration,
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
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
    if (widget.data.isEmpty) {
      return SizedBox(
        height: widget.height,
        child: const Center(
          child: Text('No data', style: TextStyle(color: Colors.white54)),
        ),
      );
    }

    final maxValue = widget.data.reduce(math.max);

    return SizedBox(
      height: widget.height + (widget.labels != null ? 30 : 0),
      child: AnimatedBuilder(
        animation: _animation,
        builder: (context, child) {
          return Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: List.generate(widget.data.length, (index) {
              return _buildBar(index, maxValue);
            }),
          );
        },
      ),
    );
  }

  Widget _buildBar(int index, double maxValue) {
    final value = widget.data[index];
    final normalizedHeight = (value / maxValue) * widget.height * _animation.value;
    final color = widget.colors != null && index < widget.colors!.length
        ? widget.colors![index]
        : widget.defaultColor ?? AppColors.primaryBlue;
    final isSelected = _selectedIndex == index;

    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedIndex = isSelected ? null : index;
        });
        HapticFeedback.lightImpact();
        widget.onBarTap?.call(index);
      },
      child: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          // Value label
          if (widget.showValues)
            AnimatedOpacity(
              opacity: _animation.value,
              duration: const Duration(milliseconds: 200),
              child: Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Text(
                  value.toStringAsFixed(0),
                  style: TextStyle(
                    color: isSelected ? color : Colors.white70,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
          // Bar
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: widget.barWidth,
            height: normalizedHeight.clamp(4.0, widget.height),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  color,
                  color.withValues(alpha: 0.6),
                ],
              ),
              borderRadius: BorderRadius.circular(widget.barWidth / 3),
              border: Border.all(
                color: isSelected ? Colors.white : color.withValues(alpha: 0.5),
                width: isSelected ? 2 : 1,
              ),
              boxShadow: isSelected
                  ? [
                      BoxShadow(
                        color: color.withValues(alpha: 0.4),
                        blurRadius: 12,
                        spreadRadius: 0,
                      ),
                    ]
                  : [
                      BoxShadow(
                        color: color.withValues(alpha: 0.2),
                        blurRadius: 8,
                        offset: const Offset(0, 4),
                      ),
                    ],
            ),
          ),
          // Label
          if (widget.labels != null && index < widget.labels!.length) ...[
            const SizedBox(height: 8),
            SizedBox(
              width: widget.barWidth + widget.spacing,
              child: Text(
                widget.labels![index],
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: isSelected ? Colors.white : Colors.white54,
                  fontSize: 10,
                  fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// =============================================================================
// PREMIUM CANDLESTICK CHART
// =============================================================================

/// Data model for candlestick
class CandleData {
  final DateTime date;
  final double open;
  final double high;
  final double low;
  final double close;
  final double? volume;

  const CandleData({
    required this.date,
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    this.volume,
  });

  bool get isUp => close >= open;
}

/// Premium candlestick chart with animations.
class PremiumCandlestickChart extends StatefulWidget {
  final List<CandleData> data;
  final bool showVolume;
  final bool animate;
  final Duration animationDuration;
  final double height;
  final double volumeHeight;
  final ValueChanged<int>? onCandleTap;

  const PremiumCandlestickChart({
    super.key,
    required this.data,
    this.showVolume = true,
    this.animate = true,
    this.animationDuration = const Duration(milliseconds: 1000),
    this.height = 250,
    this.volumeHeight = 60,
    this.onCandleTap,
  });

  @override
  State<PremiumCandlestickChart> createState() => _PremiumCandlestickChartState();
}

class _PremiumCandlestickChartState extends State<PremiumCandlestickChart>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  int? _selectedIndex;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.animationDuration,
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
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
    if (widget.data.isEmpty) {
      return SizedBox(
        height: widget.height,
        child: const Center(
          child: Text('No data', style: TextStyle(color: Colors.white54)),
        ),
      );
    }

    return Column(
      children: [
        // Candlestick chart
        SizedBox(
          height: widget.height,
          child: GestureDetector(
            onTapDown: _handleTap,
            child: AnimatedBuilder(
              animation: _animation,
              builder: (context, child) {
                return CustomPaint(
                  painter: _CandlestickPainter(
                    data: widget.data,
                    progress: _animation.value,
                    selectedIndex: _selectedIndex,
                  ),
                  size: Size.infinite,
                );
              },
            ),
          ),
        ),
        // Volume chart
        if (widget.showVolume && widget.data.any((c) => c.volume != null)) ...[
          const SizedBox(height: 8),
          SizedBox(
            height: widget.volumeHeight,
            child: AnimatedBuilder(
              animation: _animation,
              builder: (context, child) {
                return CustomPaint(
                  painter: _VolumePainter(
                    data: widget.data,
                    progress: _animation.value,
                    selectedIndex: _selectedIndex,
                  ),
                  size: Size.infinite,
                );
              },
            ),
          ),
        ],
        // Selected info
        if (_selectedIndex != null && _selectedIndex! < widget.data.length)
          _buildSelectedInfo(widget.data[_selectedIndex!]),
      ],
    );
  }

  void _handleTap(TapDownDetails details) {
    if (widget.data.isEmpty) return;

    final RenderBox box = context.findRenderObject() as RenderBox;
    final width = box.size.width;
    final candleWidth = width / widget.data.length;
    final index = (details.localPosition.dx / candleWidth).floor().clamp(0, widget.data.length - 1);

    setState(() {
      _selectedIndex = _selectedIndex == index ? null : index;
    });

    HapticFeedback.lightImpact();
    widget.onCandleTap?.call(index);
  }

  Widget _buildSelectedInfo(CandleData candle) {
    final color = candle.isUp ? AppColors.successGreen : AppColors.dangerRed;

    return Container(
      margin: const EdgeInsets.only(top: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            color.withValues(alpha: 0.15),
            color.withValues(alpha: 0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildInfoItem('O', candle.open, color),
          _buildInfoItem('H', candle.high, AppColors.successGreen),
          _buildInfoItem('L', candle.low, AppColors.dangerRed),
          _buildInfoItem('C', candle.close, color),
          if (candle.volume != null)
            _buildInfoItem('V', candle.volume!, Colors.white70, isVolume: true),
        ],
      ),
    );
  }

  Widget _buildInfoItem(String label, double value, Color color, {bool isVolume = false}) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.5),
            fontSize: 10,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          isVolume ? _formatVolume(value) : '\$${value.toStringAsFixed(2)}',
          style: TextStyle(
            color: color,
            fontSize: 12,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }

  String _formatVolume(double volume) {
    if (volume >= 1e9) return '${(volume / 1e9).toStringAsFixed(1)}B';
    if (volume >= 1e6) return '${(volume / 1e6).toStringAsFixed(1)}M';
    if (volume >= 1e3) return '${(volume / 1e3).toStringAsFixed(1)}K';
    return volume.toStringAsFixed(0);
  }
}

class _CandlestickPainter extends CustomPainter {
  final List<CandleData> data;
  final double progress;
  final int? selectedIndex;

  _CandlestickPainter({
    required this.data,
    required this.progress,
    this.selectedIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final minPrice = data.map((c) => c.low).reduce(math.min);
    final maxPrice = data.map((c) => c.high).reduce(math.max);
    final priceRange = maxPrice - minPrice;
    final padding = priceRange * 0.1;
    final effectiveMin = minPrice - padding;
    final effectiveMax = maxPrice + padding;
    final effectiveRange = effectiveMax - effectiveMin;

    final candleWidth = size.width / data.length;
    final bodyWidth = candleWidth * 0.6;
    final wickWidth = 1.5;

    // Draw grid
    _drawGrid(canvas, size);

    for (var i = 0; i < data.length; i++) {
      final candle = data[i];
      final x = candleWidth * i + candleWidth / 2;
      final isSelected = i == selectedIndex;

      // Calculate Y positions
      double normalize(double price) {
        return size.height - ((price - effectiveMin) / effectiveRange * size.height);
      }

      final openY = normalize(candle.open);
      final closeY = normalize(candle.close);
      final highY = normalize(candle.high);
      final lowY = normalize(candle.low);

      // Animate from center
      final centerY = (openY + closeY) / 2;
      final animatedOpenY = centerY + (openY - centerY) * progress;
      final animatedCloseY = centerY + (closeY - centerY) * progress;
      final animatedHighY = centerY + (highY - centerY) * progress;
      final animatedLowY = centerY + (lowY - centerY) * progress;

      final color = candle.isUp ? AppColors.successGreen : AppColors.dangerRed;

      // Draw selected highlight
      if (isSelected) {
        final highlightPaint = Paint()
          ..color = color.withValues(alpha: 0.1)
          ..style = PaintingStyle.fill;
        canvas.drawRect(
          Rect.fromLTWH(x - candleWidth / 2, 0, candleWidth, size.height),
          highlightPaint,
        );
      }

      // Draw wick
      final wickPaint = Paint()
        ..color = color
        ..strokeWidth = wickWidth
        ..style = PaintingStyle.stroke;
      canvas.drawLine(
        Offset(x, animatedHighY),
        Offset(x, animatedLowY),
        wickPaint,
      );

      // Draw body
      final bodyTop = math.min(animatedOpenY, animatedCloseY);
      final bodyBottom = math.max(animatedOpenY, animatedCloseY);
      final bodyHeight = math.max(bodyBottom - bodyTop, 1.0);

      final bodyRect = RRect.fromRectAndRadius(
        Rect.fromLTWH(x - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight),
        const Radius.circular(2),
      );

      // Body fill
      final bodyPaint = Paint()
        ..color = candle.isUp ? color : color
        ..style = PaintingStyle.fill;
      canvas.drawRRect(bodyRect, bodyPaint);

      // Body border for hollow candles
      if (candle.isUp) {
        final borderPaint = Paint()
          ..color = color
          ..strokeWidth = 1.5
          ..style = PaintingStyle.stroke;
        canvas.drawRRect(bodyRect, borderPaint);
      }

      // Selected glow
      if (isSelected) {
        final glowPaint = Paint()
          ..color = color.withValues(alpha: 0.4)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
        canvas.drawRRect(bodyRect.inflate(4), glowPaint);
      }
    }
  }

  void _drawGrid(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.06)
      ..strokeWidth = 1;

    for (var i = 0; i <= 4; i++) {
      final y = size.height * (i / 4);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _CandlestickPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.selectedIndex != selectedIndex;
  }
}

class _VolumePainter extends CustomPainter {
  final List<CandleData> data;
  final double progress;
  final int? selectedIndex;

  _VolumePainter({
    required this.data,
    required this.progress,
    this.selectedIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final volumes = data.map((c) => c.volume ?? 0.0).toList();
    final maxVolume = volumes.reduce(math.max);
    if (maxVolume == 0) return;

    final barWidth = size.width / data.length;
    final bodyWidth = barWidth * 0.6;

    for (var i = 0; i < data.length; i++) {
      final candle = data[i];
      final volume = candle.volume ?? 0;
      final x = barWidth * i + barWidth / 2;
      final barHeight = (volume / maxVolume) * size.height * progress;
      final isSelected = i == selectedIndex;

      final color = candle.isUp
          ? AppColors.successGreen.withValues(alpha: isSelected ? 0.8 : 0.4)
          : AppColors.dangerRed.withValues(alpha: isSelected ? 0.8 : 0.4);

      final rect = RRect.fromRectAndRadius(
        Rect.fromLTWH(
          x - bodyWidth / 2,
          size.height - barHeight,
          bodyWidth,
          barHeight,
        ),
        const Radius.circular(2),
      );

      canvas.drawRRect(rect, Paint()..color = color);
    }
  }

  @override
  bool shouldRepaint(covariant _VolumePainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.selectedIndex != selectedIndex;
  }
}

// =============================================================================
// PREMIUM DONUT CHART
// =============================================================================

/// Data model for donut segment
class DonutSegment {
  final String label;
  final double value;
  final Color color;

  const DonutSegment({
    required this.label,
    required this.value,
    required this.color,
  });
}

/// Premium animated donut chart with glass morphism.
class PremiumDonutChart extends StatefulWidget {
  final List<DonutSegment> segments;
  final double size;
  final double strokeWidth;
  final bool showLabels;
  final bool showCenter;
  final String? centerLabel;
  final String? centerValue;
  final bool animate;
  final Duration animationDuration;
  final ValueChanged<int>? onSegmentTap;

  const PremiumDonutChart({
    super.key,
    required this.segments,
    this.size = 200,
    this.strokeWidth = 24,
    this.showLabels = true,
    this.showCenter = true,
    this.centerLabel,
    this.centerValue,
    this.animate = true,
    this.animationDuration = const Duration(milliseconds: 1200),
    this.onSegmentTap,
  });

  @override
  State<PremiumDonutChart> createState() => _PremiumDonutChartState();
}

class _PremiumDonutChartState extends State<PremiumDonutChart>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  int? _selectedIndex;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.animationDuration,
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
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
    if (widget.segments.isEmpty) {
      return SizedBox(
        width: widget.size,
        height: widget.size,
        child: const Center(
          child: Text('No data', style: TextStyle(color: Colors.white54)),
        ),
      );
    }

    return Column(
      children: [
        SizedBox(
          width: widget.size,
          height: widget.size,
          child: GestureDetector(
            onTapDown: _handleTap,
            child: AnimatedBuilder(
              animation: _animation,
              builder: (context, child) {
                return Stack(
                  alignment: Alignment.center,
                  children: [
                    // Donut
                    CustomPaint(
                      painter: _DonutPainter(
                        segments: widget.segments,
                        progress: _animation.value,
                        strokeWidth: widget.strokeWidth,
                        selectedIndex: _selectedIndex,
                      ),
                      size: Size(widget.size, widget.size),
                    ),
                    // Center content
                    if (widget.showCenter) _buildCenter(),
                  ],
                );
              },
            ),
          ),
        ),
        // Legend
        if (widget.showLabels) ...[
          const SizedBox(height: 20),
          _buildLegend(),
        ],
      ],
    );
  }

  Widget _buildCenter() {
    final total = widget.segments.fold<double>(0, (sum, s) => sum + s.value);
    final selectedSegment = _selectedIndex != null ? widget.segments[_selectedIndex!] : null;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          selectedSegment?.label ?? widget.centerLabel ?? 'Total',
          style: TextStyle(
            color: selectedSegment?.color ?? Colors.white54,
            fontSize: 12,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          selectedSegment != null
              ? '${((selectedSegment.value / total) * 100).toStringAsFixed(1)}%'
              : widget.centerValue ?? total.toStringAsFixed(0),
          style: TextStyle(
            color: selectedSegment?.color ?? Colors.white,
            fontSize: 24,
            fontWeight: FontWeight.w800,
          ),
        ),
      ],
    );
  }

  Widget _buildLegend() {
    final total = widget.segments.fold<double>(0, (sum, s) => sum + s.value);

    return Wrap(
      spacing: 16,
      runSpacing: 8,
      alignment: WrapAlignment.center,
      children: widget.segments.asMap().entries.map((entry) {
        final index = entry.key;
        final segment = entry.value;
        final isSelected = _selectedIndex == index;
        final percentage = (segment.value / total * 100).toStringAsFixed(1);

        return GestureDetector(
          onTap: () {
            setState(() {
              _selectedIndex = isSelected ? null : index;
            });
            HapticFeedback.lightImpact();
            widget.onSegmentTap?.call(index);
          },
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: isSelected
                  ? segment.color.withValues(alpha: 0.2)
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: isSelected
                    ? segment.color.withValues(alpha: 0.5)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: segment.color,
                    shape: BoxShape.circle,
                    boxShadow: isSelected
                        ? [
                            BoxShadow(
                              color: segment.color.withValues(alpha: 0.5),
                              blurRadius: 6,
                            ),
                          ]
                        : null,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  '${segment.label} ($percentage%)',
                  style: TextStyle(
                    color: isSelected ? Colors.white : Colors.white70,
                    fontSize: 12,
                    fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
                  ),
                ),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  void _handleTap(TapDownDetails details) {
    final center = Offset(widget.size / 2, widget.size / 2);
    final tapPosition = details.localPosition;
    final dx = tapPosition.dx - center.dx;
    final dy = tapPosition.dy - center.dy;
    final distance = math.sqrt(dx * dx + dy * dy);

    // Check if tap is within donut ring
    final innerRadius = widget.size / 2 - widget.strokeWidth;
    final outerRadius = widget.size / 2;
    if (distance < innerRadius || distance > outerRadius) {
      setState(() => _selectedIndex = null);
      return;
    }

    // Calculate angle
    var angle = math.atan2(dy, dx) + math.pi / 2;
    if (angle < 0) angle += 2 * math.pi;

    // Find segment
    final total = widget.segments.fold<double>(0, (sum, s) => sum + s.value);
    var currentAngle = 0.0;
    for (var i = 0; i < widget.segments.length; i++) {
      final sweepAngle = (widget.segments[i].value / total) * 2 * math.pi;
      if (angle >= currentAngle && angle < currentAngle + sweepAngle) {
        setState(() {
          _selectedIndex = _selectedIndex == i ? null : i;
        });
        HapticFeedback.lightImpact();
        widget.onSegmentTap?.call(i);
        return;
      }
      currentAngle += sweepAngle;
    }
  }
}

class _DonutPainter extends CustomPainter {
  final List<DonutSegment> segments;
  final double progress;
  final double strokeWidth;
  final int? selectedIndex;

  _DonutPainter({
    required this.segments,
    required this.progress,
    required this.strokeWidth,
    this.selectedIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (segments.isEmpty) return;

    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;
    final total = segments.fold<double>(0, (sum, s) => sum + s.value);

    // Background circle
    final bgPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.05)
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(center, radius, bgPaint);

    // Draw segments
    var startAngle = -math.pi / 2;
    for (var i = 0; i < segments.length; i++) {
      final segment = segments[i];
      final sweepAngle = (segment.value / total) * 2 * math.pi * progress;
      final isSelected = i == selectedIndex;

      // Draw glow for selected
      if (isSelected) {
        final glowPaint = Paint()
          ..color = segment.color.withValues(alpha: 0.3)
          ..strokeWidth = strokeWidth + 8
          ..style = PaintingStyle.stroke
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
        canvas.drawArc(
          Rect.fromCircle(center: center, radius: radius),
          startAngle,
          sweepAngle,
          false,
          glowPaint,
        );
      }

      // Draw segment
      final paint = Paint()
        ..color = segment.color
        ..strokeWidth = isSelected ? strokeWidth + 4 : strokeWidth
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round;
      canvas.drawArc(
        Rect.fromCircle(center: center, radius: radius),
        startAngle,
        sweepAngle - 0.02, // Small gap between segments
        false,
        paint,
      );

      startAngle += sweepAngle;
    }
  }

  @override
  bool shouldRepaint(covariant _DonutPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.selectedIndex != selectedIndex;
  }
}

// =============================================================================
// PREMIUM GAUGE
// =============================================================================

/// Premium animated gauge widget.
class PremiumGauge extends StatefulWidget {
  final double value;
  final double min;
  final double max;
  final String? label;
  final String? unit;
  final List<GaugeRange>? ranges;
  final double size;
  final double strokeWidth;
  final bool showValue;
  final bool animate;
  final Duration animationDuration;

  const PremiumGauge({
    super.key,
    required this.value,
    this.min = 0,
    this.max = 100,
    this.label,
    this.unit,
    this.ranges,
    this.size = 180,
    this.strokeWidth = 16,
    this.showValue = true,
    this.animate = true,
    this.animationDuration = const Duration(milliseconds: 1500),
  });

  @override
  State<PremiumGauge> createState() => _PremiumGaugeState();
}

class GaugeRange {
  final double start;
  final double end;
  final Color color;

  const GaugeRange({
    required this.start,
    required this.end,
    required this.color,
  });
}

class _PremiumGaugeState extends State<PremiumGauge>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.animationDuration,
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );

    if (widget.animate) {
      _controller.forward();
    } else {
      _controller.value = 1.0;
    }
  }

  @override
  void didUpdateWidget(PremiumGauge oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.value != widget.value && widget.animate) {
      _controller.forward(from: 0);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Color _getValueColor() {
    if (widget.ranges == null) return AppColors.primaryBlue;

    for (final range in widget.ranges!) {
      if (widget.value >= range.start && widget.value <= range.end) {
        return range.color;
      }
    }
    return AppColors.primaryBlue;
  }

  @override
  Widget build(BuildContext context) {
    final normalizedValue = ((widget.value - widget.min) / (widget.max - widget.min))
        .clamp(0.0, 1.0);
    final valueColor = _getValueColor();

    return SizedBox(
      width: widget.size,
      height: widget.size * 0.7,
      child: AnimatedBuilder(
        animation: _animation,
        builder: (context, child) {
          return Stack(
            alignment: Alignment.center,
            children: [
              // Gauge arc
              CustomPaint(
                painter: _GaugePainter(
                  value: normalizedValue * _animation.value,
                  strokeWidth: widget.strokeWidth,
                  ranges: widget.ranges,
                  valueColor: valueColor,
                  min: widget.min,
                  max: widget.max,
                ),
                size: Size(widget.size, widget.size * 0.7),
              ),
              // Value display
              if (widget.showValue)
                Positioned(
                  bottom: 0,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Text(
                            (widget.value * _animation.value).toStringAsFixed(0),
                            style: TextStyle(
                              color: valueColor,
                              fontSize: 32,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                          if (widget.unit != null)
                            Padding(
                              padding: const EdgeInsets.only(left: 4, bottom: 4),
                              child: Text(
                                widget.unit!,
                                style: TextStyle(
                                  color: valueColor.withValues(alpha: 0.7),
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                        ],
                      ),
                      if (widget.label != null)
                        Text(
                          widget.label!,
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.6),
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                    ],
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  final double value;
  final double strokeWidth;
  final List<GaugeRange>? ranges;
  final Color valueColor;
  final double min;
  final double max;

  _GaugePainter({
    required this.value,
    required this.strokeWidth,
    this.ranges,
    required this.valueColor,
    required this.min,
    required this.max,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height);
    final radius = size.width / 2 - strokeWidth / 2;

    // Background arc
    final bgPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.08)
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      math.pi,
      math.pi,
      false,
      bgPaint,
    );

    // Range arcs
    if (ranges != null) {
      for (final range in ranges!) {
        final startNorm = (range.start - min) / (max - min);
        final endNorm = (range.end - min) / (max - min);
        final startAngle = math.pi + (startNorm * math.pi);
        final sweepAngle = (endNorm - startNorm) * math.pi;

        final rangePaint = Paint()
          ..color = range.color.withValues(alpha: 0.2)
          ..strokeWidth = strokeWidth
          ..style = PaintingStyle.stroke
          ..strokeCap = StrokeCap.round;
        canvas.drawArc(
          Rect.fromCircle(center: center, radius: radius),
          startAngle,
          sweepAngle,
          false,
          rangePaint,
        );
      }
    }

    // Value arc glow
    final glowPaint = Paint()
      ..color = valueColor.withValues(alpha: 0.3)
      ..strokeWidth = strokeWidth + 8
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      math.pi,
      value * math.pi,
      false,
      glowPaint,
    );

    // Value arc
    final valuePaint = Paint()
      ..color = valueColor
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      math.pi,
      value * math.pi,
      false,
      valuePaint,
    );

    // Needle
    final needleAngle = math.pi + (value * math.pi);
    final needleLength = radius - strokeWidth;
    final needleEnd = Offset(
      center.dx + needleLength * math.cos(needleAngle),
      center.dy + needleLength * math.sin(needleAngle),
    );

    // Needle line
    final needlePaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(center, needleEnd, needlePaint);

    // Needle center dot
    canvas.drawCircle(center, 8, Paint()..color = valueColor);
    canvas.drawCircle(
      center,
      8,
      Paint()
        ..color = Colors.white
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke,
    );
  }

  @override
  bool shouldRepaint(covariant _GaugePainter oldDelegate) {
    return oldDelegate.value != value;
  }
}

// =============================================================================
// PREMIUM PROGRESS RING
// =============================================================================

/// Premium animated progress ring.
class PremiumProgressRing extends StatefulWidget {
  final double value;
  final double max;
  final double size;
  final double strokeWidth;
  final Color? color;
  final Color? backgroundColor;
  final String? label;
  final bool showPercentage;
  final bool animate;
  final Duration animationDuration;
  final Widget? centerWidget;

  const PremiumProgressRing({
    super.key,
    required this.value,
    this.max = 100,
    this.size = 120,
    this.strokeWidth = 10,
    this.color,
    this.backgroundColor,
    this.label,
    this.showPercentage = true,
    this.animate = true,
    this.animationDuration = const Duration(milliseconds: 1000),
    this.centerWidget,
  });

  @override
  State<PremiumProgressRing> createState() => _PremiumProgressRingState();
}

class _PremiumProgressRingState extends State<PremiumProgressRing>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.animationDuration,
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );

    if (widget.animate) {
      _controller.forward();
    } else {
      _controller.value = 1.0;
    }
  }

  @override
  void didUpdateWidget(PremiumProgressRing oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.value != widget.value && widget.animate) {
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
    final normalizedValue = (widget.value / widget.max).clamp(0.0, 1.0);
    final color = widget.color ?? AppColors.primaryBlue;
    final bgColor = widget.backgroundColor ?? Colors.white.withValues(alpha: 0.1);

    return SizedBox(
      width: widget.size,
      height: widget.size,
      child: AnimatedBuilder(
        animation: _animation,
        builder: (context, child) {
          return Stack(
            alignment: Alignment.center,
            children: [
              // Progress ring
              CustomPaint(
                painter: _ProgressRingPainter(
                  progress: normalizedValue * _animation.value,
                  strokeWidth: widget.strokeWidth,
                  color: color,
                  backgroundColor: bgColor,
                ),
                size: Size(widget.size, widget.size),
              ),
              // Center content
              widget.centerWidget ??
                  Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (widget.showPercentage)
                        Text(
                          '${(normalizedValue * 100 * _animation.value).toStringAsFixed(0)}%',
                          style: TextStyle(
                            color: color,
                            fontSize: widget.size * 0.2,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      if (widget.label != null)
                        Text(
                          widget.label!,
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.6),
                            fontSize: 11,
                          ),
                        ),
                    ],
                  ),
            ],
          );
        },
      ),
    );
  }
}

class _ProgressRingPainter extends CustomPainter {
  final double progress;
  final double strokeWidth;
  final Color color;
  final Color backgroundColor;

  _ProgressRingPainter({
    required this.progress,
    required this.strokeWidth,
    required this.color,
    required this.backgroundColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;

    // Background
    final bgPaint = Paint()
      ..color = backgroundColor
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(center, radius, bgPaint);

    // Glow
    final glowPaint = Paint()
      ..color = color.withValues(alpha: 0.3)
      ..strokeWidth = strokeWidth + 6
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2,
      progress * 2 * math.pi,
      false,
      glowPaint,
    );

    // Progress
    final progressPaint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2,
      progress * 2 * math.pi,
      false,
      progressPaint,
    );
  }

  @override
  bool shouldRepaint(covariant _ProgressRingPainter oldDelegate) {
    return oldDelegate.progress != progress;
  }
}

// =============================================================================
// PREMIUM MINI SPARKLINE
// =============================================================================

/// Premium mini sparkline for inline display.
class PremiumMiniSparkline extends StatelessWidget {
  final List<double> data;
  final Color? color;
  final double width;
  final double height;
  final double strokeWidth;
  final bool showDot;

  const PremiumMiniSparkline({
    super.key,
    required this.data,
    this.color,
    this.width = 60,
    this.height = 24,
    this.strokeWidth = 1.5,
    this.showDot = true,
  });

  @override
  Widget build(BuildContext context) {
    if (data.isEmpty) {
      return SizedBox(width: width, height: height);
    }

    final isPositive = data.last >= data.first;
    final lineColor = color ?? (isPositive ? AppColors.successGreen : AppColors.dangerRed);

    return SizedBox(
      width: width,
      height: height,
      child: CustomPaint(
        painter: _MiniSparklinePainter(
          data: data,
          color: lineColor,
          strokeWidth: strokeWidth,
          showDot: showDot,
        ),
      ),
    );
  }
}

class _MiniSparklinePainter extends CustomPainter {
  final List<double> data;
  final Color color;
  final double strokeWidth;
  final bool showDot;

  _MiniSparklinePainter({
    required this.data,
    required this.color,
    required this.strokeWidth,
    required this.showDot,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final minValue = data.reduce(math.min);
    final maxValue = data.reduce(math.max);
    final range = (maxValue - minValue).abs() < 0.0001 ? 1.0 : maxValue - minValue;

    final points = <Offset>[];
    for (var i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final normalizedY = (data[i] - minValue) / range;
      final y = size.height - (normalizedY * size.height * 0.8) - size.height * 0.1;
      points.add(Offset(x, y));
    }

    // Draw line
    final path = Path()..moveTo(points.first.dx, points.first.dy);
    for (var i = 1; i < points.length; i++) {
      path.lineTo(points[i].dx, points[i].dy);
    }

    final linePaint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    canvas.drawPath(path, linePaint);

    // Draw end dot
    if (showDot && points.isNotEmpty) {
      // Glow
      canvas.drawCircle(
        points.last,
        4,
        Paint()
          ..color = color.withValues(alpha: 0.3)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4),
      );
      // Dot
      canvas.drawCircle(points.last, 2.5, Paint()..color = color);
    }
  }

  @override
  bool shouldRepaint(covariant _MiniSparklinePainter oldDelegate) {
    return oldDelegate.data != data;
  }
}
