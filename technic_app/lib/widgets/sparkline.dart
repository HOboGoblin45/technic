/// Sparkline Widget
/// 
/// A simple line chart widget for displaying price trends.
/// Used throughout the app for visualizing stock price movements.

import 'dart:math' as math;
import 'package:flutter/material.dart';

class Sparkline extends StatelessWidget {
  final List<double> data;
  final bool positive;
  final Color? color;
  final double strokeWidth;

  const Sparkline({
    super.key,
    required this.data,
    required this.positive,
    this.color,
    this.strokeWidth = 2.0,
  });

  @override
  Widget build(BuildContext context) {
    final effectiveColor = color ??
        (positive
            ? Theme.of(context).colorScheme.primary
            : Theme.of(context).colorScheme.error);

    return CustomPaint(
      painter: _SparklinePainter(
        data,
        effectiveColor,
        strokeWidth,
      ),
      size: const Size(double.infinity, double.infinity),
    );
  }
}

class _SparklinePainter extends CustomPainter {
  final List<double> data;
  final Color color;
  final double strokeWidth;

  const _SparklinePainter(this.data, this.color, this.strokeWidth);

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final minValue = data.reduce(math.min);
    final maxValue = data.reduce(math.max);
    final range = (maxValue - minValue).abs() < 0.0001 ? 1.0 : maxValue - minValue;

    final points = <Offset>[];
    for (var i = 0; i < data.length; i++) {
      final x = size.width * (i / math.max(1, data.length - 1));
      final normalized = (data[i] - minValue) / range;
      final y = size.height - (normalized * size.height);
      points.add(Offset(x, y));
    }

    // Draw line
    final path = Path()..moveTo(points.first.dx, points.first.dy);
    for (var i = 1; i < points.length; i++) {
      path.lineTo(points[i].dx, points[i].dy);
    }

    final paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;

    // Draw gradient fill
    final fillPaint = Paint()
      ..shader = LinearGradient(
        colors: [
          color.withValues(alpha: 0.25),
          Colors.transparent,
        ],
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height))
      ..style = PaintingStyle.fill;

    final fillPath = Path.from(path)
      ..lineTo(points.last.dx, size.height)
      ..lineTo(points.first.dx, size.height)
      ..close();

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
