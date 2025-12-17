/// MERIT Breakdown Widget
/// 
/// Displays MERIT score with visual breakdown of contributing factors.
/// Features:
/// - Circular progress indicator for overall score
/// - Factor breakdown cards (Momentum, Value, Quality, Growth)
/// - Visual progress bars
/// - Color-coded scores
/// - Band display (A+, A, B, C, D)
library;

import 'package:flutter/material.dart';
import 'dart:math' as math;
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class MeritBreakdownWidget extends StatelessWidget {
  final double meritScore;
  final String? meritBand;
  final String? meritSummary;
  final String? meritFlags;
  final double? momentumScore;
  final double? valueScore;
  final double? qualityScore;
  final double? growthScore;

  const MeritBreakdownWidget({
    super.key,
    required this.meritScore,
    this.meritBand,
    this.meritSummary,
    this.meritFlags,
    this.momentumScore,
    this.valueScore,
    this.qualityScore,
    this.growthScore,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        const Text(
          'MERIT ANALYSIS',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.2,
            color: Colors.white70,
          ),
        ),
        const SizedBox(height: 16),
        
        // Overall Score Card
        _buildOverallScoreCard(),
        
        const SizedBox(height: 16),
        
        // Factor Breakdown
        if (_hasFactorScores) ...[
          const Text(
            'FACTOR BREAKDOWN',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.2,
              color: Colors.white54,
            ),
          ),
          const SizedBox(height: 12),
          _buildFactorBreakdown(),
        ],
        
        // Summary
        if (meritSummary != null && meritSummary!.isNotEmpty) ...[
          const SizedBox(height: 16),
          _buildSummary(),
        ],
        
        // Flags
        if (meritFlags != null && meritFlags!.isNotEmpty) ...[
          const SizedBox(height: 16),
          _buildFlags(),
        ],
      ],
    );
  }

  bool get _hasFactorScores =>
      momentumScore != null ||
      valueScore != null ||
      qualityScore != null ||
      growthScore != null;

  Widget _buildOverallScoreCard() {
    final bandColor = _getMeritBandColor(meritBand ?? '');
    final scoreColor = _getScoreColor(meritScore);

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            tone(scoreColor, 0.2),
            tone(scoreColor, 0.05),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: tone(scoreColor, 0.4)),
      ),
      child: Row(
        children: [
          // Circular Progress
          SizedBox(
            width: 120,
            height: 120,
            child: _buildCircularProgress(meritScore, scoreColor),
          ),
          
          const SizedBox(width: 24),
          
          // Score Details
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Text(
                      'MERIT Score',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 12),
                    if (meritBand != null && meritBand!.isNotEmpty)
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: bandColor,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          meritBand!,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  _getScoreDescription(meritScore),
                  style: const TextStyle(
                    fontSize: 14,
                    color: Colors.white70,
                    height: 1.4,
                  ),
                ),
                const SizedBox(height: 12),
                _buildScoreBar(meritScore, scoreColor),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCircularProgress(double score, Color color) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // Background circle
        CustomPaint(
          size: const Size(120, 120),
          painter: _CircularProgressPainter(
            progress: 1.0,
            color: Colors.white.withValues(alpha: 0.1),
            strokeWidth: 12,
          ),
        ),
        // Progress circle
        CustomPaint(
          size: const Size(120, 120),
          painter: _CircularProgressPainter(
            progress: score / 100,
            color: color,
            strokeWidth: 12,
          ),
        ),
        // Score text
        Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              score.toStringAsFixed(0),
              style: const TextStyle(
                fontSize: 36,
                fontWeight: FontWeight.w900,
                height: 1.0,
              ),
            ),
            const Text(
              '/ 100',
              style: TextStyle(
                fontSize: 14,
                color: Colors.white54,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildScoreBar(double score, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Overall Rating',
              style: TextStyle(
                fontSize: 12,
                color: Colors.white54,
              ),
            ),
            Text(
              '${score.toStringAsFixed(0)}/100',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: score / 100,
            backgroundColor: Colors.white.withValues(alpha: 0.1),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 8,
          ),
        ),
      ],
    );
  }

  Widget _buildFactorBreakdown() {
    final factors = <Map<String, dynamic>>[];

    if (momentumScore != null) {
      factors.add({
        'label': 'Momentum',
        'value': momentumScore!,
        'icon': Icons.trending_up,
        'description': 'Price trend strength',
      });
    }
    if (valueScore != null) {
      factors.add({
        'label': 'Value',
        'value': valueScore!,
        'icon': Icons.attach_money,
        'description': 'Valuation metrics',
      });
    }
    if (qualityScore != null) {
      factors.add({
        'label': 'Quality',
        'value': qualityScore!,
        'icon': Icons.star,
        'description': 'Business quality',
      });
    }
    if (growthScore != null) {
      factors.add({
        'label': 'Growth',
        'value': growthScore!,
        'icon': Icons.show_chart,
        'description': 'Growth potential',
      });
    }

    return Column(
      children: factors.map((factor) {
        final isLast = factor == factors.last;
        return Column(
          children: [
            _buildFactorCard(
              factor['label'] as String,
              factor['value'] as double,
              factor['icon'] as IconData,
              factor['description'] as String,
            ),
            if (!isLast) const SizedBox(height: 12),
          ],
        );
      }).toList(),
    );
  }

  Widget _buildFactorCard(
    String label,
    double value,
    IconData icon,
    String description,
  ) {
    final color = _getScoreColor(value);
    final normalized = (value / 100).clamp(0.0, 1.0);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.03),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: tone(Colors.white, 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: tone(color, 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(icon, size: 20, color: color),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      label,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      description,
                      style: const TextStyle(
                        fontSize: 11,
                        color: Colors.white54,
                      ),
                    ),
                  ],
                ),
              ),
              Text(
                value.toStringAsFixed(0),
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  color: color,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: normalized,
              backgroundColor: Colors.white.withValues(alpha: 0.1),
              valueColor: AlwaysStoppedAnimation<Color>(color),
              minHeight: 6,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSummary() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: tone(AppColors.primaryBlue, 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: tone(AppColors.primaryBlue, 0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            Icons.info_outline,
            size: 20,
            color: AppColors.primaryBlue,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              meritSummary!,
              style: const TextStyle(
                fontSize: 13,
                color: Colors.white70,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFlags() {
    final flagList = meritFlags!
        .split(',')
        .map((f) => f.trim())
        .where((f) => f.isNotEmpty)
        .toList();

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: flagList.map((flag) {
        Color bgColor;
        IconData icon;

        if (flag.contains('EARNINGS')) {
          bgColor = Colors.red.shade900;
          icon = Icons.event;
        } else if (flag.contains('LIQUIDITY')) {
          bgColor = Colors.orange.shade900;
          icon = Icons.water_drop;
        } else if (flag.contains('VOLATILITY') || flag.contains('ATR')) {
          bgColor = Colors.yellow.shade900;
          icon = Icons.show_chart;
        } else if (flag.contains('MICRO') || flag.contains('SMALL')) {
          bgColor = Colors.purple.shade900;
          icon = Icons.business_center;
        } else {
          bgColor = Colors.grey.shade800;
          icon = Icons.flag;
        }

        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: bgColor,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 14, color: Colors.white70),
              const SizedBox(width: 6),
              Text(
                flag,
                style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  Color _getScoreColor(double score) {
    if (score >= 80) return Colors.green;
    if (score >= 70) return Colors.lightGreen;
    if (score >= 60) return Colors.blue;
    if (score >= 50) return Colors.orange;
    return Colors.red;
  }

  Color _getMeritBandColor(String band) {
    switch (band.toUpperCase()) {
      case 'A+':
      case 'A':
        return Colors.green;
      case 'B':
        return Colors.blue;
      case 'C':
        return Colors.orange;
      case 'D':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  String _getScoreDescription(double score) {
    if (score >= 80) return 'Excellent - Strong buy candidate';
    if (score >= 70) return 'Very Good - Above average opportunity';
    if (score >= 60) return 'Good - Solid investment potential';
    if (score >= 50) return 'Fair - Moderate opportunity';
    return 'Poor - High risk, proceed with caution';
  }
}

/// Custom painter for circular progress indicator
class _CircularProgressPainter extends CustomPainter {
  final double progress;
  final Color color;
  final double strokeWidth;

  _CircularProgressPainter({
    required this.progress,
    required this.color,
    required this.strokeWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;
    final rect = Rect.fromCircle(center: center, radius: radius);

    final paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    const startAngle = -math.pi / 2; // Start from top
    final sweepAngle = 2 * math.pi * progress;

    canvas.drawArc(rect, startAngle, sweepAngle, false, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
