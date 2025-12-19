/// Premium Metrics Grid Widget
/// 
/// Professional metrics display with:
/// - Glass morphism cards
/// - Animated number counters
/// - Color-coded indicators
/// - Smooth hover/tap effects
/// - Grid layout optimization
library;

import 'package:flutter/material.dart';
import 'dart:ui';
import 'dart:math' as math;

import '../../../theme/app_colors.dart';

/// Metric data model
class MetricData {
  final String label;
  final String value;
  final IconData? icon;
  final Color? color;
  final double? progress;
  final String? subtitle;

  const MetricData({
    required this.label,
    required this.value,
    this.icon,
    this.color,
    this.progress,
    this.subtitle,
  });
}

/// Premium metrics grid with glass morphism
class PremiumMetricsGrid extends StatefulWidget {
  final List<MetricData> metrics;
  final String title;

  const PremiumMetricsGrid({
    super.key,
    required this.metrics,
    this.title = 'Quantitative Metrics',
  });

  @override
  State<PremiumMetricsGrid> createState() => _PremiumMetricsGridState();
}

class _PremiumMetricsGridState extends State<PremiumMetricsGrid>
    with TickerProviderStateMixin {
  late AnimationController _fadeController;
  late AnimationController _slideController;
  late List<AnimationController> _counterControllers;
  late List<Animation<double>> _counterAnimations;
  
  @override
  void initState() {
    super.initState();
    
    // Main animations
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    
    _slideController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    
    // Counter animations for each metric
    _counterControllers = List.generate(
      widget.metrics.length,
      (index) => AnimationController(
        duration: Duration(milliseconds: 1000 + (index * 100)),
        vsync: this,
      ),
    );
    
    _counterAnimations = _counterControllers.map((controller) {
      return Tween<double>(
        begin: 0.0,
        end: 1.0,
      ).animate(CurvedAnimation(
        parent: controller,
        curve: Curves.easeOutCubic,
      ));
    }).toList();
    
    // Start animations
    _fadeController.forward();
    _slideController.forward();
    
    // Stagger counter animations
    Future.delayed(const Duration(milliseconds: 300), () {
      for (var i = 0; i < _counterControllers.length; i++) {
        Future.delayed(Duration(milliseconds: i * 50), () {
          if (mounted && i < _counterControllers.length) {
            _counterControllers[i].forward();
          }
        });
      }
    });
  }
  
  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    for (var controller in _counterControllers) {
      controller.dispose();
    }
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fadeController,
      child: SlideTransition(
        position: Tween<Offset>(
          begin: const Offset(0, 0.2),
          end: Offset.zero,
        ).animate(CurvedAnimation(
          parent: _slideController,
          curve: Curves.easeOutCubic,
        )),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Section header
            Padding(
              padding: const EdgeInsets.only(bottom: 16),
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
                          AppColors.primaryBlue.withValues(alpha: 0.3),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    widget.title,
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      letterSpacing: -0.5,
                    ),
                  ),
                ],
              ),
            ),
            
            // Metrics grid
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                childAspectRatio: 1.5,
                crossAxisSpacing: 12,
                mainAxisSpacing: 12,
              ),
              itemCount: widget.metrics.length,
              itemBuilder: (context, index) {
                return AnimatedBuilder(
                  animation: _counterAnimations[index],
                  builder: (context, child) {
                    return _buildMetricCard(
                      widget.metrics[index],
                      _counterAnimations[index].value,
                      index,
                    );
                  },
                );
              },
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildMetricCard(MetricData metric, double animationValue, int index) {
    final color = metric.color ?? _getMetricColor(metric.label);
    
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.95, end: 1.0),
      duration: const Duration(milliseconds: 200),
      builder: (context, scale, child) {
        return Transform.scale(
          scale: scale,
          child: Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  color.withValues(alpha: 0.1 * animationValue),
                  color.withValues(alpha: 0.05 * animationValue),
                ],
              ),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: Colors.white.withValues(alpha: 0.1 * animationValue),
                width: 1,
              ),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: BackdropFilter(
                filter: ImageFilter.blur(
                  sigmaX: 10 * animationValue,
                  sigmaY: 10 * animationValue,
                ),
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    onTap: () {
                      // Add haptic feedback
                      _showMetricDetail(metric);
                    },
                    borderRadius: BorderRadius.circular(20),
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          // Header with icon
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Expanded(
                                child: Text(
                                  metric.label,
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w600,
                                    color: Colors.white.withValues(alpha: 0.7 * animationValue),
                                    letterSpacing: 0.5,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                              if (metric.icon != null)
                                Transform.rotate(
                                  angle: (1 - animationValue) * math.pi * 2,
                                  child: Icon(
                                    metric.icon,
                                    size: 16,
                                    color: color.withValues(alpha: 0.8 * animationValue),
                                  ),
                                ),
                            ],
                          ),
                          
                          // Animated value
                          _buildAnimatedValue(metric, animationValue, color),
                          
                          // Progress bar or subtitle
                          if (metric.progress != null)
                            _buildProgressBar(metric.progress!, color, animationValue)
                          else if (metric.subtitle != null)
                            Text(
                              metric.subtitle!,
                              style: TextStyle(
                                fontSize: 10,
                                color: color.withValues(alpha: 0.6 * animationValue),
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
  
  Widget _buildAnimatedValue(MetricData metric, double animationValue, Color color) {
    // Parse numeric value for animation
    final numericMatch = RegExp(r'[\d.]+').firstMatch(metric.value);
    
    if (numericMatch != null && animationValue > 0) {
      final numericValue = double.tryParse(numericMatch.group(0)!) ?? 0;
      final animatedValue = (numericValue * animationValue);
      final prefix = metric.value.substring(0, numericMatch.start);
      final suffix = metric.value.substring(numericMatch.end);
      
      return Row(
        crossAxisAlignment: CrossAxisAlignment.baseline,
        textBaseline: TextBaseline.alphabetic,
        children: [
          if (prefix.isNotEmpty)
            Text(
              prefix,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w800,
                color: Colors.white.withValues(alpha: animationValue),
              ),
            ),
          Text(
            animatedValue.toStringAsFixed(
              numericMatch.group(0)!.contains('.') ? 1 : 0,
            ),
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w800,
              color: Colors.white.withValues(alpha: animationValue),
              letterSpacing: -0.5,
            ),
          ),
          if (suffix.isNotEmpty)
            Text(
              suffix,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: color.withValues(alpha: 0.8 * animationValue),
              ),
            ),
        ],
      );
    }
    
    // Non-numeric value
    return Text(
      metric.value,
      style: TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.w800,
        color: Colors.white.withValues(alpha: animationValue),
        letterSpacing: -0.5,
      ),
    );
  }
  
  Widget _buildProgressBar(double progress, Color color, double animationValue) {
    return Column(
      children: [
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: progress * animationValue,
            backgroundColor: Colors.white.withValues(alpha: 0.1),
            valueColor: AlwaysStoppedAnimation<Color>(
              color.withValues(alpha: 0.8 * animationValue),
            ),
            minHeight: 4,
          ),
        ),
      ],
    );
  }
  
  Color _getMetricColor(String label) {
    // Assign colors based on metric type
    if (label.toLowerCase().contains('win') || 
        label.toLowerCase().contains('alpha')) {
      return AppColors.successGreen;
    } else if (label.toLowerCase().contains('risk')) {
      return AppColors.dangerRed;
    } else if (label.toLowerCase().contains('quality') || 
               label.toLowerCase().contains('ics')) {
      return AppColors.primaryBlue;
    } else if (label.toLowerCase().contains('tech')) {
      return Colors.purple;
    } else {
      return Colors.cyan;
    }
  }
  
  void _showMetricDetail(MetricData metric) {
    // Show detailed metric information
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return Container(
          decoration: BoxDecoration(
            color: const Color(0xFF1A1A2E),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
              width: 1,
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Handle bar
                Center(
                  child: Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.3),
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                
                // Metric details
                Row(
                  children: [
                    if (metric.icon != null) ...[
                      Icon(
                        metric.icon,
                        size: 24,
                        color: metric.color ?? _getMetricColor(metric.label),
                      ),
                      const SizedBox(width: 12),
                    ],
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            metric.label,
                            style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: Colors.white70,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            metric.value,
                            style: const TextStyle(
                              fontSize: 28,
                              fontWeight: FontWeight.w800,
                              color: Colors.white,
                            ),
                          ),
                          if (metric.subtitle != null) ...[
                            const SizedBox(height: 8),
                            Text(
                              metric.subtitle!,
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.white.withValues(alpha: 0.5),
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ],
                ),
                
                const SizedBox(height: 24),
                
                // Description based on metric type
                Text(
                  _getMetricDescription(metric.label),
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.white.withValues(alpha: 0.7),
                    height: 1.5,
                  ),
                ),
                
                const SizedBox(height: 24),
              ],
            ),
          ),
        );
      },
    );
  }
  
  String _getMetricDescription(String label) {
    final descriptions = {
      'Tech Rating': 'Technical analysis score based on price action, volume, and momentum indicators.',
      'Win Prob (10d)': 'Probability of positive returns over the next 10 trading days based on historical patterns.',
      'Quality': 'Company quality score evaluating fundamentals, management, and competitive position.',
      'ICS': 'Intelligent Composite Score combining multiple factors for overall stock rating.',
      'Alpha': 'Expected excess return compared to market benchmark.',
      'Risk': 'Overall risk assessment considering volatility, beta, and downside potential.',
    };
    
    return descriptions[label] ?? 'Advanced metric for comprehensive stock analysis.';
  }
}
