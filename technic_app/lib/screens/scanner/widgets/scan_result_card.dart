/// Scan Result Card Widget
/// 
/// Displays a single scan result with all key metrics and actions.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../models/scan_result.dart';
import '../../../providers/app_providers.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';
import '../../../widgets/sparkline.dart';

class ScanResultCard extends ConsumerWidget {
  final ScanResult result;
  final VoidCallback? onTap;
  
  const ScanResultCard({
    super.key,
    required this.result,
    this.onTap,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final hasSparkline = result.sparkline.isNotEmpty;
    final isPositive = hasSparkline && result.sparkline.last > result.sparkline.first;
    
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header Row
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // Ticker and Signal
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              result.ticker,
                              style: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            if (result.icsTier != null && result.icsTier!.isNotEmpty) ...[
                              const SizedBox(width: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 8,
                                  vertical: 4,
                                ),
                                decoration: BoxDecoration(
                                  color: _getTierColor(result.icsTier!),
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: Text(
                                  result.icsTier!,
                                  style: const TextStyle(
                                    fontSize: 10,
                                    fontWeight: FontWeight.w700,
                                    color: Colors.white,
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                        const SizedBox(height: 4),
                        Text(
                          result.signal,
                          style: const TextStyle(
                            fontSize: 14,
                            color: Colors.white70,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  
                  // Sparkline
                  if (hasSparkline)
                    SizedBox(
                      width: 80,
                      height: 40,
                      child: Sparkline(
                        data: result.sparkline,
                        positive: isPositive,
                        color: AppColors.primaryBlue,
                      ),
                    ),
                ],
              ),
              
              const SizedBox(height: 12),
              
              // MERIT Score (prominent display)
              if (result.meritScore != null) ...[
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        tone(AppColors.primaryBlue, 0.2),
                        tone(AppColors.primaryBlue, 0.1),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: tone(AppColors.primaryBlue, 0.4)),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'MERIT SCORE',
                            style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w700,
                              color: Colors.white60,
                              letterSpacing: 1.2,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Row(
                            children: [
                              Text(
                                result.meritScore!.toStringAsFixed(0),
                                style: const TextStyle(
                                  fontSize: 32,
                                  fontWeight: FontWeight.w900,
                                  color: Colors.white,
                                ),
                              ),
                              const SizedBox(width: 8),
                              if (result.meritBand != null && result.meritBand!.isNotEmpty)
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 8,
                                    vertical: 4,
                                  ),
                                  decoration: BoxDecoration(
                                    color: _getMeritBandColor(result.meritBand!),
                                    borderRadius: BorderRadius.circular(6),
                                  ),
                                  child: Text(
                                    result.meritBand!,
                                    style: const TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.w800,
                                      color: Colors.white,
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ],
                      ),
                      Icon(
                        Icons.verified,
                        size: 40,
                        color: tone(AppColors.primaryBlue, 0.6),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
              ],
              
              // Metrics Row
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  if (result.meritScore == null)
                    _metricChip(
                      'RRR',
                      result.rrr,
                      Icons.trending_up,
                    ),
                  if (result.techRating != null)
                    _metricChip(
                      'Tech',
                      result.techRating!.toStringAsFixed(1),
                      Icons.analytics,
                    ),
                  if (result.winProb10d != null)
                    _metricChip(
                      'Win%',
                      '${(result.winProb10d! * 100).toStringAsFixed(0)}%',
                      Icons.percent,
                    ),
                  if (result.qualityScore != null)
                    _metricChip(
                      'Quality',
                      result.qualityScore!.toStringAsFixed(0),
                      Icons.star,
                    ),
                  if (result.institutionalCoreScore != null)
                    _metricChip(
                      'ICS',
                      result.institutionalCoreScore!.toStringAsFixed(0),
                      Icons.business,
                    ),
                ],
              ),
              
              // MERIT Flags (if any)
              if (result.meritFlags != null && result.meritFlags!.isNotEmpty) ...[
                const SizedBox(height: 12),
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: _buildMeritFlagChips(result.meritFlags!),
                ),
              ],
              
              const SizedBox(height: 12),
              
              // Trade Plan
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.03),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: tone(Colors.white, 0.08)),
                ),
                child: Column(
                  children: [
                    _tradePlanRow('Entry', result.entry, Colors.green),
                    const SizedBox(height: 6),
                    _tradePlanRow('Stop', result.stop, Colors.red),
                    const SizedBox(height: 6),
                    _tradePlanRow('Target', result.target, Colors.blue),
                  ],
                ),
              ),
              
              const SizedBox(height: 12),
              
              // Actions
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () {
                        ref.read(copilotContextProvider.notifier).state = result;
                        ref.read(copilotPrefillProvider.notifier).state =
                            'Analyze ${result.ticker} ${result.signal} setup';
                        ref.read(currentTabProvider.notifier).setTab(2);
                      },
                      icon: const Icon(Icons.chat_bubble_outline, size: 16),
                      label: const Text('Ask Copilot'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: () {
                      ref.read(watchlistProvider.notifier).add(
                            result.ticker,
                            note: result.signal,
                          );
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text('${result.ticker} saved to My Ideas'),
                          duration: const Duration(seconds: 2),
                        ),
                      );
                    },
                    icon: const Icon(Icons.star_outline, size: 16),
                    label: const Text('Save'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _metricChip(String label, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: tone(AppColors.primaryBlue, 0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: tone(AppColors.primaryBlue, 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: AppColors.primaryBlue),
          const SizedBox(width: 4),
          Text(
            '$label: ',
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: Colors.white70,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _tradePlanRow(String label, String value, Color color) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: Colors.white70,
          ),
        ),
        Text(
          value.isNotEmpty ? value : '-',
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: value.isNotEmpty ? color : Colors.white38,
          ),
        ),
      ],
    );
  }
  
  Color _getTierColor(String tier) {
    switch (tier.toUpperCase()) {
      case 'CORE':
        return Colors.green;
      case 'SATELLITE':
        return Colors.blue;
      case 'WATCH':
        return Colors.orange;
      default:
        return Colors.grey;
    }
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
  
  List<Widget> _buildMeritFlagChips(String flags) {
    final flagList = flags.split(',').map((f) => f.trim()).where((f) => f.isNotEmpty).toList();
    
    return flagList.map((flag) {
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
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: bgColor,
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 12, color: Colors.white70),
            const SizedBox(width: 4),
            Text(
              flag,
              style: const TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
          ],
        ),
      );
    }).toList();
  }
}
