/// Scan History Detail Page
/// 
/// Shows detailed view of a past scan with all results
library;

import 'package:flutter/material.dart';

import '../../models/scan_history_item.dart';
import '../../theme/app_colors.dart';
import '../../utils/helpers.dart';
import '../scanner/widgets/scan_result_card.dart';

class ScanHistoryDetailPage extends StatelessWidget {
  final ScanHistoryItem item;

  const ScanHistoryDetailPage({
    super.key,
    required this.item,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E27),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text(
          '${item.scanType} Scan',
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Scan Info Card
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: tone(AppColors.darkCard, 0.5),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: tone(Colors.white, 0.08)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Timestamp
                Row(
                  children: [
                    Icon(
                      Icons.access_time,
                      size: 16,
                      color: Colors.white.withValues(alpha: 0.7),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      item.formattedDateTime,
                      style: const TextStyle(
                        fontSize: 14,
                        color: Colors.white70,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),

                // Stats
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    // Results Count
                    Column(
                      children: [
                        Text(
                          '${item.resultCount}',
                          style: const TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        const Text(
                          'Results',
                          style: TextStyle(
                            fontSize: 13,
                            color: Colors.white70,
                          ),
                        ),
                      ],
                    ),

                    // Divider
                    Container(
                      width: 1,
                      height: 50,
                      color: tone(Colors.white, 0.1),
                    ),

                    // Average MERIT
                    Column(
                      children: [
                        Text(
                          item.averageMerit != null
                              ? item.averageMerit!.toStringAsFixed(1)
                              : 'N/A',
                          style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: item.averageMerit != null
                                ? AppColors.successGreen
                                : Colors.white70,
                          ),
                        ),
                        const Text(
                          'Avg MERIT',
                          style: TextStyle(
                            fontSize: 13,
                            color: Colors.white70,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),

          // Results Header
          Text(
            'Scan Results (${item.resultCount})',
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 12),

          // Results List
          if (item.results.isEmpty)
            Center(
              child: Padding(
                padding: const EdgeInsets.all(32.0),
                child: Text(
                  'No results found in this scan',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white.withValues(alpha: 0.5),
                  ),
                ),
              ),
            )
          else
            ...item.results.map((result) => ScanResultCard(result: result)),
        ],
      ),
    );
  }
}
