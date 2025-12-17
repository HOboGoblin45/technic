/// Scoreboard Card Widget
/// 
/// Displays performance metrics and strategy breakdown.
library;

import 'package:flutter/material.dart';
import '../../../models/scoreboard_slice.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class ScoreboardCard extends StatelessWidget {
  final List<ScoreboardSlice> slices;
  
  const ScoreboardCard({
    super.key,
    required this.slices,
  });

  @override
  Widget build(BuildContext context) {
    if (slices.isEmpty) {
      return const SizedBox.shrink();
    }
    
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: tone(AppColors.darkCard, 0.5),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: tone(Colors.white, 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.leaderboard,
                size: 20,
                color: AppColors.primaryBlue,
              ),
              const SizedBox(width: 8),
              const Text(
                'Performance Scoreboard',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ...slices.map((slice) => _sliceRow(slice)),
        ],
      ),
    );
  }
  
  Widget _sliceRow(ScoreboardSlice slice) {
    final winRateValue = slice.winRateValue ?? 0.0;
    final isPositive = slice.isPositive;
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.03),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: tone(Colors.white, 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                slice.label,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.08),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  slice.horizon,
                  style: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: Colors.white70,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _metricColumn(
                  'Win Rate',
                  slice.winRate,
                  winRateValue >= 50.0 ? Colors.green : Colors.orange,
                ),
              ),
              Expanded(
                child: _metricColumn(
                  'P&L',
                  slice.pnl,
                  isPositive ? Colors.green : Colors.red,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
  
  Widget _metricColumn(String label, String value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: Colors.white60,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: color,
          ),
        ),
      ],
    );
  }
}
