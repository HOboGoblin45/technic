/// Market Pulse Card Widget
/// 
/// Displays market movers with positive/negative indicators.
library;

import 'package:flutter/material.dart';
import '../../../models/market_mover.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class MarketPulseCard extends StatelessWidget {
  final List<MarketMover> movers;
  
  const MarketPulseCard({
    super.key,
    required this.movers,
  });

  @override
  Widget build(BuildContext context) {
    if (movers.isEmpty) {
      return const SizedBox.shrink();
    }
    
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            tone(AppColors.skyBlue, 0.08),
            tone(AppColors.darkDeep, 0.95),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: tone(Colors.white, 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.trending_up,
                size: 20,
                color: AppColors.skyBlue,
              ),
              const SizedBox(width: 8),
              const Text(
                'Market Pulse',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.08),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '${movers.length} movers',
                  style: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: Colors.white70,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: movers.map((mover) => _moverChip(mover)).toList(),
          ),
        ],
      ),
    );
  }
  
  Widget _moverChip(MarketMover mover) {
    final color = mover.isPositive ? Colors.green : Colors.red;
    final icon = mover.isPositive ? Icons.arrow_upward : Icons.arrow_downward;
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: tone(color, 0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: tone(color, 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            mover.ticker,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
          const SizedBox(width: 4),
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 2),
          Text(
            mover.delta,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}
