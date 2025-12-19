/// Trade Plan Widget
/// 
/// Displays trade plan with entry, stop loss, and target prices.
/// Features:
/// - Entry price indicator
/// - Stop loss level with risk amount
/// - Target prices (T1, T2, T3) with profit potential
/// - Risk/reward ratio calculation
/// - Position size calculator
/// - Visual price level markers
library;

import 'package:flutter/material.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';
import '../../../utils/formatters.dart';

class TradePlanWidget extends StatelessWidget {
  final String symbol;
  final double currentPrice;
  final double? entryPrice;
  final double? stopLoss;
  final double? target1;
  final double? target2;
  final double? target3;
  final double? positionSize;
  final double? accountSize;

  const TradePlanWidget({
    super.key,
    required this.symbol,
    required this.currentPrice,
    this.entryPrice,
    this.stopLoss,
    this.target1,
    this.target2,
    this.target3,
    this.positionSize,
    this.accountSize,
  });

  @override
  Widget build(BuildContext context) {
    if (!_hasTradePlan) {
      return _buildNoTradePlan();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        const Text(
          'TRADE PLAN',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.2,
            color: Colors.white70,
          ),
        ),
        const SizedBox(height: 16),
        
        // Price Levels Card
        _buildPriceLevelsCard(),
        
        const SizedBox(height: 16),
        
        // Risk/Reward Analysis
        _buildRiskRewardCard(),
        
        const SizedBox(height: 16),
        
        // Position Sizing
        if (accountSize != null) _buildPositionSizingCard(),
      ],
    );
  }

  bool get _hasTradePlan =>
      entryPrice != null || stopLoss != null || target1 != null;

  Widget _buildNoTradePlan() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.03),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: tone(Colors.white, 0.08)),
      ),
      child: const Center(
        child: Column(
          children: [
            Icon(Icons.trending_up, size: 48, color: Colors.white38),
            SizedBox(height: 16),
            Text(
              'No trade plan available',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white70,
              ),
            ),
            SizedBox(height: 8),
            Text(
              'Trade plan will be generated based on technical analysis',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 12,
                color: Colors.white54,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPriceLevelsCard() {
    final entry = entryPrice ?? currentPrice;
    final levels = <Map<String, dynamic>>[];

    // Add all price levels
    if (target3 != null) {
      levels.add({
        'label': 'Target 3',
        'price': target3!,
        'color': Colors.green.shade700,
        'icon': Icons.flag,
        'type': 'target',
      });
    }
    if (target2 != null) {
      levels.add({
        'label': 'Target 2',
        'price': target2!,
        'color': Colors.green.shade600,
        'icon': Icons.flag,
        'type': 'target',
      });
    }
    if (target1 != null) {
      levels.add({
        'label': 'Target 1',
        'price': target1!,
        'color': Colors.green.shade500,
        'icon': Icons.flag,
        'type': 'target',
      });
    }

    levels.add({
      'label': 'Entry',
      'price': entry,
      'color': AppColors.primaryBlue,
      'icon': Icons.login,
      'type': 'entry',
    });

    if (stopLoss != null) {
      levels.add({
        'label': 'Stop Loss',
        'price': stopLoss!,
        'color': Colors.red.shade600,
        'icon': Icons.block,
        'type': 'stop',
      });
    }

    // Sort by price (highest to lowest)
    levels.sort((a, b) => (b['price'] as double).compareTo(a['price'] as double));

    return Container(
      padding: const EdgeInsets.all(20),
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
              Icon(Icons.layers, size: 20, color: AppColors.primaryBlue),
              const SizedBox(width: 8),
              const Text(
                'Price Levels',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          // Current Price Indicator
          _buildCurrentPriceIndicator(),
          
          const SizedBox(height: 16),
          
          // Price Levels
          ...levels.map((level) {
            final isLast = level == levels.last;
            return Column(
              children: [
                _buildPriceLevel(
                  level['label'] as String,
                  level['price'] as double,
                  level['color'] as Color,
                  level['icon'] as IconData,
                  level['type'] as String,
                ),
                if (!isLast) const SizedBox(height: 12),
              ],
            );
          }),
        ],
      ),
    );
  }

  Widget _buildCurrentPriceIndicator() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: tone(Colors.white, 0.15), width: 2),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.show_chart, size: 16, color: Colors.white),
          ),
          const SizedBox(width: 12),
          const Text(
            'Current Price',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.white70,
            ),
          ),
          const Spacer(),
          Text(
            formatCurrency(currentPrice),
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPriceLevel(
    String label,
    double price,
    Color color,
    IconData icon,
    String type,
  ) {
    final entry = entryPrice ?? currentPrice;
    final diffFromEntry = price - entry;
    final diffPct = (diffFromEntry / entry) * 100;
    final isAbove = price > currentPrice;
    final diffFromCurrent = price - currentPrice;
    final diffFromCurrentPct = (diffFromCurrent / currentPrice) * 100;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: tone(color, 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: tone(color, 0.3)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Icon(icon, size: 16, color: Colors.white),
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
                    if (type != 'entry')
                      Text(
                        '${diffPct >= 0 ? '+' : ''}${diffPct.toStringAsFixed(1)}% from entry',
                        style: TextStyle(
                          fontSize: 11,
                          color: diffPct >= 0 ? Colors.green : Colors.red,
                        ),
                      ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    formatCurrency(price),
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
                      color: color,
                    ),
                  ),
                  Row(
                    children: [
                      Icon(
                        isAbove ? Icons.arrow_upward : Icons.arrow_downward,
                        size: 12,
                        color: Colors.white54,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        '${diffFromCurrentPct.abs().toStringAsFixed(1)}%',
                        style: const TextStyle(
                          fontSize: 11,
                          color: Colors.white54,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRiskRewardCard() {
    if (stopLoss == null || target1 == null) {
      return const SizedBox.shrink();
    }

    final entry = entryPrice ?? currentPrice;
    final risk = (entry - stopLoss!).abs();
    final reward1 = (target1! - entry).abs();
    final reward2 = target2 != null ? (target2! - entry).abs() : null;
    final reward3 = target3 != null ? (target3! - entry).abs() : null;

    final rr1 = reward1 / risk;
    final rr2 = reward2 != null ? reward2 / risk : null;
    final rr3 = reward3 != null ? reward3 / risk : null;

    final riskPct = (risk / entry) * 100;
    final reward1Pct = (reward1 / entry) * 100;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            tone(AppColors.primaryBlue, 0.2),
            tone(AppColors.primaryBlue, 0.05),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: tone(AppColors.primaryBlue, 0.4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.analytics, size: 20, color: AppColors.primaryBlue),
              const SizedBox(width: 8),
              const Text(
                'Risk/Reward Analysis',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          // Risk Amount
          _buildRiskRewardRow(
            'Risk per Share',
            formatCurrency(risk),
            '${riskPct.toStringAsFixed(2)}%',
            Colors.red.shade400,
            Icons.trending_down,
          ),
          
          const SizedBox(height: 12),
          
          // Reward Amount (T1)
          _buildRiskRewardRow(
            'Reward (T1)',
            formatCurrency(reward1),
            '${reward1Pct.toStringAsFixed(2)}%',
            Colors.green.shade400,
            Icons.trending_up,
          ),
          
          const SizedBox(height: 16),
          const Divider(color: Colors.white24),
          const SizedBox(height: 16),
          
          // R:R Ratios
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildRRRatio('T1', rr1, Colors.green.shade500),
              if (rr2 != null) _buildRRRatio('T2', rr2, Colors.green.shade600),
              if (rr3 != null) _buildRRRatio('T3', rr3, Colors.green.shade700),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRiskRewardRow(
    String label,
    String amount,
    String percentage,
    Color color,
    IconData icon,
  ) {
    return Row(
      children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(width: 8),
        Text(
          label,
          style: const TextStyle(
            fontSize: 14,
            color: Colors.white70,
          ),
        ),
        const Spacer(),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              amount,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w800,
                color: color,
              ),
            ),
            Text(
              percentage,
              style: TextStyle(
                fontSize: 12,
                color: color.withValues(alpha: 0.7),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildRRRatio(String target, double ratio, Color color) {
    return Column(
      children: [
        Text(
          target,
          style: const TextStyle(
            fontSize: 12,
            color: Colors.white54,
          ),
        ),
        const SizedBox(height: 4),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            '1:${ratio.toStringAsFixed(1)}',
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w800,
              color: Colors.white,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPositionSizingCard() {
    if (stopLoss == null || accountSize == null) {
      return const SizedBox.shrink();
    }

    final entry = entryPrice ?? currentPrice;
    final riskPerShare = (entry - stopLoss!).abs();
    
    // Calculate position sizes for different risk percentages
    final risk1Pct = accountSize! * 0.01; // 1% risk
    final risk2Pct = accountSize! * 0.02; // 2% risk
    final risk3Pct = accountSize! * 0.03; // 3% risk
    
    final shares1Pct = (risk1Pct / riskPerShare).floor();
    final shares2Pct = (risk2Pct / riskPerShare).floor();
    final shares3Pct = (risk3Pct / riskPerShare).floor();
    
    final cost1Pct = shares1Pct * entry;
    final cost2Pct = shares2Pct * entry;
    final cost3Pct = shares3Pct * entry;

    return Container(
      padding: const EdgeInsets.all(20),
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
              Icon(Icons.calculate, size: 20, color: AppColors.primaryBlue),
              const SizedBox(width: 8),
              const Text(
                'Position Sizing',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Account Size: ${formatCurrency(accountSize!)}',
            style: const TextStyle(
              fontSize: 12,
              color: Colors.white54,
            ),
          ),
          const SizedBox(height: 16),
          
          _buildPositionSizeOption('1% Risk', shares1Pct, cost1Pct, risk1Pct),
          const SizedBox(height: 12),
          _buildPositionSizeOption('2% Risk', shares2Pct, cost2Pct, risk2Pct),
          const SizedBox(height: 12),
          _buildPositionSizeOption('3% Risk', shares3Pct, cost3Pct, risk3Pct),
        ],
      ),
    );
  }

  Widget _buildPositionSizeOption(
    String label,
    int shares,
    double cost,
    double riskAmount,
  ) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.05),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
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
                const SizedBox(height: 4),
                Text(
                  'Risk: ${formatCurrency(riskAmount)}',
                  style: const TextStyle(
                    fontSize: 11,
                    color: Colors.white54,
                  ),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                '$shares shares',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              Text(
                formatCurrency(cost),
                style: TextStyle(
                  fontSize: 12,
                  color: AppColors.primaryBlue,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
