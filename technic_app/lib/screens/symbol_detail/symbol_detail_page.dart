/// Symbol Detail Page
/// 
/// Detailed view of a stock symbol with charts, metrics, and analysis.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/scan_result.dart';
import '../../theme/app_colors.dart'; // Using tone from helpers.dart
import '../../utils/helpers.dart';
import '../../widgets/sparkline.dart';
import '../../widgets/section_header.dart';
import '../../widgets/info_card.dart';

/// Symbol detail page showing comprehensive stock information
class SymbolDetailPage extends ConsumerWidget {
  final String ticker;
  final ScanResult? scanResult;

  const SymbolDetailPage({
    super.key,
    required this.ticker,
    this.scanResult,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final result = scanResult;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          ticker,
          style: const TextStyle(fontWeight: FontWeight.w800),
        ),
        actions: [
          IconButton(
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('$ticker added to watchlist')),
              );
            },
            icon: const Icon(Icons.star_border),
            tooltip: 'Add to watchlist',
          ),
        ],
      ),
      body: result == null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.info_outline, size: 64, color: Colors.white38),
                  const SizedBox(height: 16),
                  Text(
                    'No data available for $ticker',
                    style: const TextStyle(fontSize: 16, color: Colors.white70),
                  ),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildPriceCard(result),
                  const SizedBox(height: 16),
                  if (result.sparkline.isNotEmpty) _buildChartCard(result),
                  const SizedBox(height: 16),
                  _buildTechnicalMetrics(result),
                  const SizedBox(height: 16),
                  _buildTradePlan(result),
                  const SizedBox(height: 16),
                  _buildFundamentals(result),
                  const SizedBox(height: 16),
                  _buildActions(context),
                  const SizedBox(height: 80),
                ],
              ),
            ),
    );
  }

  Widget _buildPriceCard(ScanResult result) {
    return InfoCard(
      title: ticker,
      subtitle: result.sector ?? '',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                ticker,
                style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w800),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: tone(AppColors.primaryBlue, 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  result.signal,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    color: AppColors.primaryBlue,
                  ),
                ),
              ),
            ],
          ),
          if (result.sector != null) ...[
            const SizedBox(height: 8),
            Text(
              result.sector!,
              style: const TextStyle(fontSize: 16, color: Colors.white70),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildChartCard(ScanResult result) {
    return InfoCard(
      title: 'Price Chart',
      subtitle: '90-day history',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Price Chart (90 days)',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 200,
            child: Sparkline(
              data: result.sparkline,
              positive: true,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTechnicalMetrics(ScanResult result) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SectionHeader('Technical Analysis'),
        const SizedBox(height: 12),
        InfoCard(
          title: 'Metrics',
          subtitle: 'Technical indicators',
          child: Column(
            children: [
              if (result.techRating != null)
                _buildMetricRow('Tech Rating', result.techRating!.toStringAsFixed(1)),
              if (result.techRating != null) const Divider(height: 24),
              _buildMetricRow('Reward/Risk', result.rrr),
              if (result.winProb10d != null) ...[
                const Divider(height: 24),
                _buildMetricRow(
                  'Win Probability',
                  '${(result.winProb10d! * 100).toStringAsFixed(0)}%',
                ),
              ],
              if (result.institutionalCoreScore != null) ...[
                const Divider(height: 24),
                _buildMetricRow(
                  'ICS Score',
                  result.institutionalCoreScore!.toStringAsFixed(1),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildTradePlan(ScanResult result) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SectionHeader('Trade Plan'),
        const SizedBox(height: 12),
        InfoCard(
          title: 'Entry & Exit',
          subtitle: 'Suggested levels',
          child: Column(
            children: [
              _buildMetricRow('Entry', result.entry),
              const Divider(height: 24),
              _buildMetricRow('Stop Loss', result.stop),
              const Divider(height: 24),
              _buildMetricRow('Target', result.target),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildFundamentals(ScanResult result) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SectionHeader('Fundamentals'),
        const SizedBox(height: 12),
        InfoCard(
          title: 'Company Info',
          subtitle: 'Basic fundamentals',
          child: Column(
            children: [
              if (result.sector != null)
                _buildMetricRow('Sector', result.sector!),
              if (result.sector != null && result.industry != null)
                const Divider(height: 24),
              if (result.industry != null)
                _buildMetricRow('Industry', result.industry!),
              if (result.qualityScore != null) ...[
                const Divider(height: 24),
                _buildMetricRow(
                  'Quality Score',
                  result.qualityScore!.toStringAsFixed(1),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildActions(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ElevatedButton.icon(
          onPressed: () {
            Navigator.pop(context);
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Opening Copilot for $ticker...')),
            );
          },
          icon: const Icon(Icons.chat_bubble),
          label: const Text('Ask Copilot'),
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
            backgroundColor: AppColors.primaryBlue,
          ),
        ),
        const SizedBox(height: 12),
        OutlinedButton.icon(
          onPressed: () {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Options chain coming soon')),
            );
          },
          icon: const Icon(Icons.show_chart),
          label: const Text('View Options'),
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
        ),
      ],
    );
  }

  Widget _buildMetricRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 14, color: Colors.white70),
        ),
        Text(
          value,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
        ),
      ],
    );
  }
}
