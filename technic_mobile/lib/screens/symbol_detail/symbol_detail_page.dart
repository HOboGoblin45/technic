/// Symbol Detail Page - Comprehensive Stock Analysis
/// 
/// Displays detailed information for a stock symbol including:
/// - Real-time price and chart
/// - MERIT Score with breakdown
/// - Technical metrics
/// - Fundamentals
/// - Events (earnings, dividends)
/// - Factor analysis
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/symbol_detail.dart';
import '../../providers/app_providers.dart';
import '../../services/api_service.dart';
import '../../theme/app_colors.dart';
import '../../utils/formatters.dart';
import '../../widgets/section_header.dart';
import 'widgets/merit_breakdown_widget.dart';
import 'widgets/trade_plan_widget.dart';
import 'widgets/premium_price_header.dart';
import 'widgets/premium_chart_section.dart';

/// Symbol detail page with comprehensive analysis
class SymbolDetailPage extends ConsumerStatefulWidget {
  final String ticker;

  const SymbolDetailPage({
    super.key,
    required this.ticker,
  });

  @override
  ConsumerState<SymbolDetailPage> createState() => _SymbolDetailPageState();
}

class _SymbolDetailPageState extends ConsumerState<SymbolDetailPage> {
  late Future<SymbolDetail> _detailFuture;
  final _apiService = ApiService();

  @override
  void initState() {
    super.initState();
    _detailFuture = _apiService.fetchSymbolDetail(widget.ticker);
  }

  @override
  void dispose() {
    _apiService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          widget.ticker,
          style: const TextStyle(fontWeight: FontWeight.w800),
        ),
      ),
      body: FutureBuilder<SymbolDetail>(
        future: _detailFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (snapshot.hasError) {
            return _buildError(snapshot.error.toString());
          }

          if (!snapshot.hasData) {
            return _buildError('No data available');
          }

          final detail = snapshot.data!;
          return _buildContent(detail);
        },
      ),
    );
  }

  Widget _buildError(String message) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 64, color: Colors.white38),
          const SizedBox(height: 16),
          Text(
            message,
            style: const TextStyle(fontSize: 16, color: Colors.white70),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          ElevatedButton(
            onPressed: () {
              setState(() {
                _detailFuture = _apiService.fetchSymbolDetail(widget.ticker);
              });
            },
            child: const Text('Retry'),
          ),
        ],
      ),
    );
  }

  Widget _buildContent(SymbolDetail detail) {
    return RefreshIndicator(
      onRefresh: () async {
        setState(() {
          _detailFuture = _apiService.fetchSymbolDetail(widget.ticker);
        });
        await _detailFuture;
      },
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Premium Price Header
            Consumer(
              builder: (context, ref, _) {
                final watchlist = ref.watch(watchlistProvider);
                final isWatched = watchlist.any((item) => item.ticker == detail.symbol);
                
                return PremiumPriceHeader(
                  symbol: detail.symbol,
                  companyName: null, // TODO: Add company name to API response
                  currentPrice: detail.lastPrice,
                  changePct: detail.changePct,
                  changeAmount: detail.lastPrice != null && detail.changePct != null
                      ? detail.lastPrice! * (detail.changePct! / 100)
                      : null,
                  icsTier: detail.icsTier,
                  isWatched: isWatched,
                  onWatchlistToggle: () async {
                    if (isWatched) {
                      await ref.read(watchlistProvider.notifier).remove(detail.symbol);
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text('${detail.symbol} removed from watchlist'),
                          ),
                        );
                      }
                    } else {
                      await ref.read(watchlistProvider.notifier).add(detail.symbol);
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text('${detail.symbol} added to watchlist'),
                            backgroundColor: AppColors.successGreen,
                          ),
                        );
                      }
                    }
                  },
                  isMarketOpen: _isMarketOpen(),
                  volume: detail.history.isNotEmpty ? detail.history.last.volume.toDouble() : null,
                  marketCap: detail.fundamentals?.marketCap,
                );
              },
            ),
            const SizedBox(height: 24),

            // Premium Chart Section (NEW - Glass morphism with timeframe selector)
            if (detail.history.isNotEmpty) ...[
              PremiumChartSection(
                history: detail.history,
                symbol: detail.symbol,
                currentPrice: detail.lastPrice,
              ),
              const SizedBox(height: 24),
            ],

            // MERIT Breakdown (NEW - Professional widget with circular progress)
            if (detail.meritScore != null) ...[
              MeritBreakdownWidget(
                meritScore: detail.meritScore!,
                meritBand: detail.meritBand,
                meritSummary: detail.meritSummary,
                meritFlags: detail.meritFlags,
                momentumScore: detail.momentumScore,
                valueScore: detail.valueScore,
                qualityScore: detail.qualityFactor,
                growthScore: detail.growthScore,
              ),
              const SizedBox(height: 24),
            ],

            // Trade Plan (NEW - Professional widget with R:R and position sizing)
            TradePlanWidget(
              symbol: detail.symbol,
              currentPrice: detail.lastPrice ?? 0,
              // TODO: Get these from backend trade plan API
              entryPrice: detail.lastPrice,
              stopLoss: detail.lastPrice != null ? detail.lastPrice! * 0.95 : null,
              target1: detail.lastPrice != null ? detail.lastPrice! * 1.05 : null,
              target2: detail.lastPrice != null ? detail.lastPrice! * 1.10 : null,
              target3: detail.lastPrice != null ? detail.lastPrice! * 1.15 : null,
              accountSize: 10000, // TODO: Get from user settings
            ),
            const SizedBox(height: 24),

            // Quantitative Metrics
            _buildMetricsGrid(detail),
            const SizedBox(height: 16),

            // Fundamentals
            if (detail.fundamentals != null) ...[
              _buildFundamentals(detail.fundamentals!),
              const SizedBox(height: 16),
            ],

            // Events
            if (detail.events != null) ...[
              _buildEvents(detail.events!),
              const SizedBox(height: 16),
            ],

            // Actions
            _buildActions(detail),
            const SizedBox(height: 80),
          ],
        ),
      ),
    );
  }


  Widget _buildMetricsGrid(SymbolDetail detail) {
    final metrics = <Map<String, dynamic>>[];

    if (detail.techRating != null) {
      metrics.add({'label': 'Tech Rating', 'value': detail.techRating!.toStringAsFixed(1)});
    }
    if (detail.winProb10d != null) {
      metrics.add({'label': 'Win Prob (10d)', 'value': '${(detail.winProb10d! * 100).toStringAsFixed(0)}%'});
    }
    if (detail.qualityScore != null) {
      metrics.add({'label': 'Quality', 'value': detail.qualityScore!.toStringAsFixed(1)});
    }
    if (detail.ics != null) {
      metrics.add({'label': 'ICS', 'value': detail.ics!.toStringAsFixed(1)});
    }
    if (detail.alphaScore != null) {
      metrics.add({'label': 'Alpha', 'value': detail.alphaScore!.toStringAsFixed(1)});
    }
    if (detail.riskScore != null) {
      metrics.add({'label': 'Risk', 'value': detail.riskScore!});
    }

    if (metrics.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SectionHeader('Quantitative Metrics'),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                childAspectRatio: 2.5,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
              ),
              itemCount: metrics.length,
              itemBuilder: (context, index) {
                final metric = metrics[index];
                return _buildMetricTile(
                  metric['label'] as String,
                  metric['value'] as String,
                );
              },
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMetricTile(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: Colors.white54,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }

  Widget _buildFundamentals(Fundamentals fundamentals) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SectionHeader('Fundamentals'),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                if (fundamentals.pe != null)
                  _buildFundamentalRow('P/E Ratio', fundamentals.pe!.toStringAsFixed(2)),
                if (fundamentals.pe != null && fundamentals.eps != null)
                  const Divider(height: 24),
                if (fundamentals.eps != null)
                  _buildFundamentalRow('EPS', formatCurrency(fundamentals.eps!)),
                if (fundamentals.eps != null && fundamentals.roe != null)
                  const Divider(height: 24),
                if (fundamentals.roe != null)
                  _buildFundamentalRow('ROE', '${fundamentals.roe!.toStringAsFixed(1)}%'),
                if (fundamentals.roe != null && fundamentals.debtToEquity != null)
                  const Divider(height: 24),
                if (fundamentals.debtToEquity != null)
                  _buildFundamentalRow('Debt/Equity', fundamentals.debtToEquity!.toStringAsFixed(2)),
                if (fundamentals.debtToEquity != null && fundamentals.marketCap != null)
                  const Divider(height: 24),
                if (fundamentals.marketCap != null)
                  _buildFundamentalRow('Market Cap', formatCompact(fundamentals.marketCap!)),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildFundamentalRow(String label, String value) {
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

  Widget _buildEvents(EventInfo events) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SectionHeader('Upcoming Events'),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                if (events.nextEarnings != null) ...[
                  _buildEventRow(
                    Icons.event,
                    'Earnings',
                    formatDate(events.nextEarnings!),
                    events.daysToEarnings != null
                        ? 'in ${events.daysToEarnings} days'
                        : null,
                  ),
                  if (events.nextDividend != null) const Divider(height: 24),
                ],
                if (events.nextDividend != null)
                  _buildEventRow(
                    Icons.payments,
                    'Dividend',
                    formatDate(events.nextDividend!),
                    events.dividendAmount != null
                        ? formatCurrency(events.dividendAmount!)
                        : null,
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildEventRow(IconData icon, String label, String date, String? subtitle) {
    return Row(
      children: [
        Icon(icon, size: 24, color: AppColors.primaryBlue),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                date,
                style: const TextStyle(
                  fontSize: 13,
                  color: Colors.white70,
                ),
              ),
            ],
          ),
        ),
        if (subtitle != null)
          Text(
            subtitle,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: AppColors.primaryBlue,
            ),
          ),
      ],
    );
  }

  Widget _buildActions(SymbolDetail detail) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ElevatedButton.icon(
          onPressed: () {
            Navigator.pop(context);
            // TODO: Navigate to Copilot with symbol context
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Opening Copilot for ${detail.symbol}...')),
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
        if (detail.optionsAvailable)
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

  bool _isMarketOpen() {
    final now = DateTime.now();
    final hour = now.hour;
    final minute = now.minute;
    final dayOfWeek = now.weekday;
    
    // Market is closed on weekends
    if (dayOfWeek == DateTime.saturday || dayOfWeek == DateTime.sunday) {
      return false;
    }
    
    // Market hours: 9:30 AM - 4:00 PM ET (simplified, not accounting for timezone)
    final marketOpen = hour > 9 || (hour == 9 && minute >= 30);
    final marketClosed = hour >= 16;
    
    return marketOpen && !marketClosed;
  }
}
