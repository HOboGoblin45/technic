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
import 'widgets/premium_metrics_grid.dart';
import 'widgets/premium_fundamentals_card.dart';
import 'widgets/premium_events_timeline.dart';
import 'widgets/premium_action_buttons.dart';

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

            // Premium Metrics Grid (NEW - Glass morphism with animated counters)
            _buildPremiumMetricsGrid(detail),
            const SizedBox(height: 16),

            // Premium Fundamentals Card (NEW - Glass morphism with color-coded indicators)
            if (detail.fundamentals != null) ...[
              PremiumFundamentalsCard(fundamentals: detail.fundamentals!),
              const SizedBox(height: 24),
            ],

            // Premium Events Timeline (NEW - Timeline visualization with countdowns)
            if (detail.events != null) ...[
              PremiumEventsTimeline(events: detail.events!),
              const SizedBox(height: 24),
            ],

            // Premium Action Buttons (NEW - Gradient buttons with animations)
            PremiumActionButtons(
              symbol: detail.symbol,
              optionsAvailable: detail.optionsAvailable,
              onCopilotTap: () {
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Opening Copilot for ${detail.symbol}...')),
                );
              },
              onOptionsTap: () {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Options chain coming soon')),
                );
              },
            ),
            const SizedBox(height: 80),
          ],
        ),
      ),
    );
  }


  Widget _buildPremiumMetricsGrid(SymbolDetail detail) {
    final metrics = <MetricData>[];

    if (detail.techRating != null) {
      metrics.add(MetricData(
        label: 'Tech Rating',
        value: detail.techRating!.toStringAsFixed(1),
        icon: Icons.trending_up,
        progress: detail.techRating! / 10,
      ));
    }
    if (detail.winProb10d != null) {
      metrics.add(MetricData(
        label: 'Win Prob (10d)',
        value: '${(detail.winProb10d! * 100).toStringAsFixed(0)}%',
        icon: Icons.show_chart,
        progress: detail.winProb10d,
      ));
    }
    if (detail.qualityScore != null) {
      metrics.add(MetricData(
        label: 'Quality',
        value: detail.qualityScore!.toStringAsFixed(1),
        icon: Icons.star,
        progress: detail.qualityScore! / 10,
      ));
    }
    if (detail.ics != null) {
      metrics.add(MetricData(
        label: 'ICS',
        value: detail.ics!.toStringAsFixed(1),
        icon: Icons.analytics,
        progress: detail.ics! / 100,
      ));
    }
    if (detail.alphaScore != null) {
      metrics.add(MetricData(
        label: 'Alpha',
        value: detail.alphaScore!.toStringAsFixed(1),
        icon: Icons.rocket_launch,
        progress: (detail.alphaScore! + 5) / 10, // Normalize -5 to 5 range
      ));
    }
    if (detail.riskScore != null) {
      metrics.add(MetricData(
        label: 'Risk',
        value: detail.riskScore!,
        icon: Icons.shield,
      ));
    }

    if (metrics.isEmpty) {
      return const SizedBox.shrink();
    }

    return PremiumMetricsGrid(
      metrics: metrics,
      title: 'Quantitative Metrics',
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
