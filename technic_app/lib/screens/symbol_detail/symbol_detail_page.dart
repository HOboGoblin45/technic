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
import '../../services/api_service.dart';
import '../../theme/app_colors.dart';
import '../../utils/formatters.dart';
import '../../widgets/section_header.dart';

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
        actions: [
          IconButton(
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('${widget.ticker} added to watchlist')),
              );
            },
            icon: const Icon(Icons.star_border),
            tooltip: 'Add to watchlist',
          ),
        ],
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
            // Price Header
            _buildPriceHeader(detail),
            const SizedBox(height: 16),

            // MERIT Score Card (if available)
            if (detail.meritScore != null) ...[
              _buildMeritCard(detail),
              const SizedBox(height: 16),
            ],

            // Price Chart
            if (detail.history.isNotEmpty) ...[
              _buildPriceChart(detail),
              const SizedBox(height: 16),
            ],

            // Quantitative Metrics
            _buildMetricsGrid(detail),
            const SizedBox(height: 16),

            // Factor Breakdown
            if (detail.momentumScore != null ||
                detail.valueScore != null ||
                detail.qualityFactor != null ||
                detail.growthScore != null) ...[
              _buildFactorBreakdown(detail),
              const SizedBox(height: 16),
            ],

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

  Widget _buildPriceHeader(SymbolDetail detail) {
    final isPositive = (detail.changePct ?? 0) >= 0;
    final changeColor = isPositive ? Colors.green : Colors.red;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      detail.symbol,
                      style: const TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    if (detail.icsTier != null) ...[
                      const SizedBox(height: 4),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: _getTierColor(detail.icsTier!),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          detail.icsTier!,
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
                if (detail.lastPrice != null)
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        formatCurrency(detail.lastPrice!),
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      if (detail.changePct != null) ...[
                        const SizedBox(height: 4),
                        Text(
                          '${isPositive ? '+' : ''}${detail.changePct!.toStringAsFixed(2)}%',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                            color: changeColor,
                          ),
                        ),
                      ],
                    ],
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMeritCard(SymbolDetail detail) {
    final score = detail.meritScore!;
    final band = detail.meritBand ?? 'N/A';
    final bandColor = _getMeritBandColor(band);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'MERIT SCORE',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.2,
                    color: Colors.white70,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: bandColor,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    band,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Text(
                  score.toStringAsFixed(1),
                  style: const TextStyle(
                    fontSize: 48,
                    fontWeight: FontWeight.w800,
                    height: 1.0,
                  ),
                ),
                const SizedBox(width: 8),
                const Text(
                  '/ 100',
                  style: TextStyle(
                    fontSize: 20,
                    color: Colors.white54,
                  ),
                ),
              ],
            ),
            if (detail.meritSummary != null) ...[
              const SizedBox(height: 12),
              Text(
                detail.meritSummary!,
                style: const TextStyle(
                  fontSize: 14,
                  color: Colors.white70,
                  height: 1.4,
                ),
              ),
            ],
            if (detail.meritFlags != null && detail.meritFlags!.isNotEmpty) ...[
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _buildMeritFlags(detail.meritFlags!),
              ),
            ],
          ],
        ),
      ),
    );
  }

  List<Widget> _buildMeritFlags(String flags) {
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
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: bgColor,
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: Colors.white70),
            const SizedBox(width: 6),
            Text(
              flag,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
          ],
        ),
      );
    }).toList();
  }

  Widget _buildPriceChart(SymbolDetail detail) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'PRICE CHART',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.2,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '${detail.history.length} days',
              style: const TextStyle(
                fontSize: 12,
                color: Colors.white54,
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: _buildCandlestickChart(detail.history),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCandlestickChart(List<PricePoint> history) {
    if (history.isEmpty) {
      return const Center(
        child: Text(
          'No price data available',
          style: TextStyle(color: Colors.white54),
        ),
      );
    }

    // Simple line chart for now (candlestick would require custom painter)
    final closes = history.map((p) => p.close).toList();
    final min = closes.reduce((a, b) => a < b ? a : b);
    final max = closes.reduce((a, b) => a > b ? a : b);
    final range = max - min;

    return CustomPaint(
      painter: _SimpleLinePainter(closes, min, range),
      child: Container(),
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

  Widget _buildFactorBreakdown(SymbolDetail detail) {
    final factors = <Map<String, dynamic>>[];

    if (detail.momentumScore != null) {
      factors.add({'label': 'Momentum', 'value': detail.momentumScore!});
    }
    if (detail.valueScore != null) {
      factors.add({'label': 'Value', 'value': detail.valueScore!});
    }
    if (detail.qualityFactor != null) {
      factors.add({'label': 'Quality', 'value': detail.qualityFactor!});
    }
    if (detail.growthScore != null) {
      factors.add({'label': 'Growth', 'value': detail.growthScore!});
    }

    if (factors.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SectionHeader('Factor Analysis'),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: factors.map((factor) {
                final isLast = factor == factors.last;
                return Column(
                  children: [
                    _buildFactorBar(
                      factor['label'] as String,
                      factor['value'] as double,
                    ),
                    if (!isLast) const SizedBox(height: 16),
                  ],
                );
              }).toList(),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildFactorBar(String label, double value) {
    final normalized = (value / 100).clamp(0.0, 1.0);
    final color = value >= 70
        ? Colors.green
        : value >= 50
            ? Colors.blue
            : value >= 30
                ? Colors.orange
                : Colors.red;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
            Text(
              value.toStringAsFixed(1),
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: normalized,
            backgroundColor: Colors.white12,
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 8,
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
}

/// Simple line chart painter
class _SimpleLinePainter extends CustomPainter {
  final List<double> data;
  final double min;
  final double range;

  _SimpleLinePainter(this.data, this.min, this.range);

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty || range == 0) return;

    final paint = Paint()
      ..color = AppColors.primaryBlue
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final path = Path();
    final stepX = size.width / (data.length - 1);

    for (var i = 0; i < data.length; i++) {
      final x = i * stepX;
      final normalized = (data[i] - min) / range;
      final y = size.height - (normalized * size.height);

      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    canvas.drawPath(path, paint);

    // Fill area under line
    final fillPath = Path.from(path);
    fillPath.lineTo(size.width, size.height);
    fillPath.lineTo(0, size.height);
    fillPath.close();

    final fillPaint = Paint()
      ..color = AppColors.primaryBlue.withAlpha(25)
      ..style = PaintingStyle.fill;

    canvas.drawPath(fillPath, fillPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
