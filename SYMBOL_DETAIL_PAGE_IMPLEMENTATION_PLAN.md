# Symbol Detail Page - Implementation Plan

## Overview
Create a comprehensive Symbol Detail Page that displays when users tap on a stock from scan results, ideas, or watchlist. This page will showcase all the sophisticated backend data that's currently underutilized.

---

## 🎯 Goals

1. **Surface Backend Power**: Display MERIT Score, ICS, Quality, Win Probability, Factor Breakdown
2. **Visual Appeal**: Large chart, clean metrics grid, intuitive layout
3. **Actionable**: Quick access to Copilot, options strategies, save to watchlist
4. **Performance**: Fast loading, cached data, smooth animations

---

## 📐 Design Specification

### Layout Structure

```
┌─────────────────────────────────────────┐
│ ← AAPL                    ⋮ [Menu]      │ ← App Bar
├─────────────────────────────────────────┤
│                                         │
│        📈 Price Chart (90 days)         │ ← Hero Section
│           Candlestick                   │
│        $150.25  +2.5%                   │
│                                         │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │ MERIT SCORE: 87  [A]                │ │ ← MERIT Card
│ │ Elite institutional-grade setup...   │ │
│ │ [⚠ EARNINGS_SOON]                   │ │
│ └─────────────────────────────────────┘ │
├─────────────────────────────────────────┤
│ Key Metrics                             │ ← Metrics Grid
│ ┌──────────┬──────────┬──────────┐     │
│ │ Tech     │ Win Prob │ Quality  │     │
│ │ 18.5     │ 75%      │ 82       │     │
│ ├──────────┼──────────┼──────────┤     │
│ │ ICS      │ Alpha    │ Risk     │     │
│ │ 85       │ 0.45     │ Moderate │     │
│ └──────────┴──────────┴──────────┘     │
├─────────────────────────────────────────┤
│ Factor Breakdown                        │ ← Expandable
│ ▼ Momentum: 8.5/10  ████████░░          │
│ ▼ Value: 6.2/10     ██████░░░░          │
│ ▼ Quality: 8.0/10   ████████░░          │
│ ▼ Growth: 7.5/10    ███████░░░          │
├─────────────────────────────────────────┤
│ Events & Catalysts                      │ ← Timeline
│ • Earnings: Feb 15 (in 7 days)         │
│ • Dividend: $0.25 (ex-date Feb 10)     │
│ • Insider Buy: 50K shares (Jan 28)     │
├─────────────────────────────────────────┤
│ Options Strategies                      │ ← Expandable
│ ▼ Bull Call Spread                     │
│   Buy $150 Call, Sell $155 Call        │
│   Max Profit: $350  Risk: $150         │
├─────────────────────────────────────────┤
│ Fundamentals                            │ ← Expandable
│ ▼ P/E: 25.3  │ EPS: $6.00             │
│   ROE: 45%   │ Debt/Eq: 1.2           │
├─────────────────────────────────────────┤
│ [Ask Copilot]  [Add to Watchlist]      │ ← Action Buttons
└─────────────────────────────────────────┘
```

---

## 🏗️ Implementation Steps

### Step 1: Backend API Endpoint (Already Exists!)

The `/symbol/{ticker}` endpoint is already implemented in `api_server.py`:

```python
@app.get("/symbol/{ticker}")
async def symbol_detail(ticker: str, days: int = 90):
    history = data_engine.get_price_history(ticker, days)
    fundamentals = data_engine.get_fundamentals(ticker)
    events = get_event_info(ticker)
    
    return SymbolDetailResponse(
        symbol=ticker,
        last_price=history["Close"].iloc[-1],
        history=history.to_dict("records"),
        fundamentals=fundamentals.to_dict(),
        events=events,
    )
```

**Enhancement Needed**: Add MERIT Score, ICS, Quality, Factor Breakdown to response

```python
# Add to SymbolDetailResponse model
class SymbolDetailResponse(BaseModel):
    symbol: str
    last_price: Optional[float]
    change_pct: Optional[float]
    history: List[dict]
    fundamentals: Optional[dict]
    events: Optional[dict]
    
    # NEW FIELDS
    merit_score: Optional[float] = None
    merit_band: Optional[str] = None
    merit_flags: Optional[str] = None
    tech_rating: Optional[float] = None
    win_prob_10d: Optional[float] = None
    quality_score: Optional[float] = None
    ics: Optional[float] = None
    ics_tier: Optional[str] = None
    alpha_score: Optional[float] = None
    risk_score: Optional[str] = None
    
    # Factor breakdown
    momentum_score: Optional[float] = None
    value_score: Optional[float] = None
    quality_factor: Optional[float] = None
    growth_score: Optional[float] = None
    
    # Options
    options_available: bool = False
    options_strategies: Optional[List[dict]] = None
```

### Step 2: Flutter Model

Create `lib/models/symbol_detail.dart`:

```dart
class SymbolDetail {
  final String symbol;
  final double? lastPrice;
  final double? changePct;
  final List<PricePoint> history;
  final Fundamentals? fundamentals;
  final Events? events;
  
  // MERIT & Scores
  final double? meritScore;
  final String? meritBand;
  final String? meritFlags;
  final double? techRating;
  final double? winProb10d;
  final double? qualityScore;
  final double? ics;
  final String? icsTier;
  final double? alphaScore;
  final String? riskScore;
  
  // Factors
  final double? momentumScore;
  final double? valueScore;
  final double? qualityFactor;
  final double? growthScore;
  
  // Options
  final bool optionsAvailable;
  final List<OptionStrategy>? optionsStrategies;
  
  SymbolDetail({
    required this.symbol,
    this.lastPrice,
    this.changePct,
    required this.history,
    this.fundamentals,
    this.events,
    this.meritScore,
    this.meritBand,
    this.meritFlags,
    this.techRating,
    this.winProb10d,
    this.qualityScore,
    this.ics,
    this.icsTier,
    this.alphaScore,
    this.riskScore,
    this.momentumScore,
    this.valueScore,
    this.qualityFactor,
    this.growthScore,
    this.optionsAvailable = false,
    this.optionsStrategies,
  });
  
  factory SymbolDetail.fromJson(Map<String, dynamic> json) {
    return SymbolDetail(
      symbol: json['symbol'] as String,
      lastPrice: json['last_price'] as double?,
      changePct: json['change_pct'] as double?,
      history: (json['history'] as List?)
          ?.map((e) => PricePoint.fromJson(e as Map<String, dynamic>))
          .toList() ?? [],
      fundamentals: json['fundamentals'] != null
          ? Fundamentals.fromJson(json['fundamentals'] as Map<String, dynamic>)
          : null,
      events: json['events'] != null
          ? Events.fromJson(json['events'] as Map<String, dynamic>)
          : null,
      meritScore: json['merit_score'] as double?,
      meritBand: json['merit_band'] as String?,
      meritFlags: json['merit_flags'] as String?,
      techRating: json['tech_rating'] as double?,
      winProb10d: json['win_prob_10d'] as double?,
      qualityScore: json['quality_score'] as double?,
      ics: json['ics'] as double?,
      icsTier: json['ics_tier'] as String?,
      alphaScore: json['alpha_score'] as double?,
      riskScore: json['risk_score'] as String?,
      momentumScore: json['momentum_score'] as double?,
      valueScore: json['value_score'] as double?,
      qualityFactor: json['quality_factor'] as double?,
      growthScore: json['growth_score'] as double?,
      optionsAvailable: json['options_available'] as bool? ?? false,
      optionsStrategies: (json['options_strategies'] as List?)
          ?.map((e) => OptionStrategy.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}

class PricePoint {
  final DateTime date;
  final double open;
  final double high;
  final double low;
  final double close;
  final int volume;
  
  PricePoint({
    required this.date,
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    required this.volume,
  });
  
  factory PricePoint.fromJson(Map<String, dynamic> json) {
    return PricePoint(
      date: DateTime.parse(json['date'] as String),
      open: (json['Open'] as num).toDouble(),
      high: (json['High'] as num).toDouble(),
      low: (json['Low'] as num).toDouble(),
      close: (json['Close'] as num).toDouble(),
      volume: json['Volume'] as int,
    );
  }
}

class Fundamentals {
  final double? pe;
  final double? eps;
  final double? roe;
  final double? debtToEquity;
  final double? marketCap;
  
  Fundamentals({
    this.pe,
    this.eps,
    this.roe,
    this.debtToEquity,
    this.marketCap,
  });
  
  factory Fundamentals.fromJson(Map<String, dynamic> json) {
    return Fundamentals(
      pe: json['pe'] as double?,
      eps: json['eps'] as double?,
      roe: json['roe'] as double?,
      debtToEquity: json['debt_to_equity'] as double?,
      marketCap: json['market_cap'] as double?,
    );
  }
}

class Events {
  final DateTime? nextEarnings;
  final int? daysToEarnings;
  final DateTime? nextDividend;
  final double? dividendAmount;
  final List<InsiderActivity>? insiderActivity;
  
  Events({
    this.nextEarnings,
    this.daysToEarnings,
    this.nextDividend,
    this.dividendAmount,
    this.insiderActivity,
  });
  
  factory Events.fromJson(Map<String, dynamic> json) {
    return Events(
      nextEarnings: json['next_earnings'] != null
          ? DateTime.parse(json['next_earnings'] as String)
          : null,
      daysToEarnings: json['days_to_earnings'] as int?,
      nextDividend: json['next_dividend'] != null
          ? DateTime.parse(json['next_dividend'] as String)
          : null,
      dividendAmount: json['dividend_amount'] as double?,
      insiderActivity: (json['insider_activity'] as List?)
          ?.map((e) => InsiderActivity.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}

class InsiderActivity {
  final DateTime date;
  final String type; // 'buy' or 'sell'
  final int shares;
  final String insider;
  
  InsiderActivity({
    required this.date,
    required this.type,
    required this.shares,
    required this.insider,
  });
  
  factory InsiderActivity.fromJson(Map<String, dynamic> json) {
    return InsiderActivity(
      date: DateTime.parse(json['date'] as String),
      type: json['type'] as String,
      shares: json['shares'] as int,
      insider: json['insider'] as String,
    );
  }
}
```

### Step 3: API Service Method

Add to `lib/services/api_service.dart`:

```dart
Future<SymbolDetail> getSymbolDetail(String ticker) async {
  try {
    final response = await _client.get(
      Uri.parse('$baseUrl/symbol/$ticker'),
      headers: _headers,
    );
    
    if (response.statusCode == 200) {
      final json = jsonDecode(response.body) as Map<String, dynamic>;
      return SymbolDetail.fromJson(json);
    } else {
      throw Exception('Failed to load symbol detail: ${response.statusCode}');
    }
  } catch (e) {
    throw Exception('Error fetching symbol detail: $e');
  }
}
```

### Step 4: Symbol Detail Page UI

Create `lib/screens/symbol_detail/symbol_detail_page.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:fl_chart/fl_chart.dart';

import '../../models/symbol_detail.dart';
import '../../services/api_service.dart';
import '../../theme/app_colors.dart';
import '../../utils/formatters.dart';
import 'widgets/price_chart.dart';
import 'widgets/merit_card.dart';
import 'widgets/metrics_grid.dart';
import 'widgets/factor_breakdown.dart';
import 'widgets/events_timeline.dart';
import 'widgets/options_section.dart';
import 'widgets/fundamentals_section.dart';

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
  
  @override
  void initState() {
    super.initState();
    _detailFuture = ref.read(apiServiceProvider).getSymbolDetail(widget.ticker);
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.ticker),
        actions: [
          IconButton(
            icon: const Icon(Icons.more_vert),
            onPressed: () => _showMenu(context),
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
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, size: 48, color: Colors.red),
                  const SizedBox(height: 16),
                  Text('Error: ${snapshot.error}'),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        _detailFuture = ref.read(apiServiceProvider)
                            .getSymbolDetail(widget.ticker);
                      });
                    },
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          }
          
          final detail = snapshot.data!;
          
          return RefreshIndicator(
            onRefresh: () async {
              setState(() {
                _detailFuture = ref.read(apiServiceProvider)
                    .getSymbolDetail(widget.ticker);
              });
            },
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Price Chart
                  PriceChart(
                    ticker: detail.symbol,
                    history: detail.history,
                    lastPrice: detail.lastPrice,
                    changePct: detail.changePct,
                  ),
                  
                  const SizedBox(height: 24),
                  
                  // MERIT Score Card
                  if (detail.meritScore != null)
                    MeritCard(
                      score: detail.meritScore!,
                      band: detail.meritBand,
                      flags: detail.meritFlags,
                    ),
                  
                  const SizedBox(height: 24),
                  
                  // Key Metrics Grid
                  MetricsGrid(
                    techRating: detail.techRating,
                    winProb: detail.winProb10d,
                    quality: detail.qualityScore,
                    ics: detail.ics,
                    alpha: detail.alphaScore,
                    risk: detail.riskScore,
                  ),
                  
                  const SizedBox(height: 24),
                  
                  // Factor Breakdown
                  if (detail.momentumScore != null)
                    FactorBreakdown(
                      momentum: detail.momentumScore!,
                      value: detail.valueScore ?? 0,
                      quality: detail.qualityFactor ?? 0,
                      growth: detail.growthScore ?? 0,
                    ),
                  
                  const SizedBox(height: 24),
                  
                  // Events Timeline
                  if (detail.events != null)
                    EventsTimeline(events: detail.events!),
                  
                  const SizedBox(height: 24),
                  
                  // Options Strategies
                  if (detail.optionsAvailable && detail.optionsStrategies != null)
                    OptionsSection(strategies: detail.optionsStrategies!),
                  
                  const SizedBox(height: 24),
                  
                  // Fundamentals
                  if (detail.fundamentals != null)
                    FundamentalsSection(fundamentals: detail.fundamentals!),
                  
                  const SizedBox(height: 24),
                  
                  // Action Buttons
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () => _askCopilot(context, detail),
                          icon: const Icon(Icons.chat_bubble_outline),
                          label: const Text('Ask Copilot'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () => _addToWatchlist(detail),
                          icon: const Icon(Icons.star_outline),
                          label: const Text('Add to Watchlist'),
                        ),
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 32),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
  
  void _showMenu(BuildContext context) {
    showModalBottomSheet(
      context: context,
      builder: (context) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ListTile(
            leading: const Icon(Icons.share),
            title: const Text('Share'),
            onTap: () {
              Navigator.pop(context);
              // TODO: Implement share
            },
          ),
          ListTile(
            leading: const Icon(Icons.refresh),
            title: const Text('Refresh'),
            onTap: () {
              Navigator.pop(context);
              setState(() {
                _detailFuture = ref.read(apiServiceProvider)
                    .getSymbolDetail(widget.ticker);
              });
            },
          ),
        ],
      ),
    );
  }
  
  void _askCopilot(BuildContext context, SymbolDetail detail) {
    // Navigate to Copilot with context
    ref.read(copilotContextProvider.notifier).state = detail;
    ref.read(copilotPrefillProvider.notifier).state =
        'Tell me about ${detail.symbol}';
    ref.read(currentTabProvider.notifier).setTab(2); // Copilot tab
    Navigator.pop(context);
  }
  
  void _addToWatchlist(SymbolDetail detail) {
    ref.read(watchlistProvider.notifier).add(
      detail.symbol,
      note: 'MERIT: ${detail.meritScore?.toStringAsFixed(0) ?? "N/A"}',
    );
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('${detail.symbol} added to My Ideas'),
        duration: const Duration(seconds: 2),
      ),
    );
  }
}
```

### Step 5: Widget Components

Create widget files in `lib/screens/symbol_detail/widgets/`:

1. **price_chart.dart** - Candlestick chart using fl_chart
2. **merit_card.dart** - MERIT Score display (reuse from scan_result_card)
3. **metrics_grid.dart** - 2x3 grid of key metrics
4. **factor_breakdown.dart** - Expandable factor scores with bars
5. **events_timeline.dart** - Timeline of upcoming events
6. **options_section.dart** - Expandable options strategies
7. **fundamentals_section.dart** - Expandable fundamentals

### Step 6: Navigation Integration

Update scan_result_card.dart to navigate to Symbol Detail:

```dart
// In ScanResultCard
onTap: () {
  Navigator.push(
    context,
    MaterialPageRoute(
      builder: (context) => SymbolDetailPage(ticker: result.ticker),
    ),
  );
},
```

---

## 📦 Dependencies

Add to `pubspec.yaml`:

```yaml
dependencies:
  fl_chart: ^0.66.0  # For candlestick charts
```

---

## ✅ Acceptance Criteria

1. **Functionality**
   - [ ] Page loads symbol data from API
   - [ ] Displays 90-day candlestick chart
   - [ ] Shows MERIT Score prominently
   - [ ] Displays all key metrics
   - [ ] Shows factor breakdown
   - [ ] Lists upcoming events
   - [ ] Shows options strategies (if available)
   - [ ] Displays fundamentals
   - [ ] "Ask Copilot" navigates with context
   - [ ] "Add to Watchlist" works

2. **Performance**
   - [ ] Loads in < 2 seconds
   - [ ] Smooth scrolling
   - [ ] Pull-to-refresh works
   - [ ] Handles errors gracefully

3. **Design**
   - [ ] Follows app design system
   - [ ] Responsive on all screen sizes
   - [ ] Proper spacing and typography
   - [ ] Accessible (contrast, touch targets)

4. **Testing**
   - [ ] Unit tests for models
   - [ ] Widget tests for components
   - [ ] Integration test for full flow
   - [ ] Manual testing on iOS/Android

---

## 🚀 Implementation Timeline

- **Day 1-2**: Backend API enhancement + Flutter models
- **Day 3-4**: Main page layout + chart widget
- **Day 5-6**: All widget components
- **Day 7**: Navigation integration + testing
- **Day 8**: Polish + bug fixes

**Total**: 8 days (1.5 weeks)

---

## 📝 Next Steps After Symbol Detail Page

1. Enhanced Scanner Features (tier badges, event flags)
2. Ideas Page Redesign (card stack UI)
3. Copilot Enhancements (bubble UI, suggested prompts)
4. My Ideas Upgrade (rich cards, sparklines)
5. Onboarding Flow

---

**Ready to start? Let's begin with Step 1: Backend API Enhancement!**
