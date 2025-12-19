/// Symbol Detail Model
/// 
/// Comprehensive model for symbol detail page data including:
/// - Price history (candlestick data)
/// - MERIT Score and quantitative metrics
/// - Fundamentals
/// - Events (earnings, dividends)
/// - Factor breakdown
library;

class SymbolDetail {
  final String symbol;
  final double? lastPrice;
  final double? changePct;
  final List<PricePoint> history;
  final Fundamentals? fundamentals;
  final EventInfo? events;
  
  // MERIT & Scores
  final double? meritScore;
  final String? meritBand;
  final String? meritFlags;
  final String? meritSummary;
  final double? techRating;
  final double? winProb10d;
  final double? qualityScore;
  final double? ics;
  final String? icsTier;
  final double? alphaScore;
  final String? riskScore;
  
  // Factor breakdown
  final double? momentumScore;
  final double? valueScore;
  final double? qualityFactor;
  final double? growthScore;
  
  // Options
  final bool optionsAvailable;
  
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
    this.meritSummary,
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
          ? EventInfo.fromJson(json['events'] as Map<String, dynamic>)
          : null,
      meritScore: json['merit_score'] as double?,
      meritBand: json['merit_band'] as String?,
      meritFlags: json['merit_flags'] as String?,
      meritSummary: json['merit_summary'] as String?,
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
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'symbol': symbol,
      'last_price': lastPrice,
      'change_pct': changePct,
      'history': history.map((e) => e.toJson()).toList(),
      'fundamentals': fundamentals?.toJson(),
      'events': events?.toJson(),
      'merit_score': meritScore,
      'merit_band': meritBand,
      'merit_flags': meritFlags,
      'merit_summary': meritSummary,
      'tech_rating': techRating,
      'win_prob_10d': winProb10d,
      'quality_score': qualityScore,
      'ics': ics,
      'ics_tier': icsTier,
      'alpha_score': alphaScore,
      'risk_score': riskScore,
      'momentum_score': momentumScore,
      'value_score': valueScore,
      'quality_factor': qualityFactor,
      'growth_score': growthScore,
      'options_available': optionsAvailable,
    };
  }
}

/// Price Point for candlestick chart
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
  
  Map<String, dynamic> toJson() {
    return {
      'date': date.toIso8601String(),
      'Open': open,
      'High': high,
      'Low': low,
      'Close': close,
      'Volume': volume,
    };
  }
}

/// Fundamental metrics
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
  
  Map<String, dynamic> toJson() {
    return {
      'pe': pe,
      'eps': eps,
      'roe': roe,
      'debt_to_equity': debtToEquity,
      'market_cap': marketCap,
    };
  }
}

/// Event information (earnings, dividends)
class EventInfo {
  final DateTime? nextEarnings;
  final int? daysToEarnings;
  final DateTime? nextDividend;
  final double? dividendAmount;
  
  EventInfo({
    this.nextEarnings,
    this.daysToEarnings,
    this.nextDividend,
    this.dividendAmount,
  });
  
  factory EventInfo.fromJson(Map<String, dynamic> json) {
    return EventInfo(
      nextEarnings: json['next_earnings'] != null
          ? DateTime.parse(json['next_earnings'] as String)
          : null,
      daysToEarnings: json['days_to_earnings'] as int?,
      nextDividend: json['next_dividend'] != null
          ? DateTime.parse(json['next_dividend'] as String)
          : null,
      dividendAmount: json['dividend_amount'] as double?,
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'next_earnings': nextEarnings?.toIso8601String(),
      'days_to_earnings': daysToEarnings,
      'next_dividend': nextDividend?.toIso8601String(),
      'dividend_amount': dividendAmount,
    };
  }
}
