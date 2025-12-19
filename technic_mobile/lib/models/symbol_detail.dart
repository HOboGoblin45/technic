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
    // Helper for safe double parsing
    double? toDouble(dynamic v) {
      if (v == null) return null;
      if (v is num) return v.toDouble();
      return double.tryParse(v.toString());
    }

    return SymbolDetail(
      symbol: json['symbol']?.toString() ?? '',
      lastPrice: toDouble(json['last_price']),
      changePct: toDouble(json['change_pct']),
      history: (json['history'] as List?)
          ?.map((e) => PricePoint.fromJson(Map<String, dynamic>.from(e as Map)))
          .toList() ?? [],
      fundamentals: json['fundamentals'] != null
          ? Fundamentals.fromJson(Map<String, dynamic>.from(json['fundamentals'] as Map))
          : null,
      events: json['events'] != null
          ? EventInfo.fromJson(Map<String, dynamic>.from(json['events'] as Map))
          : null,
      meritScore: toDouble(json['merit_score']),
      meritBand: json['merit_band']?.toString(),
      meritFlags: json['merit_flags']?.toString(),
      meritSummary: json['merit_summary']?.toString(),
      techRating: toDouble(json['tech_rating']),
      winProb10d: toDouble(json['win_prob_10d']),
      qualityScore: toDouble(json['quality_score']),
      ics: toDouble(json['ics']),
      icsTier: json['ics_tier']?.toString(),
      alphaScore: toDouble(json['alpha_score']),
      riskScore: json['risk_score']?.toString(),
      momentumScore: toDouble(json['momentum_score']),
      valueScore: toDouble(json['value_score']),
      qualityFactor: toDouble(json['quality_factor']),
      growthScore: toDouble(json['growth_score']),
      optionsAvailable: json['options_available'] == true,
    );
  }

  /// Create a copy with updated fields
  SymbolDetail copyWith({
    String? symbol,
    double? lastPrice,
    double? changePct,
    List<PricePoint>? history,
    Fundamentals? fundamentals,
    EventInfo? events,
    double? meritScore,
    String? meritBand,
    String? meritFlags,
    String? meritSummary,
    double? techRating,
    double? winProb10d,
    double? qualityScore,
    double? ics,
    String? icsTier,
    double? alphaScore,
    String? riskScore,
    double? momentumScore,
    double? valueScore,
    double? qualityFactor,
    double? growthScore,
    bool? optionsAvailable,
  }) {
    return SymbolDetail(
      symbol: symbol ?? this.symbol,
      lastPrice: lastPrice ?? this.lastPrice,
      changePct: changePct ?? this.changePct,
      history: history ?? this.history,
      fundamentals: fundamentals ?? this.fundamentals,
      events: events ?? this.events,
      meritScore: meritScore ?? this.meritScore,
      meritBand: meritBand ?? this.meritBand,
      meritFlags: meritFlags ?? this.meritFlags,
      meritSummary: meritSummary ?? this.meritSummary,
      techRating: techRating ?? this.techRating,
      winProb10d: winProb10d ?? this.winProb10d,
      qualityScore: qualityScore ?? this.qualityScore,
      ics: ics ?? this.ics,
      icsTier: icsTier ?? this.icsTier,
      alphaScore: alphaScore ?? this.alphaScore,
      riskScore: riskScore ?? this.riskScore,
      momentumScore: momentumScore ?? this.momentumScore,
      valueScore: valueScore ?? this.valueScore,
      qualityFactor: qualityFactor ?? this.qualityFactor,
      growthScore: growthScore ?? this.growthScore,
      optionsAvailable: optionsAvailable ?? this.optionsAvailable,
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
    double toDouble(dynamic v) => (v as num?)?.toDouble() ?? 0.0;
    return PricePoint(
      date: DateTime.tryParse(json['date']?.toString() ?? '') ?? DateTime.now(),
      open: toDouble(json['Open'] ?? json['open']),
      high: toDouble(json['High'] ?? json['high']),
      low: toDouble(json['Low'] ?? json['low']),
      close: toDouble(json['Close'] ?? json['close']),
      volume: (json['Volume'] ?? json['volume'] as num?)?.toInt() ?? 0,
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
    double? toDouble(dynamic v) {
      if (v == null) return null;
      if (v is num) return v.toDouble();
      return double.tryParse(v.toString());
    }
    return Fundamentals(
      pe: toDouble(json['pe']),
      eps: toDouble(json['eps']),
      roe: toDouble(json['roe']),
      debtToEquity: toDouble(json['debt_to_equity']),
      marketCap: toDouble(json['market_cap']),
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
    double? toDouble(dynamic v) {
      if (v == null) return null;
      if (v is num) return v.toDouble();
      return double.tryParse(v.toString());
    }
    return EventInfo(
      nextEarnings: json['next_earnings'] != null
          ? DateTime.tryParse(json['next_earnings'].toString())
          : null,
      daysToEarnings: (json['days_to_earnings'] as num?)?.toInt(),
      nextDividend: json['next_dividend'] != null
          ? DateTime.tryParse(json['next_dividend'].toString())
          : null,
      dividendAmount: toDouble(json['dividend_amount']),
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
