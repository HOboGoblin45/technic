/// ScanResult Model
/// 
/// Represents a single stock scan result with all associated metrics,
/// scores, and trade planning information.

class ScanResult {
  final String ticker;
  final String signal;
  final String rrr;
  final String entry;
  final String stop;
  final String target;
  final String note;
  final List<double> sparkline;
  final double? institutionalCoreScore;
  final String? icsTier;
  final double? winProb10d;
  final double? qualityScore;
  final String? playStyle;
  final bool? isUltraRisky;
  final String? profileName;
  final String? profileLabel;
  final String? sector;
  final String? industry;
  final double? techRating;
  final double? alphaScore;
  final double? atrPct;
  final String? eventSummary;
  final String? eventFlags;
  final String? fundamentalSnapshot;
  final List<OptionStrategy> optionStrategies;

  const ScanResult(
    this.ticker,
    this.signal,
    this.rrr,
    this.entry,
    this.stop,
    this.target,
    this.note, [
    this.sparkline = const [],
    this.institutionalCoreScore,
    this.icsTier,
    this.winProb10d,
    this.qualityScore,
    this.playStyle,
    this.isUltraRisky,
    this.profileName,
    this.profileLabel,
    this.sector,
    this.industry,
    this.techRating,
    this.alphaScore,
    this.atrPct,
    this.eventSummary,
    this.eventFlags,
    this.fundamentalSnapshot,
    this.optionStrategies = const [],
  ]);

  factory ScanResult.fromJson(Map<String, dynamic> json) {
    String num(dynamic v) => v == null ? '' : v.toString();
    double? dbl(dynamic v) => v == null ? null : double.tryParse(v.toString());
    bool? bl(dynamic v) {
      if (v == null) return null;
      if (v is bool) return v;
      final s = v.toString().toLowerCase();
      return s == 'true' || s == '1';
    }
    
    final rawSpark = json['sparkline'] ?? json['spark'] ?? [];
    final spark = rawSpark is List
        ? rawSpark.map((e) => double.tryParse(e.toString()) ?? 0).toList()
        : <double>[];
    
    return ScanResult(
      json['ticker']?.toString() ?? '',
      json['signal']?.toString() ?? '',
      json['rrr']?.toString() ?? json['rr']?.toString() ?? '',
      num(json['entry']),
      num(json['stop']),
      num(json['target']),
      json['note']?.toString() ?? '',
      spark,
      dbl(json['InstitutionalCoreScore'] ?? json['ics'] ?? json['ICS']),
      json['ICS_Tier']?.toString() ?? json['Tier']?.toString(),
      dbl(json['win_prob_10d']),
      dbl(json['QualityScore'] ?? json['fundamental_quality_score']),
      json['PlayStyle']?.toString(),
      bl(json['IsUltraRisky']),
      json['Profile']?.toString(),
      json['ProfileLabel']?.toString(),
      json['Sector']?.toString(),
      json['Industry']?.toString(),
      dbl(json['TechRating']),
      dbl(json['AlphaScore']),
      dbl(json['ATR14_pct']),
      json['EventSummary']?.toString(),
      json['EventFlags']?.toString(),
      json['FundamentalSnapshot']?.toString(),
      _parseOptionStrategies(json),
    );
  }

  static List<OptionStrategy> _parseOptionStrategies(Map<String, dynamic> json) {
    final raw = json['option_strategies'] ?? json['OptionStrategies'];
    if (raw is List) {
      return raw
          .whereType<Map<String, dynamic>>()
          .map(OptionStrategy.fromJson)
          .toList();
    }
    return const <OptionStrategy>[];
  }

  Map<String, dynamic> toJson() => {
        'ticker': ticker,
        'signal': signal,
        'rrr': rrr,
        'entry': entry,
        'stop': stop,
        'target': target,
        'note': note,
        'sparkline': sparkline,
        'InstitutionalCoreScore': institutionalCoreScore,
        'ICS_Tier': icsTier,
        'win_prob_10d': winProb10d,
        'QualityScore': qualityScore,
        'PlayStyle': playStyle,
        'IsUltraRisky': isUltraRisky,
        'Profile': profileName,
        'ProfileLabel': profileLabel,
        'Sector': sector,
        'Industry': industry,
        'TechRating': techRating,
        'AlphaScore': alphaScore,
        'ATR14_pct': atrPct,
        'EventSummary': eventSummary,
        'EventFlags': eventFlags,
        'FundamentalSnapshot': fundamentalSnapshot,
        'option_strategies': optionStrategies.map((e) => e.toJson()).toList(),
      };
  
  /// Get tier label based on ICS score
  String getTierLabel() {
    if (icsTier != null && icsTier!.isNotEmpty) return icsTier!;
    if (profileLabel != null && profileLabel!.isNotEmpty) return profileLabel!;
    if (profileName != null && profileName!.isNotEmpty) return profileName!;
    
    final ics = institutionalCoreScore;
    if (ics == null) return '';
    
    if (ics >= 80) return 'CORE';
    if (ics >= 65) return 'SATELLITE';
    return 'WATCH';
  }
  
  /// Check if this is a high-quality setup
  bool get isHighQuality {
    final ics = institutionalCoreScore;
    if (ics == null) return false;
    return ics >= 80;
  }
  
  /// Check if this has options strategies available
  bool get hasOptions => optionStrategies.isNotEmpty;
  
  /// Get count of defined-risk options strategies
  int get definedRiskCount {
    return optionStrategies.where((s) => s.definedRisk).length;
  }
}

/// OptionStrategy Model
class OptionStrategy {
  final String id;
  final String label;
  final String side;
  final String type;
  final bool definedRisk;
  final String? description;
  final String? expiry;
  final double? maxProfit;
  final double? maxLoss;
  final double? probabilityITM;

  const OptionStrategy({
    required this.id,
    required this.label,
    required this.side,
    required this.type,
    required this.definedRisk,
    this.description,
    this.expiry,
    this.maxProfit,
    this.maxLoss,
    this.probabilityITM,
  });

  factory OptionStrategy.fromJson(Map<String, dynamic> json) {
    double? toDbl(dynamic v) => v == null ? null : (v as num?)?.toDouble();
    return OptionStrategy(
      id: (json['id'] ?? json['strategy_id'] ?? json['strategyId'] ?? '').toString(),
      label: (json['label'] ?? json['name'] ?? 'Option strategy').toString(),
      side: (json['side'] ?? 'long').toString(),
      type: (json['type'] ?? json['strategy_type'] ?? 'custom').toString(),
      definedRisk: json['defined_risk'] == true || json['definedRisk'] == true,
      description: json['description']?.toString(),
      expiry: json['expiry']?.toString() ?? json['expiration']?.toString(),
      maxProfit: toDbl(json['max_profit']),
      maxLoss: toDbl(json['max_loss']),
      probabilityITM: toDbl(json['prob_itm'] ?? json['probability_itm']),
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'label': label,
        'side': side,
        'type': type,
        'defined_risk': definedRisk,
        'description': description,
        'expiry': expiry,
        'max_profit': maxProfit,
        'max_loss': maxLoss,
        'prob_itm': probabilityITM,
      };
  
  /// Get short label for UI display
  String getShortLabel() {
    final base = label.isNotEmpty ? label : 'Option';
    if (definedRisk) return '$base (defined risk)';
    return base;
  }
}
