/// Idea Model
/// 
/// Represents a trade idea with rationale, plan, and optional options strategy.
library;

class Idea {
  final String title;
  final String ticker;
  final String meta;
  final String plan;
  final List<double> sparkline;
  final Map<String, dynamic>? option;

  const Idea(
    this.title,
    this.ticker,
    this.meta,
    this.plan,
    this.sparkline, {
    this.option,
  });

  factory Idea.fromJson(Map<String, dynamic> json) {
    final rawSpark = json['sparkline'] ?? json['spark'] ?? [];
    final spark = rawSpark is List
        ? rawSpark.map((e) => double.tryParse(e.toString()) ?? 0).toList()
        : <double>[];
    
    return Idea(
      json['title']?.toString() ?? '',
      json['ticker']?.toString() ?? '',
      json['meta']?.toString() ?? '',
      json['plan']?.toString() ?? '',
      spark,
      option: json['option'] as Map<String, dynamic>?,
    );
  }

  Map<String, dynamic> toJson() => {
        'title': title,
        'ticker': ticker,
        'meta': meta,
        'plan': plan,
        'sparkline': sparkline,
        'option': option,
      };
  
  /// Check if this idea has an options component
  bool get hasOption => option != null && option!.isNotEmpty;
  
  /// Get option contract type (call/put)
  String? get optionType => option?['contract_type']?.toString();
  
  /// Get option strike price
  String? get optionStrike => option?['strike']?.toString();
  
  /// Get option expiration
  String? get optionExpiration => option?['expiration']?.toString();
  
  /// Get option delta
  String? get optionDelta => option?['delta']?.toString();
  
  /// Check if sparkline shows upward trend
  bool get isTrendingUp {
    if (sparkline.length < 2) return true;
    return sparkline.last >= sparkline.first;
  }

  /// Create a copy with updated fields
  Idea copyWith({
    String? title,
    String? ticker,
    String? meta,
    String? plan,
    List<double>? sparkline,
    Map<String, dynamic>? option,
  }) {
    return Idea(
      title ?? this.title,
      ticker ?? this.ticker,
      meta ?? this.meta,
      plan ?? this.plan,
      sparkline ?? this.sparkline,
      option: option ?? this.option,
    );
  }
}
