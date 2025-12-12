/// MarketMover Model
/// 
/// Represents a market mover (stock with significant price movement)
/// with delta, note, and sparkline data.
library;

class MarketMover {
  final String ticker;
  final String delta;
  final String note;
  final bool isPositive;
  final List<double> sparkline;

  const MarketMover(
    this.ticker,
    this.delta,
    this.note,
    this.isPositive, [
    this.sparkline = const [],
  ]);

  factory MarketMover.fromJson(Map<String, dynamic> json) {
    final deltaRaw = json['delta']?.toString() ?? '';
    final delta = deltaRaw.isEmpty
        ? ''
        : deltaRaw.startsWith('+') || deltaRaw.startsWith('-')
            ? deltaRaw
            : '+$deltaRaw';
    
    final isPositiveField = json['isPositive'] ?? json['is_positive'];
    final isPos = isPositiveField is bool
        ? isPositiveField
        : delta.startsWith('+') ||
            (json['change']?.toString().startsWith('+') ?? false);
    
    final rawSpark = json['sparkline'] ?? json['spark'] ?? [];
    final spark = rawSpark is List
        ? rawSpark.map((e) => double.tryParse(e.toString()) ?? 0).toList()
        : <double>[];
    
    return MarketMover(
      json['ticker']?.toString() ?? '',
      delta,
      json['note']?.toString() ?? json['label']?.toString() ?? '',
      isPos,
      spark,
    );
  }

  Map<String, dynamic> toJson() => {
        'ticker': ticker,
        'delta': delta,
        'note': note,
        'isPositive': isPositive,
        'sparkline': sparkline,
      };
  
  /// Get formatted delta for display
  String getFormattedDelta() {
    if (delta.isEmpty) return '0.0%';
    return delta;
  }
  
  /// Check if sparkline shows upward trend
  bool get isTrendingUp {
    if (sparkline.length < 2) return isPositive;
    return sparkline.last >= sparkline.first;
  }
}
