/// Price Alert Model
/// 
/// Represents a price alert for a watchlist symbol.
library;

/// Alert type enumeration
enum AlertType {
  priceAbove('Price Above'),
  priceBelow('Price Below'),
  percentChange('Percent Change');

  const AlertType(this.displayName);
  final String displayName;
}

/// Price Alert Model
class PriceAlert {
  final String id;
  final String ticker;
  final AlertType type;
  final double targetValue;
  final bool isActive;
  final DateTime createdAt;
  final DateTime? triggeredAt;
  final String? note;

  const PriceAlert({
    required this.id,
    required this.ticker,
    required this.type,
    required this.targetValue,
    this.isActive = true,
    required this.createdAt,
    this.triggeredAt,
    this.note,
  });

  /// Create a new alert with generated ID
  factory PriceAlert.create({
    required String ticker,
    required AlertType type,
    required double targetValue,
    String? note,
  }) {
    // Generate a simple unique ID using timestamp and random number
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final random = (DateTime.now().microsecond * 1000).toString();
    final id = '$ticker-$timestamp-$random';
    
    return PriceAlert(
      id: id,
      ticker: ticker,
      type: type,
      targetValue: targetValue,
      createdAt: DateTime.now(),
      note: note,
    );
  }

  /// Check if alert has been triggered
  bool get isTriggered => triggeredAt != null;

  /// Get formatted target value
  String get formattedTarget {
    switch (type) {
      case AlertType.priceAbove:
      case AlertType.priceBelow:
        return '\$${targetValue.toStringAsFixed(2)}';
      case AlertType.percentChange:
        return '${targetValue.toStringAsFixed(1)}%';
    }
  }

  /// Get alert description
  String get description {
    switch (type) {
      case AlertType.priceAbove:
        return '$ticker above $formattedTarget';
      case AlertType.priceBelow:
        return '$ticker below $formattedTarget';
      case AlertType.percentChange:
        return '$ticker changes by $formattedTarget';
    }
  }

  /// Copy with method
  PriceAlert copyWith({
    String? id,
    String? ticker,
    AlertType? type,
    double? targetValue,
    bool? isActive,
    DateTime? createdAt,
    DateTime? triggeredAt,
    String? note,
  }) {
    return PriceAlert(
      id: id ?? this.id,
      ticker: ticker ?? this.ticker,
      type: type ?? this.type,
      targetValue: targetValue ?? this.targetValue,
      isActive: isActive ?? this.isActive,
      createdAt: createdAt ?? this.createdAt,
      triggeredAt: triggeredAt ?? this.triggeredAt,
      note: note ?? this.note,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'ticker': ticker,
      'type': type.name,
      'targetValue': targetValue,
      'isActive': isActive,
      'createdAt': createdAt.toIso8601String(),
      'triggeredAt': triggeredAt?.toIso8601String(),
      'note': note,
    };
  }

  /// Create from JSON
  factory PriceAlert.fromJson(Map<String, dynamic> json) {
    return PriceAlert(
      id: json['id']?.toString() ?? '',
      ticker: json['ticker']?.toString() ?? '',
      type: AlertType.values.firstWhere(
        (e) => e.name == json['type']?.toString(),
        orElse: () => AlertType.priceAbove,
      ),
      targetValue: (json['targetValue'] as num?)?.toDouble() ?? 0.0,
      isActive: json['isActive'] == true,
      createdAt: DateTime.tryParse(json['createdAt']?.toString() ?? '') ?? DateTime.now(),
      triggeredAt: json['triggeredAt'] != null
          ? DateTime.tryParse(json['triggeredAt'].toString())
          : null,
      note: json['note']?.toString(),
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is PriceAlert && other.id == id;
  }

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() {
    return 'PriceAlert(id: $id, ticker: $ticker, type: ${type.name}, '
        'targetValue: $targetValue, isActive: $isActive)';
  }
}
