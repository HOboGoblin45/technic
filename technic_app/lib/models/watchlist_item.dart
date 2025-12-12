/// WatchlistItem Model
/// 
/// Represents a saved symbol in the user's watchlist.
library;

class WatchlistItem {
  final String ticker;
  final String? signal;
  final String? note;
  final DateTime addedAt;

  const WatchlistItem({
    required this.ticker,
    this.signal,
    this.note,
    required this.addedAt,
  });

  factory WatchlistItem.fromJson(Map<String, dynamic> json) {
    return WatchlistItem(
      ticker: json['ticker']?.toString() ?? '',
      signal: json['signal']?.toString(),
      note: json['note']?.toString(),
      addedAt: json['addedAt'] != null
          ? DateTime.parse(json['addedAt'].toString())
          : DateTime.now(),
    );
  }

  Map<String, dynamic> toJson() => {
        'ticker': ticker,
        'signal': signal,
        'note': note,
        'addedAt': addedAt.toIso8601String(),
      };
  
  /// Check if item has a signal
  bool get hasSignal => signal != null && signal!.isNotEmpty;
  
  /// Check if item has a note
  bool get hasNote => note != null && note!.isNotEmpty;
  
  /// Get days since added
  int get daysSinceAdded {
    return DateTime.now().difference(addedAt).inDays;
  }
}
