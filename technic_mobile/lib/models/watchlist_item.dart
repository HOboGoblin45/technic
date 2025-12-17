/// WatchlistItem Model
/// 
/// Represents a saved symbol in the user's watchlist.
library;

class WatchlistItem {
  final String ticker;
  final String? signal;
  final String? note;
  final List<String> tags;
  final DateTime addedAt;

  const WatchlistItem({
    required this.ticker,
    this.signal,
    this.note,
    this.tags = const [],
    required this.addedAt,
  });

  factory WatchlistItem.fromJson(Map<String, dynamic> json) {
    return WatchlistItem(
      ticker: json['ticker']?.toString() ?? '',
      signal: json['signal']?.toString(),
      note: json['note']?.toString(),
      tags: json['tags'] != null
          ? List<String>.from(json['tags'] as List)
          : [],
      addedAt: json['addedAt'] != null
          ? DateTime.parse(json['addedAt'].toString())
          : DateTime.now(),
    );
  }

  Map<String, dynamic> toJson() => {
        'ticker': ticker,
        'signal': signal,
        'note': note,
        'tags': tags,
        'addedAt': addedAt.toIso8601String(),
      };
  
  /// Check if item has a signal
  bool get hasSignal => signal != null && signal!.isNotEmpty;
  
  /// Check if item has a note
  bool get hasNote => note != null && note!.isNotEmpty;
  
  /// Check if item has tags
  bool get hasTags => tags.isNotEmpty;
  
  /// Get days since added
  int get daysSinceAdded {
    return DateTime.now().difference(addedAt).inDays;
  }
  
  /// Copy with method for updating fields
  WatchlistItem copyWith({
    String? ticker,
    String? signal,
    String? note,
    List<String>? tags,
    DateTime? addedAt,
  }) {
    return WatchlistItem(
      ticker: ticker ?? this.ticker,
      signal: signal ?? this.signal,
      note: note ?? this.note,
      tags: tags ?? this.tags,
      addedAt: addedAt ?? this.addedAt,
    );
  }
}
