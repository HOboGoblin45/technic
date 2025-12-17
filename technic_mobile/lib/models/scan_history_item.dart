/// Scan History Item Model
/// 
/// Represents a saved scan result with metadata
library;

import 'scan_result.dart';

class ScanHistoryItem {
  final String id;
  final DateTime timestamp;
  final List<ScanResult> results;
  final Map<String, dynamic> scanParams;
  final int resultCount;
  final double? averageMerit;

  const ScanHistoryItem({
    required this.id,
    required this.timestamp,
    required this.results,
    required this.scanParams,
    required this.resultCount,
    this.averageMerit,
  });

  factory ScanHistoryItem.fromJson(Map<String, dynamic> json) {
    return ScanHistoryItem(
      id: json['id']?.toString() ?? '',
      timestamp: json['timestamp'] != null
          ? DateTime.parse(json['timestamp'].toString())
          : DateTime.now(),
      results: json['results'] != null
          ? (json['results'] as List)
              .map((r) => ScanResult.fromJson(r as Map<String, dynamic>))
              .toList()
          : [],
      scanParams: json['scanParams'] as Map<String, dynamic>? ?? {},
      resultCount: json['resultCount'] as int? ?? 0,
      averageMerit: json['averageMerit'] as double?,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'timestamp': timestamp.toIso8601String(),
        'results': results.map((r) => r.toJson()).toList(),
        'scanParams': scanParams,
        'resultCount': resultCount,
        'averageMerit': averageMerit,
      };

  /// Get formatted timestamp
  String get formattedTime {
    final now = DateTime.now();
    final difference = now.difference(timestamp);

    if (difference.inMinutes < 1) {
      return 'Just now';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else if (difference.inDays < 7) {
      return '${difference.inDays}d ago';
    } else {
      return '${timestamp.month}/${timestamp.day}/${timestamp.year}';
    }
  }

  /// Get scan type from params
  String get scanType {
    final riskProfile = scanParams['risk_profile'] as String?;
    if (riskProfile != null) {
      return riskProfile.split('_').map((word) {
        return word[0].toUpperCase() + word.substring(1);
      }).join(' ');
    }
    return 'Custom';
  }

  /// Get formatted date/time
  String get formattedDateTime {
    final hour = timestamp.hour > 12 ? timestamp.hour - 12 : timestamp.hour;
    final period = timestamp.hour >= 12 ? 'PM' : 'AM';
    return '${timestamp.month}/${timestamp.day}/${timestamp.year} $hour:${timestamp.minute.toString().padLeft(2, '0')} $period';
  }
}
