/// ScannerBundle Model
/// 
/// Aggregates all data needed for the Scanner page:
/// - Market movers
/// - Scan results
/// - Scoreboard slices
/// - Progress information
library;

import 'market_mover.dart';
import 'scan_result.dart';
import 'scoreboard_slice.dart';

class ScannerBundle {
  final List<MarketMover> movers;
  final List<ScanResult> scanResults;
  final List<ScoreboardSlice> scoreboard;
  final String? progress;
  final int? universeSize;  // Total symbols in selected universe
  final int? symbolsScanned;  // Actual symbols scanned

  const ScannerBundle({
    required this.movers,
    required this.scanResults,
    required this.scoreboard,
    this.progress,
    this.universeSize,
    this.symbolsScanned,
  });

  factory ScannerBundle.fromJson(Map<String, dynamic> json) {
    return ScannerBundle(
      movers: (json['movers'] as List<dynamic>?)
              ?.map((e) => MarketMover.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      scanResults: (json['scanResults'] as List<dynamic>?)
              ?.map((e) => ScanResult.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      scoreboard: (json['scoreboard'] as List<dynamic>?)
              ?.map((e) => ScoreboardSlice.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      progress: json['progress']?.toString(),
      universeSize: json['universe_size'] as int?,
      symbolsScanned: json['symbols_scanned'] as int?,
    );
  }

  Map<String, dynamic> toJson() => {
        'movers': movers.map((e) => e.toJson()).toList(),
        'scanResults': scanResults.map((e) => e.toJson()).toList(),
        'scoreboard': scoreboard.map((e) => e.toJson()).toList(),
        'progress': progress,
        'universe_size': universeSize,
        'symbols_scanned': symbolsScanned,
      };
  
  /// Check if bundle has any data
  bool get isEmpty => movers.isEmpty && scanResults.isEmpty && scoreboard.isEmpty;
  
  /// Check if bundle has scan results
  bool get hasResults => scanResults.isNotEmpty;
  
  /// Check if bundle has movers
  bool get hasMovers => movers.isNotEmpty;
  
  /// Check if bundle has scoreboard data
  bool get hasScoreboard => scoreboard.isNotEmpty;
  
  /// Get count of CORE tier results
  int get coreCount {
    return scanResults.where((r) => r.getTierLabel() == 'CORE').length;
  }
  
  /// Get count of SATELLITE tier results
  int get satelliteCount {
    return scanResults.where((r) => r.getTierLabel() == 'SATELLITE').length;
  }
}
