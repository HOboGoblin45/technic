/// ScoreboardSlice Model
/// 
/// Represents a performance metric slice for a trading strategy category.
library;

import 'package:flutter/material.dart';

class ScoreboardSlice {
  final String label;
  final String pnl;
  final String winRate;
  final String horizon;
  final Color accent;

  const ScoreboardSlice(
    this.label,
    this.pnl,
    this.winRate,
    this.horizon,
    this.accent,
  );

  factory ScoreboardSlice.fromJson(Map<String, dynamic> json) {
    final accentStr = json['accent']?.toString();
    return ScoreboardSlice(
      json['label']?.toString() ?? '',
      json['pnl']?.toString() ?? '',
      json['winRate']?.toString() ?? json['win_rate']?.toString() ?? '',
      json['horizon']?.toString() ?? '',
      accentStr != null ? _colorFromHex(accentStr) : const Color(0xFFB0CAFF),
    );
  }

  Map<String, dynamic> toJson() {
    // Convert Color to hex string (ARGB format)
    // Using the recommended approach for Flutter 3.10+
    final alpha = (accent.a * 255.0).round().clamp(0, 255);
    final red = (accent.r * 255.0).round().clamp(0, 255);
    final green = (accent.g * 255.0).round().clamp(0, 255);
    final blue = (accent.b * 255.0).round().clamp(0, 255);
    
    final argb = '${alpha.toRadixString(16).padLeft(2, '0')}'
        '${red.toRadixString(16).padLeft(2, '0')}'
        '${green.toRadixString(16).padLeft(2, '0')}'
        '${blue.toRadixString(16).padLeft(2, '0')}';
    
    return {
      'label': label,
      'pnl': pnl,
      'winRate': winRate,
      'horizon': horizon,
      'accent': '#${argb.substring(2)}', // Remove alpha for hex color
    };
  }
  
  /// Check if PnL is positive
  bool get isPositive => !pnl.startsWith('-');
  
  /// Get numeric PnL value (if parseable)
  double? get pnlValue {
    final cleaned = pnl.replaceAll(RegExp(r'[^0-9.-]'), '');
    return double.tryParse(cleaned);
  }
  
  /// Get numeric win rate value (if parseable)
  double? get winRateValue {
    final cleaned = winRate.replaceAll(RegExp(r'[^0-9.]'), '');
    return double.tryParse(cleaned);
  }
  
  /// Helper function to convert hex string to Color
  static Color _colorFromHex(String hex) {
    final buffer = StringBuffer();
    if (hex.length == 6 || hex.length == 7) buffer.write('ff');
    buffer.write(hex.replaceFirst('#', ''));
    return Color(int.parse(buffer.toString(), radix: 16));
  }
}
