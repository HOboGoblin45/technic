/// Helper Functions
/// 
/// Utility functions used throughout the Technic app.
library;

import 'package:flutter/material.dart';

/// Creates a toned version of a color with the specified opacity
Color tone(Color base, double opacity) =>
    base.withAlpha((opacity * 255).clamp(0, 255).round());

/// Formats a field value, returning '?' for null, empty, or 'nan' values
String fmtField(String? v) {
  if (v == null) return '?';
  final trimmed = v.trim();
  if (trimmed.isEmpty) return '?';
  if (trimmed.toLowerCase() == 'nan') return '?';
  return trimmed;
}

/// Formats a DateTime to a human-readable local time string
/// Examples: "Today 3:45p", "Yesterday 10:30a", "Jan 15 2:15p"
String fmtLocalTime(DateTime dt) {
  final local = dt.toLocal();
  final today = DateTime.now();
  final daysDiff = DateTime(today.year, today.month, today.day)
      .difference(DateTime(local.year, local.month, local.day))
      .inDays;
  final hour12 = local.hour % 12 == 0 ? 12 : local.hour % 12;
  final mm = local.minute.toString().padLeft(2, '0');
  final ampm = local.hour >= 12 ? 'p' : 'a';
  
  String dayLabel;
  if (daysDiff == 0) {
    dayLabel = 'Today';
  } else if (daysDiff == 1) {
    dayLabel = 'Yesterday';
  } else {
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    final month = months[local.month - 1];
    final yearSuffix = local.year == today.year ? '' : ' ${local.year}';
    dayLabel = '$month ${local.day}$yearSuffix';
  }
  
  return '$dayLabel $hour12:$mm$ampm';
}

/// Converts a hex color string to a Color object
/// Supports both 6-character (RRGGBB) and 7-character (#RRGGBB) formats
/// Returns Colors.transparent if the hex string is invalid
Color colorFromHex(String hex) {
  try {
    final cleanHex = hex.replaceFirst('#', '').trim();
    if (cleanHex.isEmpty || cleanHex.length < 6) {
      return Colors.transparent;
    }
    final buffer = StringBuffer();
    if (cleanHex.length == 6) buffer.write('ff');
    buffer.write(cleanHex.substring(0, cleanHex.length > 8 ? 8 : cleanHex.length));
    return Color(int.parse(buffer.toString(), radix: 16));
  } catch (e) {
    return Colors.transparent;
  }
}
