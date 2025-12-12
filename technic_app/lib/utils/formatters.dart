/// Formatters and Utility Functions
/// 
/// Common formatting functions used throughout the app.

/// Format a field value, returning '?' for null/empty/NaN
String formatField(String? value) {
  if (value == null) return '?';
  final trimmed = value.trim();
  if (trimmed.isEmpty) return '?';
  if (trimmed.toLowerCase() == 'nan') return '?';
  return trimmed;
}

/// Format a number with specified decimal places
String formatNumber(double? value, {int decimals = 2}) {
  if (value == null) return '?';
  if (value.isNaN || value.isInfinite) return '?';
  return value.toStringAsFixed(decimals);
}

/// Format a percentage
String formatPercent(double? value, {int decimals = 1, bool includeSign = false}) {
  if (value == null) return '?';
  if (value.isNaN || value.isInfinite) return '?';
  
  final formatted = value.toStringAsFixed(decimals);
  if (includeSign && value > 0) {
    return '+$formatted%';
  }
  return '$formatted%';
}

/// Format currency
String formatCurrency(double? value, {String symbol = '\$', int decimals = 2}) {
  if (value == null) return '?';
  if (value.isNaN || value.isInfinite) return '?';
  
  final formatted = value.toStringAsFixed(decimals);
  return '$symbol$formatted';
}

/// Format large numbers with K/M/B suffixes
String formatCompact(double? value, {int decimals = 1}) {
  if (value == null) return '?';
  if (value.isNaN || value.isInfinite) return '?';
  
  if (value.abs() >= 1e9) {
    return '${(value / 1e9).toStringAsFixed(decimals)}B';
  } else if (value.abs() >= 1e6) {
    return '${(value / 1e6).toStringAsFixed(decimals)}M';
  } else if (value.abs() >= 1e3) {
    return '${(value / 1e3).toStringAsFixed(decimals)}K';
  }
  return value.toStringAsFixed(decimals);
}

/// Format date/time in local timezone
String formatLocalTime(DateTime dateTime) {
  final local = dateTime.toLocal();
  final now = DateTime.now();
  final today = DateTime(now.year, now.month, now.day);
  final targetDay = DateTime(local.year, local.month, local.day);
  final daysDiff = today.difference(targetDay).inDays;
  
  final hour12 = local.hour % 12 == 0 ? 12 : local.hour % 12;
  final mm = local.minute.toString().padLeft(2, '0');
  final ampm = local.hour >= 12 ? 'PM' : 'AM';
  
  String dayLabel;
  if (daysDiff == 0) {
    dayLabel = 'Today';
  } else if (daysDiff == 1) {
    dayLabel = 'Yesterday';
  } else if (daysDiff < 7) {
    dayLabel = _getDayName(local.weekday);
  } else {
    dayLabel = '${_getMonthAbbr(local.month)} ${local.day}';
    if (local.year != now.year) {
      dayLabel += ', ${local.year}';
    }
  }
  
  return '$dayLabel $hour12:$mm $ampm';
}

/// Format date only
String formatDate(DateTime dateTime) {
  final local = dateTime.toLocal();
  final now = DateTime.now();
  final today = DateTime(now.year, now.month, now.day);
  final targetDay = DateTime(local.year, local.month, local.day);
  final daysDiff = today.difference(targetDay).inDays;
  
  if (daysDiff == 0) {
    return 'Today';
  } else if (daysDiff == 1) {
    return 'Yesterday';
  } else if (daysDiff < 7) {
    return _getDayName(local.weekday);
  } else {
    final formatted = '${_getMonthAbbr(local.month)} ${local.day}';
    if (local.year != now.year) {
      return '$formatted, ${local.year}';
    }
    return formatted;
  }
}

/// Format time only
String formatTime(DateTime dateTime) {
  final local = dateTime.toLocal();
  final hour12 = local.hour % 12 == 0 ? 12 : local.hour % 12;
  final mm = local.minute.toString().padLeft(2, '0');
  final ampm = local.hour >= 12 ? 'PM' : 'AM';
  return '$hour12:$mm $ampm';
}

/// Format duration (e.g., "2h 30m", "45s")
String formatDuration(Duration duration) {
  if (duration.inDays > 0) {
    return '${duration.inDays}d ${duration.inHours % 24}h';
  } else if (duration.inHours > 0) {
    return '${duration.inHours}h ${duration.inMinutes % 60}m';
  } else if (duration.inMinutes > 0) {
    return '${duration.inMinutes}m';
  } else {
    return '${duration.inSeconds}s';
  }
}

/// Format relative time (e.g., "2 hours ago", "just now")
String formatRelativeTime(DateTime dateTime) {
  final now = DateTime.now();
  final difference = now.difference(dateTime);
  
  if (difference.inSeconds < 60) {
    return 'just now';
  } else if (difference.inMinutes < 60) {
    final mins = difference.inMinutes;
    return '$mins ${mins == 1 ? 'minute' : 'minutes'} ago';
  } else if (difference.inHours < 24) {
    final hours = difference.inHours;
    return '$hours ${hours == 1 ? 'hour' : 'hours'} ago';
  } else if (difference.inDays < 7) {
    final days = difference.inDays;
    return '$days ${days == 1 ? 'day' : 'days'} ago';
  } else if (difference.inDays < 30) {
    final weeks = (difference.inDays / 7).floor();
    return '$weeks ${weeks == 1 ? 'week' : 'weeks'} ago';
  } else if (difference.inDays < 365) {
    final months = (difference.inDays / 30).floor();
    return '$months ${months == 1 ? 'month' : 'months'} ago';
  } else {
    final years = (difference.inDays / 365).floor();
    return '$years ${years == 1 ? 'year' : 'years'} ago';
  }
}

/// Truncate text with ellipsis
String truncate(String text, int maxLength, {String ellipsis = '...'}) {
  if (text.length <= maxLength) return text;
  return '${text.substring(0, maxLength - ellipsis.length)}$ellipsis';
}

/// Capitalize first letter
String capitalize(String text) {
  if (text.isEmpty) return text;
  return text[0].toUpperCase() + text.substring(1);
}

/// Title case (capitalize each word)
String titleCase(String text) {
  if (text.isEmpty) return text;
  return text.split(' ').map((word) => capitalize(word)).join(' ');
}

/// Format ticker symbol (uppercase, trim)
String formatTicker(String ticker) {
  return ticker.trim().toUpperCase();
}

/// Parse double safely
double? parseDouble(dynamic value) {
  if (value == null) return null;
  if (value is double) return value;
  if (value is int) return value.toDouble();
  if (value is String) return double.tryParse(value);
  return null;
}

/// Parse int safely
int? parseInt(dynamic value) {
  if (value == null) return null;
  if (value is int) return value;
  if (value is double) return value.toInt();
  if (value is String) return int.tryParse(value);
  return null;
}

/// Parse bool safely
bool? parseBool(dynamic value) {
  if (value == null) return null;
  if (value is bool) return value;
  if (value is String) {
    final lower = value.toLowerCase();
    if (lower == 'true' || lower == '1' || lower == 'yes') return true;
    if (lower == 'false' || lower == '0' || lower == 'no') return false;
  }
  if (value is int) return value != 0;
  return null;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

String _getDayName(int weekday) {
  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  return days[weekday - 1];
}

String _getMonthAbbr(int month) {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return months[month - 1];
}
