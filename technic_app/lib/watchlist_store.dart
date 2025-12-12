import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _kWatchlistKey = 'technic_watchlist_v1';

class WatchlistItem {
  final String symbol;
  final String? nickname;
  final DateTime addedAt;

  WatchlistItem({
    required this.symbol,
    this.nickname,
    required this.addedAt,
  });

  Map<String, dynamic> toJson() => {
        'symbol': symbol,
        'nickname': nickname,
        'addedAt': addedAt.toIso8601String(),
      };

  static WatchlistItem fromJson(Map<String, dynamic> json) {
    return WatchlistItem(
      symbol: (json['symbol'] ?? '').toString(),
      nickname: json['nickname']?.toString(),
      addedAt: DateTime.tryParse(json['addedAt']?.toString() ?? '') ??
          DateTime.now(),
    );
  }
}

class WatchlistStore {
  final ValueNotifier<List<WatchlistItem>> items =
      ValueNotifier<List<WatchlistItem>>(<WatchlistItem>[]);

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_kWatchlistKey);
    if (raw == null) return;
    try {
      final decoded = jsonDecode(raw) as List<dynamic>;
      items.value = decoded
          .whereType<Map<String, dynamic>>()
          .map(WatchlistItem.fromJson)
          .toList();
    } catch (_) {
      items.value = <WatchlistItem>[];
    }
  }

  Future<void> _persist() async {
    final prefs = await SharedPreferences.getInstance();
    final encoded =
        jsonEncode(items.value.map((e) => e.toJson()).toList());
    await prefs.setString(_kWatchlistKey, encoded);
  }

  bool contains(String symbol) {
    final sym = symbol.toUpperCase();
    return items.value.any((w) => w.symbol.toUpperCase() == sym);
  }

  Future<void> toggle(String symbol) async {
    final sym = symbol.toUpperCase();
    final list = List<WatchlistItem>.from(items.value);
    final idx = list.indexWhere((w) => w.symbol.toUpperCase() == sym);
    if (idx >= 0) {
      list.removeAt(idx);
    } else {
      list.add(WatchlistItem(
        symbol: sym,
        addedAt: DateTime.now(),
      ));
    }
    items.value = list;
    await _persist();
  }
}
