import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _kUserProfileKey = 'technic_user_profile_v1';

class UserProfile {
  final String riskProfile; // 'conservative' | 'balanced' | 'aggressive'
  final String optionsMode; // 'stock_only' | 'stock_plus_options'
  final String timeHorizon; // 'short_term' | 'swing' | 'position'
  final String themeMode; // 'light' | 'dark'

  const UserProfile({
    required this.riskProfile,
    required this.optionsMode,
    required this.timeHorizon,
    required this.themeMode,
  });

  UserProfile copyWith({
    String? riskProfile,
    String? optionsMode,
    String? timeHorizon,
    String? themeMode,
  }) {
    return UserProfile(
      riskProfile: riskProfile ?? this.riskProfile,
      optionsMode: optionsMode ?? this.optionsMode,
      timeHorizon: timeHorizon ?? this.timeHorizon,
      themeMode: themeMode ?? this.themeMode,
    );
  }

  Map<String, dynamic> toJson() => {
        'riskProfile': riskProfile,
        'optionsMode': optionsMode,
        'timeHorizon': timeHorizon,
        'themeMode': themeMode,
      };

  static UserProfile fromJson(Map<String, dynamic> json) {
    return UserProfile(
      riskProfile: (json['riskProfile'] ?? 'balanced') as String,
      optionsMode: (json['optionsMode'] ?? 'stock_plus_options') as String,
      timeHorizon: (json['timeHorizon'] ?? 'swing') as String,
      themeMode: (json['themeMode'] ?? 'light') as String,
    );
  }

  static const UserProfile defaults = UserProfile(
    riskProfile: 'balanced',
    optionsMode: 'stock_plus_options',
    timeHorizon: 'swing',
    themeMode: 'light',
  );
}

class UserProfileStore {
  final ValueNotifier<UserProfile> current =
      ValueNotifier<UserProfile>(UserProfile.defaults);

  final ValueNotifier<bool> ready = ValueNotifier<bool>(false);

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_kUserProfileKey);
    if (raw == null) {
      current.value = UserProfile.defaults;
      ready.value = true;
      return;
    }
    try {
      final decoded = jsonDecode(raw) as Map<String, dynamic>;
      current.value = UserProfile.fromJson(decoded);
    } catch (_) {
      current.value = UserProfile.defaults;
    }
    ready.value = true;
  }

  Future<void> save(UserProfile profile) async {
    current.value = profile;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kUserProfileKey, jsonEncode(profile.toJson()));
  }

  bool get isReady => ready.value;
}
