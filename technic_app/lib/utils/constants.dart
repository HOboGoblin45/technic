/// App Constants
/// 
/// Centralized constants used throughout the application.
library;

import 'package:flutter/material.dart';

// ============================================================================
// APP METADATA
// ============================================================================

const String appName = 'technic';
const String appVersion = '1.0.0';
const String appTagline = 'Quant Trading Companion';

// ============================================================================
// DEFAULT TICKERS
// ============================================================================

const List<String> defaultTickers = [
  'AAPL',
  'MSFT',
  'NVDA',
  'TSLA',
  'AMZN',
  'GOOGL',
  'META',
  'NFLX',
  'AMD',
  'INTC',
  'JPM',
  'GS',
  'XOM',
  'CVX',
  'BA',
  'CAT',
  'LMT',
  'KO',
  'PEP',
  'WMT',
];

// ============================================================================
// SECTORS
// ============================================================================

const List<String> sectors = [
  'Information Technology',
  'Health Care',
  'Industrials',
  'Financials',
  'Energy',
  'Consumer Discretionary',
  'Consumer Staples',
  'Communication Services',
  'Materials',
  'Real Estate',
  'Utilities',
];

// ============================================================================
// TRADE STYLES
// ============================================================================

const List<String> tradeStyles = [
  'Short-term swing',
  'Weekly',
  'Multi-day',
  'Momentum',
];

// ============================================================================
// RISK PROFILES
// ============================================================================

const List<String> riskProfiles = [
  'Conservative',
  'Balanced',
  'Aggressive',
];

// ============================================================================
// TIME HORIZONS
// ============================================================================

const List<String> timeHorizons = [
  'Short-term',
  'Swing',
  'Position',
];

// ============================================================================
// OPTIONS MODES
// ============================================================================

const String optionsModeStockOnly = 'stock_only';
const String optionsModeStockPlusOptions = 'stock_plus_options';

const List<String> optionsModes = [
  optionsModeStockOnly,
  optionsModeStockPlusOptions,
];

// ============================================================================
// COPILOT PROMPTS
// ============================================================================

const List<String> copilotPrompts = [
  "Summarize today's scan",
  'Explain risk on NVDA setup',
  'Compare TSLA vs AAPL momentum',
  'What moved semis this week?',
];

// ============================================================================
// QUICK ACTIONS
// ============================================================================

class QuickAction {
  final IconData icon;
  final String label;
  final String hint;

  const QuickAction(this.icon, this.label, this.hint);
}

const List<QuickAction> quickActions = [
  QuickAction(Icons.bolt, 'Fast scan', 'Run presets'),
  QuickAction(Icons.tune, 'Filters', 'Adjust criteria'),
  QuickAction(Icons.refresh, 'Refresh', 'Pull latest data'),
  QuickAction(Icons.save_alt, 'Save as preset', 'Reuse later'),
  QuickAction(Icons.dashboard_customize, 'Layout', 'Switch density'),
  QuickAction(Icons.analytics, 'Send to Copilot', 'Explain results'),
  QuickAction(Icons.shuffle, 'Randomize', 'Run a surprise scan'),
];

// ============================================================================
// SCAN DEFAULTS
// ============================================================================

const int defaultMaxSymbols = 50;
const double defaultLookbackDays = 90.0;
const double defaultMinTechRating = 0.0;
const bool defaultAllowShorts = true;
const bool defaultOnlyTradeable = false;
const String defaultTradeStyle = 'Short-term swing';
const String defaultRiskProfile = 'Balanced';
const String defaultTimeHorizon = 'Swing';

// ============================================================================
// UI CONSTANTS
// ============================================================================

const double defaultPadding = 16.0;
const double defaultBorderRadius = 16.0;
const double defaultCardElevation = 0.0;
const double defaultIconSize = 24.0;

const Duration defaultAnimationDuration = Duration(milliseconds: 250);
const Duration defaultSnackBarDuration = Duration(seconds: 2);

// ============================================================================
// API TIMEOUTS
// ============================================================================

const Duration apiTimeout = Duration(seconds: 30);
const Duration apiShortTimeout = Duration(seconds: 10);
const Duration apiLongTimeout = Duration(seconds: 60);

// ============================================================================
// CACHE DURATIONS
// ============================================================================

const Duration cacheShortDuration = Duration(minutes: 5);
const Duration cacheMediumDuration = Duration(minutes: 30);
const Duration cacheLongDuration = Duration(hours: 24);

// ============================================================================
// VALIDATION CONSTANTS
// ============================================================================

const int minSymbolsToScan = 1;
const int maxSymbolsToScan = 500;
const double minLookbackDays = 30.0;
const double maxLookbackDays = 365.0;
const double minTechRating = 0.0;
const double maxTechRating = 100.0;

// ============================================================================
// TIER THRESHOLDS
// ============================================================================

const double tierCoreThreshold = 80.0;
const double tierSatelliteThreshold = 65.0;

// ============================================================================
// STORAGE KEYS
// ============================================================================

class StorageKeys {
  static const String userId = 'user_id';
  static const String lastTab = 'last_tab';
  static const String filters = 'filters';
  static const String savedScreens = 'saved_screens';
  static const String scanCount = 'scan_count';
  static const String streakDays = 'streak_days';
  static const String lastScan = 'last_scan';
  static const String advancedMode = 'advanced_mode';
  static const String showOnboarding = 'show_onboarding';
  static const String lastScans = 'last_scans';
  static const String lastMovers = 'last_movers';
  static const String themeMode = 'theme_mode';
  static const String optionsMode = 'options_mode';
}

// ============================================================================
// THEME MODES
// ============================================================================

const String themeModeLight = 'light';
const String themeModeDark = 'dark';
const String themeModeSystem = 'system';

// ============================================================================
// ERROR MESSAGES
// ============================================================================

class ErrorMessages {
  static const String networkError = 'Network error. Please check your connection.';
  static const String serverError = 'Server error. Please try again later.';
  static const String unknownError = 'An unknown error occurred.';
  static const String noData = 'No data available.';
  static const String invalidInput = 'Invalid input. Please check your entries.';
  static const String copilotOffline = 'Copilot is temporarily offline.';
  static const String scanFailed = 'Scan failed. Please try again.';
}

// ============================================================================
// SUCCESS MESSAGES
// ============================================================================

class SuccessMessages {
  static const String scanComplete = 'Scan completed successfully!';
  static const String presetSaved = 'Preset saved successfully!';
  static const String settingsSaved = 'Settings saved!';
  static const String signedIn = 'Signed in successfully!';
  static const String signedOut = 'Signed out successfully!';
}

// ============================================================================
// NAVIGATION DESTINATIONS
// ============================================================================

class NavDestination {
  final String label;
  final IconData icon;
  final IconData selectedIcon;

  const NavDestination({
    required this.label,
    required this.icon,
    required this.selectedIcon,
  });
}

const List<NavDestination> navDestinations = [
  NavDestination(
    label: 'Scan',
    icon: Icons.assessment_outlined,
    selectedIcon: Icons.assessment,
  ),
  NavDestination(
    label: 'Ideas',
    icon: Icons.lightbulb_outline,
    selectedIcon: Icons.lightbulb,
  ),
  NavDestination(
    label: 'Copilot',
    icon: Icons.chat_bubble_outline,
    selectedIcon: Icons.chat_bubble,
  ),
  NavDestination(
    label: 'My Ideas',
    icon: Icons.star_border,
    selectedIcon: Icons.star,
  ),
  NavDestination(
    label: 'Settings',
    icon: Icons.settings_outlined,
    selectedIcon: Icons.settings,
  ),
];

// ============================================================================
// REGEX PATTERNS
// ============================================================================

class RegexPatterns {
  static final RegExp ticker = RegExp(r'^[A-Z]{1,5}$');
  static final RegExp email = RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$');
  static final RegExp number = RegExp(r'^\d+(\.\d+)?$');
  static final RegExp percentage = RegExp(r'^[+-]?\d+(\.\d+)?%?$');
}

// ============================================================================
// FEATURE FLAGS
// ============================================================================

class FeatureFlags {
  static const bool enableAdvancedMode = true;
  static const bool enableOptionsTrading = true;
  static const bool enableCopilot = true;
  static const bool enableScoreboard = true;
  static const bool enableSavedScreens = true;
  static const bool enableVoiceInput = false; // Coming soon
  static const bool enableNotifications = false; // Coming soon
  static const bool enableSocialSharing = false; // Coming soon
}
