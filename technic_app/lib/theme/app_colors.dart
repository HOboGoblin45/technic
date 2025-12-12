/// technic Brand Colors
/// 
/// Updated color palette emphasizing white-dominant, Apple-inspired sterile cleanliness
/// with strategic use of sky-blue, imperial blue, and pine grove accents.
library;

import 'package:flutter/material.dart';

class AppColors {
  AppColors._(); // Private constructor to prevent instantiation

  // ============================================================================
  // PRIMARY COLORS (White-Dominant)
  // ============================================================================
  
  /// Pure white - Main background in light mode
  static const white = Color(0xFFFFFFFF);
  
  /// Off-white - Cards and elevated surfaces in light mode
  static const offWhite = Color(0xFFFAFAFA);
  
  /// Light gray - Secondary elevated surfaces
  static const lightGray = Color(0xFFF5F5F5);
  
  // ============================================================================
  // BRAND COLORS (Updated)
  // ============================================================================
  
  /// Soft Pastel Sky-Blue - Primary brand color
  /// HEX: #B0CAFF
  /// Usage: Primary actions, highlights, logo, interactive elements
  static const skyBlue = Color(0xFFB0CAFF);
  
  /// Sky-Blue variants
  static const skyBlueLight = Color(0xFFD4E3FF);  // Hover states, backgrounds
  static const skyBlueDark = Color(0xFF8BA8E6);   // Pressed states
  
  /// Pantone Imperial Blue - Accent/Depth
  /// HEX: #001D51
  /// Usage: Text, borders, shadows, depth elements
  static const imperialBlue = Color(0xFF001D51);
  
  /// Imperial Blue variants
  static const imperialBlueLight = Color(0xFF1A3A6B);  // Secondary text
  
  /// Pantone Pine Grove - Accent/Organic
  /// HEX: #213631
  /// Usage: Success states, positive indicators, organic elements
  static const pineGrove = Color(0xFF213631);
  
  /// Pine Grove variants
  static const pineGroveLight = Color(0xFF3A5A4F);  // Hover states
  static const pineGroveDark = Color(0xFF1A2A26);   // Darker variant
  
  // ============================================================================
  // DARK MODE BACKGROUNDS
  // ============================================================================
  
  /// Near black - Base background in dark mode
  static const darkBg = Color(0xFF0A0A0A);
  
  /// Dark gray - Cards in dark mode
  static const darkCard = Color(0xFF1A1A1A);
  
  /// Medium gray - Elevated surfaces in dark mode
  static const darkElevated = Color(0xFF2A2A2A);
  
  /// Deep dark - Legacy support (can be phased out)
  static const darkDeep = Color(0xFF0A1214);
  
  // ============================================================================
  // SEMANTIC COLORS
  // ============================================================================
  
  /// Success/Positive - Uses Pine Grove
  static const success = pineGrove;
  static const successLight = pineGroveLight;
  
  /// Warning/Caution - Amber
  static const warning = Color(0xFFFFB84D);
  static const warningLight = Color(0xFFFFD699);
  
  /// Error/Negative - Coral Red
  static const error = Color(0xFFFF6B6B);
  static const errorLight = Color(0xFFFF9999);
  
  /// Info/Neutral - Sky Blue
  static const info = skyBlue;
  static const infoLight = skyBlueLight;
  
  // ============================================================================
  // TEXT COLORS
  // ============================================================================
  
  /// Light mode text
  static const textPrimary = imperialBlue;
  static const textSecondary = imperialBlueLight;
  static const textTertiary = Color(0xFF6B7280);
  static const textDisabled = Color(0xFF9CA3AF);
  
  /// Dark mode text
  static const textPrimaryDark = white;
  static const textSecondaryDark = skyBlue;
  static const textTertiaryDark = Color(0xFF9CA3AF);
  static const textDisabledDark = Color(0xFF6B7280);
  
  // ============================================================================
  // UI ELEMENTS
  // ============================================================================
  
  /// Dividers
  static const dividerLight = Color(0xFFE5E7EB);
  static const dividerDark = Color(0xFF2A2A2A);
  
  /// Borders
  static const borderLight = Color(0xFFD1D5DB);
  static const borderDark = Color(0xFF3A3A3A);
  
  /// Shadows (use with opacity)
  static const shadowLight = imperialBlue;  // Use with 8-24% opacity
  static const shadowDark = Color(0xFF000000);  // Use with 30-60% opacity
  
  // ============================================================================
  // TIER BADGE COLORS
  // ============================================================================
  
  /// CORE tier - Gradient start
  static const tierCoreStart = pineGrove;
  /// CORE tier - Gradient end
  static const tierCoreEnd = pineGroveLight;
  
  /// SATELLITE tier - Gradient start
  static const tierSatelliteStart = skyBlue;
  /// SATELLITE tier - Gradient end
  static const tierSatelliteEnd = skyBlueDark;
  
  /// REJECT tier
  static const tierReject = Color(0xFFF3F4F6);
  static const tierRejectText = Color(0xFF6B7280);
  
  // ============================================================================
  // LEGACY COLORS (For Migration - Will be removed)
  // ============================================================================
  
  /// Old brand primary - DEPRECATED, use skyBlue instead
  @Deprecated('Use skyBlue instead')
  static const brandPrimaryOld = Color(0xFF99BFFF);
  
  /// Old brand accent - Keep as imperialBlue
  static const brandAccent = imperialBlue;
  
  /// Old brand bg - Keep as pineGrove
  static const brandBg = pineGrove;
  
  /// Old brand deep - Use darkDeep instead
  static const brandDeep = darkDeep;
  
  // ============================================================================
  // HELPER METHODS
  // ============================================================================
  
  /// Create a color with specified opacity
  static Color withOpacity(Color color, double opacity) {
    return color.withAlpha((opacity * 255).clamp(0, 255).round());
  }
  
  /// Get appropriate text color for background
  static Color textOnBackground(Color background) {
    final luminance = background.computeLuminance();
    return luminance > 0.5 ? textPrimary : textPrimaryDark;
  }
  
  /// Get appropriate border color for theme
  static Color borderColor(bool isDark) {
    return isDark ? borderDark : borderLight;
  }
  
  /// Get appropriate divider color for theme
  static Color dividerColor(bool isDark) {
    return isDark ? dividerDark : dividerLight;
  }
}
