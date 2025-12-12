/// Technic App Colors - Institutional Design System
/// 
/// Based on best-in-class finance apps (Robinhood, Webull, etc.)
/// Ultra-minimal, professional, trustworthy color palette.
library;

import 'package:flutter/material.dart';

/// Technic Color System v2.0
/// Institutional-grade palette for professional finance app
class AppColors {
  AppColors._(); // Private constructor to prevent instantiation

  // ============================================================================
  // DARK THEME (Primary)
  // ============================================================================
  
  /// Background Hierarchy
  static const Color darkBackground = Color(0xFF0A0E27);      // Deep navy, almost black
  static const Color darkCard = Color(0xFF141B2D);            // Slate-900 equivalent
  static const Color darkCardElevated = Color(0xFF1A2332);    // Subtle lift
  static const Color darkBorder = Color(0xFF2D3748);          // Slate-700, very subtle
  
  /// Text Colors (Dark Theme)
  static const Color darkTextPrimary = Color(0xFFF7FAFC);     // Slate-50, high contrast
  static const Color darkTextSecondary = Color(0xFFA0AEC0);   // Slate-400, readable
  static const Color darkTextTertiary = Color(0xFF718096);    // Slate-500, de-emphasized
  
  // ============================================================================
  // LIGHT THEME (Secondary)
  // ============================================================================
  
  /// Background Hierarchy
  static const Color lightBackground = Color(0xFFF8FAFC);     // Slate-50
  static const Color lightCard = Color(0xFFFFFFFF);           // Pure white
  static const Color lightCardElevated = Color(0xFFF1F5F9);   // Slate-100
  static const Color lightBorder = Color(0xFFE2E8F0);         // Slate-200
  
  /// Text Colors (Light Theme)
  static const Color lightTextPrimary = Color(0xFF1E293B);    // Slate-800
  static const Color lightTextSecondary = Color(0xFF475569);  // Slate-600
  static const Color lightTextTertiary = Color(0xFF94A3B8);   // Slate-400
  
  // ============================================================================
  // ACCENT COLORS (Same for both themes)
  // ============================================================================
  
  /// Primary Actions & Links
  static const Color primaryBlue = Color(0xFF3B82F6);         // Blue-500, trust/action
  
  /// Success & Gains
  static const Color successGreen = Color(0xFF10B981);        // Emerald-500, NOT neon
  
  /// Danger & Losses
  static const Color dangerRed = Color(0xFFEF4444);           // Red-500, losses/stops
  
  /// Warning & Caution
  static const Color warningAmber = Color(0xFFF59E0B);        // Amber-500, caution
  
  /// Info & Neutral
  static const Color infoTeal = Color(0xFF14B8A6);            // Teal-500, neutral info
  
  // ============================================================================
  // CHART COLORS
  // ============================================================================
  
  /// Bullish Candle
  static const Color chartBullish = Color(0xFF10B981);        // Muted green
  
  /// Bearish Candle
  static const Color chartBearish = Color(0xFFEF4444);        // Muted red
  
  /// Line Chart
  static const Color chartLine = Color(0xFF3B82F6);           // Primary blue
  
  /// Volume Bars
  static const Color chartVolume = Color(0xFF4B5563);         // Gray-600, subtle
  
  // ============================================================================
  // LEGACY COMPATIBILITY (Deprecated - will be removed)
  // ============================================================================
  
  @Deprecated('Use primaryBlue instead')
  static const Color skyBlue = primaryBlue;
  
  @Deprecated('Use darkBackground instead')
  static const Color darkDeep = darkBackground;
  
  @Deprecated('Use darkCard instead')
  static const Color darkBg = darkCard;
  
  @Deprecated('Use darkBorder instead')
  static const Color darkAccent = darkBorder;
  
  // ============================================================================
  // HELPER METHODS
  // ============================================================================
  
  /// Apply opacity to a color
  static Color withOpacity(Color color, double opacity) {
    return color.withOpacity(opacity.clamp(0.0, 1.0));
  }
  
  /// Get appropriate text color based on background
  static Color getTextColor(Color background, {bool isDark = true}) {
    if (isDark) {
      return darkTextPrimary;
    } else {
      return lightTextPrimary;
    }
  }
  
  /// Get appropriate card color based on theme
  static Color getCardColor({required bool isDark, bool elevated = false}) {
    if (isDark) {
      return elevated ? darkCardElevated : darkCard;
    } else {
      return elevated ? lightCardElevated : lightCard;
    }
  }
  
  /// Get appropriate border color based on theme
  static Color getBorderColor({required bool isDark}) {
    return isDark ? darkBorder : lightBorder;
  }
}


