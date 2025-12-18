/// Technic App Colors - Premium Design System v3.0
/// 
/// Inspired by world-class fintech apps (Robinhood, Webull, Bloomberg)
/// Billion-dollar app quality with Technic branding
library;

import 'package:flutter/material.dart';

/// Technic Color System v3.0
/// Premium palette for institutional-grade finance app
class AppColors {
  AppColors._(); // Private constructor to prevent instantiation

  // ============================================================================
  // DARK THEME (Primary) - Enhanced
  // ============================================================================
  
  /// Background Hierarchy
  static const Color darkBackground = Color(0xFF0A0E27);      // Deep navy, almost black
  static const Color darkCard = Color(0xFF1A1F3A);            // Enhanced card surface
  static const Color darkCardElevated = Color(0xFF252B4A);    // More prominent lift
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
  // BRAND COLORS - Technic Identity
  // ============================================================================
  
  /// Primary Brand Color - Technic Blue
  static const Color technicBlue = Color(0xFF4A9EFF);         // Signature Technic blue
  static const Color technicBlueDark = Color(0xFF2E7FD9);     // Darker variant
  static const Color technicBlueLight = Color(0xFF6FB3FF);    // Lighter variant
  
  // ============================================================================
  // ACCENT COLORS - Premium Vibrant Palette
  // ============================================================================
  
  /// Primary Actions & Links (Using Technic Blue)
  static const Color primaryBlue = technicBlue;               // Technic brand color
  
  /// Success & Gains (Neon Green - Robinhood style)
  static const Color successGreen = Color(0xFF00FF88);        // Vibrant neon green
  static const Color successGreenDark = Color(0xFF00CC6A);    // Darker for gradients
  static const Color successGreenMuted = Color(0xFF10B981);   // Muted for subtlety
  
  /// Danger & Losses (Bright Red - High visibility)
  static const Color dangerRed = Color(0xFFFF3B5C);           // Bright, attention-grabbing
  static const Color dangerRedDark = Color(0xFFE6284A);       // Darker for gradients
  static const Color dangerRedMuted = Color(0xFFEF4444);      // Muted for subtlety
  
  /// Warning & Caution
  static const Color warningOrange = Color(0xFFFFB84D);       // Warm, friendly warning
  static const Color warningAmber = Color(0xFFF59E0B);        // Amber-500, caution
  
  /// Info & Neutral
  static const Color infoTeal = Color(0xFF14B8A6);            // Teal-500, neutral info
  static const Color infoPurple = Color(0xFF8B5CF6);          // Purple for premium features
  
  // ============================================================================
  // CHART COLORS - Enhanced for Technical Analysis
  // ============================================================================
  
  /// Bullish Candle (Vibrant Green)
  static const Color chartBullish = successGreen;             // Neon green for gains
  static const Color chartBullishMuted = successGreenMuted;   // Muted alternative
  
  /// Bearish Candle (Bright Red)
  static const Color chartBearish = dangerRed;                // Bright red for losses
  static const Color chartBearishMuted = dangerRedMuted;      // Muted alternative
  
  /// Line Chart (Technic Blue)
  static const Color chartLine = technicBlue;                 // Brand color
  static const Color chartLineSecondary = Color(0xFF8B5CF6);  // Purple for secondary lines
  
  /// Volume Bars
  static const Color chartVolume = Color(0xFF4B5563);         // Gray-600, subtle
  static const Color chartVolumeHigh = Color(0xFF6B7280);     // Slightly brighter for high volume
  
  /// Technical Indicators
  static const Color chartRSI = Color(0xFFFFB84D);            // Orange for RSI
  static const Color chartMACD = Color(0xFF14B8A6);           // Teal for MACD
  static const Color chartBollingerBands = Color(0xFF8B5CF6); // Purple for Bollinger
  
  // ============================================================================
  // GRADIENTS - Premium Visual Effects
  // ============================================================================
  
  /// Primary Gradient (Technic Blue)
  static const LinearGradient primaryGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [technicBlue, technicBlueDark],
  );
  
  /// Success Gradient (Neon Green)
  static const LinearGradient successGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [successGreen, successGreenDark],
  );
  
  /// Danger Gradient (Bright Red)
  static const LinearGradient dangerGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [dangerRed, dangerRedDark],
  );
  
  /// Card Gradient (Subtle depth)
  static const LinearGradient cardGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFF1A1F3A), Color(0xFF252B4A)],
  );
  
  /// Shimmer Gradient (Loading states)
  static const LinearGradient shimmerGradient = LinearGradient(
    begin: Alignment(-1.0, 0.0),
    end: Alignment(1.0, 0.0),
    colors: [
      Color(0xFF1A1F3A),
      Color(0xFF252B4A),
      Color(0xFF1A1F3A),
    ],
    stops: [0.0, 0.5, 1.0],
  );
  
  // ============================================================================
  // SEMANTIC COLORS - Context-specific
  // ============================================================================
  
  /// Buy/Long Actions
  static const Color buyColor = successGreen;
  
  /// Sell/Short Actions
  static const Color sellColor = dangerRed;
  
  /// Hold/Neutral
  static const Color holdColor = warningOrange;
  
  /// Premium Features
  static const Color premiumColor = infoPurple;
  static const LinearGradient premiumGradient = LinearGradient(
    colors: [Color(0xFF8B5CF6), Color(0xFF6366F1)],
  );
  
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
}
