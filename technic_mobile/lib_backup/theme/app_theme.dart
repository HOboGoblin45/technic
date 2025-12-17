import 'package:flutter/material.dart';
import 'spacing.dart';
import 'border_radii.dart';
import 'shadows.dart';
import 'animations.dart';

/// Technic App Theme - Mac-Inspired Institutional Design System
/// Integrates spacing, shadows, border radii, and animations for a polished Mac aesthetic
class AppTheme {
  // CORRECT Technic Colors (Institutional Finance App Palette)
  // Dark Theme - Primary
  static const Color darkBackground = Color(0xFF0A0E27);      // Deep navy, almost black
  static const Color darkCard = Color(0xFF141B2D);            // Slate-900 equivalent
  static const Color darkCardElevated = Color(0xFF1A2332);    // Subtle lift
  static const Color darkBorder = Color(0xFF2D3748);          // Slate-700, very subtle
  
  // Text Colors (Dark Theme)
  static const Color darkTextPrimary = Color(0xFFF7FAFC);     // Slate-50, high contrast
  static const Color darkTextSecondary = Color(0xFFA0AEC0);   // Slate-400, readable
  static const Color darkTextTertiary = Color(0xFF718096);    // Slate-500, de-emphasized
  
  // Accent Colors
  static const Color primaryBlue = Color(0xFF3B82F6);         // Blue-500, trust/action
  static const Color successGreen = Color(0xFF10B981);        // Emerald-500, NOT neon
  static const Color warningOrange = Color(0xFFFF9800);       // Orange-500
  static const Color dangerRed = Color(0xFFEF4444);           // Red-500, losses/stops
  static const Color warningAmber = Color(0xFFF59E0B);        // Amber-500, caution
  static const Color infoTeal = Color(0xFF14B8A6);            // Teal-500, neutral info
  
  // Dark Theme (Primary - Institutional Finance Aesthetic)
  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: darkBackground,
      colorScheme: ColorScheme.dark(
        primary: primaryBlue,
        secondary: successGreen,
        surface: darkCard,
        background: darkBackground,
        error: dangerRed,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: darkTextPrimary,
        onBackground: darkTextPrimary,
      ),
      appBarTheme: AppBarTheme(
        centerTitle: true,
        elevation: 0,
        backgroundColor: darkBackground,
        foregroundColor: darkTextPrimary,
      ),
      cardTheme: CardThemeData(
        elevation: 0, // Use box shadows instead
        color: darkCard,
        margin: Spacing.edgeInsetsMD,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadii.cardBorderRadius,
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryBlue,
          foregroundColor: Colors.white,
          padding: Spacing.horizontalPaddingLG.add(Spacing.verticalPaddingMD),
          elevation: 0, // Use box shadows instead
          shadowColor: Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadii.buttonBorderRadius,
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: darkCardElevated,
        contentPadding: Spacing.edgeInsetsMD,
        border: OutlineInputBorder(
          borderRadius: BorderRadii.inputBorderRadius,
          borderSide: BorderSide(color: darkBorder),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadii.inputBorderRadius,
          borderSide: BorderSide(color: darkBorder),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadii.inputBorderRadius,
          borderSide: BorderSide(color: primaryBlue, width: 2),
        ),
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.bold),
        displayMedium: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.bold),
        displaySmall: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.bold),
        headlineLarge: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.bold),
        headlineMedium: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.bold),
        headlineSmall: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.bold),
        titleLarge: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.w600),
        titleMedium: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.w600),
        titleSmall: TextStyle(color: darkTextPrimary, fontWeight: FontWeight.w600),
        bodyLarge: TextStyle(color: darkTextPrimary),
        bodyMedium: TextStyle(color: darkTextSecondary),
        bodySmall: TextStyle(color: darkTextTertiary),
      ),
      iconTheme: const IconThemeData(color: darkTextPrimary),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: darkCard,
        selectedItemColor: primaryBlue,
        unselectedItemColor: darkTextSecondary,
      ),
    );
  }
  
  // Light Theme (Optional - for users who prefer light mode)
  static ThemeData get lightTheme {
    const Color lightBackground = Color(0xFFF8FAFC);     // Slate-50
    const Color lightCard = Color(0xFFFFFFFF);           // Pure white
    const Color lightTextPrimary = Color(0xFF1E293B);    // Slate-800
    const Color lightTextSecondary = Color(0xFF475569);  // Slate-600
    
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: lightBackground,
      colorScheme: ColorScheme.light(
        primary: primaryBlue,
        secondary: successGreen,
        surface: lightCard,
        background: lightBackground,
        error: dangerRed,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: lightTextPrimary,
        onBackground: lightTextPrimary,
      ),
      appBarTheme: const AppBarTheme(
        centerTitle: true,
        elevation: 0,
        backgroundColor: lightBackground,
        foregroundColor: lightTextPrimary,
      ),
      cardTheme: CardThemeData(
        elevation: 0, // Use box shadows instead
        color: lightCard,
        margin: Spacing.edgeInsetsMD,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadii.cardBorderRadius,
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryBlue,
          foregroundColor: Colors.white,
          padding: Spacing.horizontalPaddingLG.add(Spacing.verticalPaddingMD),
          elevation: 0, // Use box shadows instead
          shadowColor: Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadii.buttonBorderRadius,
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: const Color(0xFFF1F5F9), // Slate-100
        contentPadding: Spacing.edgeInsetsMD,
        border: OutlineInputBorder(
          borderRadius: BorderRadii.inputBorderRadius,
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadii.inputBorderRadius,
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadii.inputBorderRadius,
          borderSide: BorderSide(color: primaryBlue, width: 2),
        ),
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.bold),
        displayMedium: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.bold),
        displaySmall: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.bold),
        headlineLarge: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.bold),
        headlineMedium: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.bold),
        headlineSmall: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.bold),
        titleLarge: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.w600),
        titleMedium: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.w600),
        titleSmall: TextStyle(color: lightTextPrimary, fontWeight: FontWeight.w600),
        bodyLarge: TextStyle(color: lightTextPrimary),
        bodyMedium: TextStyle(color: lightTextSecondary),
        bodySmall: TextStyle(color: lightTextSecondary),
      ),
      iconTheme: const IconThemeData(color: lightTextPrimary),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: lightCard,
        selectedItemColor: primaryBlue,
        unselectedItemColor: lightTextSecondary,
      ),
    );
  }
}
