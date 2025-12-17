import 'package:flutter/material.dart';

class AppTheme {
  // Technic Brand Colors (from BRANDING.md)
  static const Color neonGreen = Color(0xFFB6FF3B);      // Primary: neon growth green
  static const Color aquaAccent = Color(0xFF5EEAD4);     // Secondary: aqua accent
  static const Color textLight = Color(0xFFE5E7EB);      // Text color
  static const Color surfaceDark = Color(0xFF02040D);    // Dark surface
  static const Color surfaceMedium = Color(0xFF0A1020);  // Medium surface
  static const Color positiveGreen = Color(0xFF9EF01A);  // Positive indicator
  static const Color negativeRed = Color(0xFFFF6B81);    // Negative indicator
  
  // Dark Theme (Primary - matches Technic aesthetic)
  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: surfaceDark,
      colorScheme: ColorScheme.dark(
        primary: neonGreen,
        secondary: aquaAccent,
        surface: surfaceMedium,
        background: surfaceDark,
        error: negativeRed,
        onPrimary: surfaceDark,
        onSecondary: surfaceDark,
        onSurface: textLight,
        onBackground: textLight,
      ),
      appBarTheme: AppBarTheme(
        centerTitle: true,
        elevation: 0,
        backgroundColor: surfaceDark,
        foregroundColor: textLight,
      ),
      cardTheme: CardThemeData(
        elevation: 2,
        color: surfaceMedium,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: neonGreen,
          foregroundColor: surfaceDark,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        ),
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(color: textLight, fontWeight: FontWeight.bold),
        displayMedium: TextStyle(color: textLight, fontWeight: FontWeight.bold),
        displaySmall: TextStyle(color: textLight, fontWeight: FontWeight.bold),
        headlineLarge: TextStyle(color: textLight, fontWeight: FontWeight.bold),
        headlineMedium: TextStyle(color: textLight, fontWeight: FontWeight.bold),
        headlineSmall: TextStyle(color: textLight, fontWeight: FontWeight.bold),
        titleLarge: TextStyle(color: textLight, fontWeight: FontWeight.w600),
        titleMedium: TextStyle(color: textLight, fontWeight: FontWeight.w600),
        titleSmall: TextStyle(color: textLight, fontWeight: FontWeight.w600),
        bodyLarge: TextStyle(color: textLight),
        bodyMedium: TextStyle(color: textLight),
        bodySmall: TextStyle(color: textLight),
      ),
      iconTheme: const IconThemeData(color: textLight),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: surfaceMedium,
        selectedItemColor: neonGreen,
        unselectedItemColor: textLight.withOpacity(0.6),
      ),
    );
  }
  
  // Light Theme (Optional - for users who prefer light mode)
  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: Colors.white,
      colorScheme: ColorScheme.light(
        primary: neonGreen,
        secondary: aquaAccent,
        surface: Colors.grey[100]!,
        background: Colors.white,
        error: negativeRed,
        onPrimary: surfaceDark,
        onSecondary: surfaceDark,
        onSurface: surfaceDark,
        onBackground: surfaceDark,
      ),
      appBarTheme: AppBarTheme(
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.white,
        foregroundColor: surfaceDark,
      ),
      cardTheme: CardThemeData(
        elevation: 2,
        color: Colors.grey[100],
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: neonGreen,
          foregroundColor: surfaceDark,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        ),
      ),
      textTheme: TextTheme(
        displayLarge: TextStyle(color: surfaceDark, fontWeight: FontWeight.bold),
        displayMedium: TextStyle(color: surfaceDark, fontWeight: FontWeight.bold),
        displaySmall: TextStyle(color: surfaceDark, fontWeight: FontWeight.bold),
        headlineLarge: TextStyle(color: surfaceDark, fontWeight: FontWeight.bold),
        headlineMedium: TextStyle(color: surfaceDark, fontWeight: FontWeight.bold),
        headlineSmall: TextStyle(color: surfaceDark, fontWeight: FontWeight.bold),
        titleLarge: TextStyle(color: surfaceDark, fontWeight: FontWeight.w600),
        titleMedium: TextStyle(color: surfaceDark, fontWeight: FontWeight.w600),
        titleSmall: TextStyle(color: surfaceDark, fontWeight: FontWeight.w600),
        bodyLarge: TextStyle(color: surfaceDark),
        bodyMedium: TextStyle(color: surfaceDark),
        bodySmall: TextStyle(color: surfaceDark),
      ),
      iconTheme: IconThemeData(color: surfaceDark),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: Colors.white,
        selectedItemColor: neonGreen,
        unselectedItemColor: surfaceDark.withOpacity(0.6),
      ),
    );
  }
}
