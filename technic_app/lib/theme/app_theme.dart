/// technic App Theme
/// 
/// Complete theme configuration for light and dark modes
/// Following Apple HIG and Material Design 3 principles

import 'package:flutter/material.dart';
import 'app_colors.dart';

class AppTheme {
  AppTheme._(); // Private constructor

  // ============================================================================
  // LIGHT THEME (Primary)
  // ============================================================================
  
  static ThemeData lightTheme() {
    return ThemeData(
      brightness: Brightness.light,
      useMaterial3: true,
      
      // Color Scheme
      colorScheme: ColorScheme.light(
        primary: AppColors.skyBlue,
        onPrimary: AppColors.imperialBlue,
        secondary: AppColors.imperialBlue,
        onSecondary: AppColors.white,
        tertiary: AppColors.pineGrove,
        onTertiary: AppColors.white,
        error: AppColors.error,
        onError: AppColors.white,
        surface: AppColors.white,
        onSurface: AppColors.textPrimary,
        surfaceContainerHighest: AppColors.offWhite,
      ),
      
      // Scaffold
      scaffoldBackgroundColor: AppColors.white,
      
      // App Bar
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.textPrimary,
        centerTitle: false,
        titleTextStyle: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w800,
          color: AppColors.textPrimary,
          letterSpacing: 0.2,
        ),
      ),
      
      // Card
      cardTheme: CardThemeData(
        color: AppColors.offWhite,
        elevation: 0,
        margin: const EdgeInsets.symmetric(vertical: 6),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(
            color: AppColors.borderLight,
            width: 1,
          ),
        ),
        shadowColor: AppColors.withOpacity(AppColors.shadowLight, 0.08),
      ),
      
      // Text Theme
      textTheme: const TextTheme(
        // Display
        displayLarge: TextStyle(fontSize: 57, fontWeight: FontWeight.bold, color: AppColors.textPrimary),
        displayMedium: TextStyle(fontSize: 45, fontWeight: FontWeight.bold, color: AppColors.textPrimary),
        displaySmall: TextStyle(fontSize: 36, fontWeight: FontWeight.bold, color: AppColors.textPrimary),
        
        // Headline
        headlineLarge: TextStyle(fontSize: 32, fontWeight: FontWeight.bold, color: AppColors.textPrimary),
        headlineMedium: TextStyle(fontSize: 28, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
        headlineSmall: TextStyle(fontSize: 24, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
        
        // Title
        titleLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
        titleMedium: TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: AppColors.textPrimary),
        titleSmall: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textPrimary),
        
        // Body
        bodyLarge: TextStyle(fontSize: 16, fontWeight: FontWeight.w400, color: AppColors.textPrimary),
        bodyMedium: TextStyle(fontSize: 14, fontWeight: FontWeight.w400, color: AppColors.textPrimary),
        bodySmall: TextStyle(fontSize: 12, fontWeight: FontWeight.w400, color: AppColors.textSecondary),
        
        // Label
        labelLarge: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textPrimary),
        labelMedium: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: AppColors.textSecondary),
        labelSmall: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: AppColors.textTertiary),
      ).apply(
        fontFamily: 'Inter',
      ),
      
      // Input Decoration
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.withOpacity(AppColors.textPrimary, 0.04),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(
            color: AppColors.borderLight,
            width: 1,
          ),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(
            color: AppColors.borderLight,
            width: 1,
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(
            color: AppColors.skyBlue,
            width: 2,
          ),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(
            color: AppColors.error,
            width: 1,
          ),
        ),
        hintStyle: const TextStyle(
          color: AppColors.textTertiary,
        ),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 14,
          vertical: 12,
        ),
      ),
      
      // Elevated Button
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.skyBlue,
          foregroundColor: AppColors.imperialBlue,
          elevation: 0,
          shadowColor: Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      
      // Outlined Button
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.imperialBlue,
          side: const BorderSide(
            color: AppColors.skyBlue,
            width: 1.5,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      
      // Text Button
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: AppColors.skyBlue,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      
      // Chip
      chipTheme: ChipThemeData(
        backgroundColor: AppColors.withOpacity(AppColors.textPrimary, 0.04),
        selectedColor: AppColors.withOpacity(AppColors.skyBlue, 0.15),
        labelStyle: const TextStyle(color: AppColors.textPrimary),
        secondaryLabelStyle: const TextStyle(color: AppColors.textSecondary),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
        ),
        side: const BorderSide(
          color: AppColors.borderLight,
          width: 1,
        ),
      ),
      
      // Navigation Bar
      navigationBarTheme: NavigationBarThemeData(
        indicatorColor: AppColors.withOpacity(AppColors.skyBlue, 0.15),
        backgroundColor: AppColors.white,
        elevation: 0,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        height: 70,
        iconTheme: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const IconThemeData(color: AppColors.skyBlue);
          }
          return const IconThemeData(color: AppColors.textTertiary);
        }),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const TextStyle(
              color: AppColors.skyBlue,
              fontWeight: FontWeight.w700,
              fontSize: 12,
            );
          }
          return const TextStyle(
            color: AppColors.textTertiary,
            fontWeight: FontWeight.w500,
            fontSize: 12,
          );
        }),
      ),
      
      // Dialog
      dialogTheme: DialogThemeData(
        backgroundColor: AppColors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        titleTextStyle: const TextStyle(
          color: AppColors.textPrimary,
          fontWeight: FontWeight.w800,
          fontSize: 18,
          fontFamily: 'Inter',
        ),
        contentTextStyle: const TextStyle(
          color: AppColors.textPrimary,
          fontSize: 16,
          fontFamily: 'Inter',
        ),
      ),
      
      // Snackbar
      snackBarTheme: SnackBarThemeData(
        backgroundColor: AppColors.withOpacity(AppColors.skyBlue, 0.12),
        contentTextStyle: const TextStyle(color: AppColors.textPrimary),
        actionTextColor: AppColors.skyBlue,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      
      // Divider
      dividerTheme: const DividerThemeData(
        color: AppColors.dividerLight,
        thickness: 1,
        space: 1,
      ),
    );
  }
  
  // ============================================================================
  // DARK THEME (Secondary)
  // ============================================================================
  
  static ThemeData darkTheme() {
    return ThemeData(
      brightness: Brightness.dark,
      useMaterial3: true,
      
      // Color Scheme
      colorScheme: ColorScheme.dark(
        primary: AppColors.skyBlue,
        onPrimary: AppColors.imperialBlue,
        secondary: AppColors.skyBlue,
        onSecondary: AppColors.imperialBlue,
        tertiary: AppColors.pineGrove,
        onTertiary: AppColors.white,
        error: AppColors.error,
        onError: AppColors.white,
        surface: AppColors.darkBg,
        onSurface: AppColors.textPrimaryDark,
        surfaceContainerHighest: AppColors.darkCard,
      ),
      
      // Scaffold
      scaffoldBackgroundColor: AppColors.darkBg,
      
      // App Bar
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.textPrimaryDark,
        centerTitle: false,
        titleTextStyle: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w800,
          color: AppColors.textPrimaryDark,
          letterSpacing: 0.2,
        ),
      ),
      
      // Card
      cardTheme: CardThemeData(
        color: AppColors.darkCard,
        elevation: 0,
        margin: const EdgeInsets.symmetric(vertical: 6),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(
            color: AppColors.borderDark,
            width: 1,
          ),
        ),
        shadowColor: AppColors.withOpacity(AppColors.shadowDark, 0.4),
      ),
      
      // Text Theme
      textTheme: const TextTheme(
        // Display
        displayLarge: TextStyle(fontSize: 57, fontWeight: FontWeight.bold, color: AppColors.textPrimaryDark),
        displayMedium: TextStyle(fontSize: 45, fontWeight: FontWeight.bold, color: AppColors.textPrimaryDark),
        displaySmall: TextStyle(fontSize: 36, fontWeight: FontWeight.bold, color: AppColors.textPrimaryDark),
        
        // Headline
        headlineLarge: TextStyle(fontSize: 32, fontWeight: FontWeight.bold, color: AppColors.textPrimaryDark),
        headlineMedium: TextStyle(fontSize: 28, fontWeight: FontWeight.w600, color: AppColors.textPrimaryDark),
        headlineSmall: TextStyle(fontSize: 24, fontWeight: FontWeight.w600, color: AppColors.textPrimaryDark),
        
        // Title
        titleLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w600, color: AppColors.textPrimaryDark),
        titleMedium: TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: AppColors.textPrimaryDark),
        titleSmall: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textPrimaryDark),
        
        // Body
        bodyLarge: TextStyle(fontSize: 16, fontWeight: FontWeight.w400, color: AppColors.textPrimaryDark),
        bodyMedium: TextStyle(fontSize: 14, fontWeight: FontWeight.w400, color: AppColors.textPrimaryDark),
        bodySmall: TextStyle(fontSize: 12, fontWeight: FontWeight.w400, color: AppColors.textSecondaryDark),
        
        // Label
        labelLarge: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textPrimaryDark),
        labelMedium: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: AppColors.textSecondaryDark),
        labelSmall: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: AppColors.textTertiaryDark),
      ).apply(
        fontFamily: 'Inter',
      ),
      
      // Input Decoration
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.withOpacity(AppColors.white, 0.03),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(
            color: AppColors.borderDark,
            width: 1,
          ),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(
            color: AppColors.borderDark,
            width: 1,
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(
            color: AppColors.skyBlue,
            width: 2,
          ),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(
            color: AppColors.error,
            width: 1,
          ),
        ),
        hintStyle: const TextStyle(
          color: AppColors.textTertiaryDark,
        ),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 14,
          vertical: 12,
        ),
      ),
      
      // Elevated Button
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.skyBlue,
          foregroundColor: AppColors.imperialBlue,
          elevation: 0,
          shadowColor: Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      
      // Outlined Button
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.skyBlue,
          side: const BorderSide(
            color: AppColors.skyBlue,
            width: 1.5,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      
      // Text Button
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: AppColors.skyBlue,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      
      // Chip
      chipTheme: ChipThemeData(
        backgroundColor: AppColors.withOpacity(AppColors.textPrimary, 0.04),
        selectedColor: AppColors.withOpacity(AppColors.skyBlue, 0.15),
        labelStyle: const TextStyle(color: AppColors.textPrimaryDark),
        secondaryLabelStyle: const TextStyle(color: AppColors.textSecondaryDark),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
        ),
        side: const BorderSide(
          color: AppColors.borderDark,
          width: 1,
        ),
      ),
      
      // Navigation Bar
      navigationBarTheme: NavigationBarThemeData(
        indicatorColor: AppColors.withOpacity(AppColors.skyBlue, 0.18),
        backgroundColor: AppColors.darkCard,
        elevation: 0,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        height: 70,
        iconTheme: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const IconThemeData(color: AppColors.skyBlue);
          }
          return const IconThemeData(color: AppColors.textTertiaryDark);
        }),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const TextStyle(
              color: AppColors.skyBlue,
              fontWeight: FontWeight.w700,
              fontSize: 12,
            );
          }
          return const TextStyle(
            color: AppColors.textTertiaryDark,
            fontWeight: FontWeight.w500,
            fontSize: 12,
          );
        }),
      ),
      
      // Dialog
      dialogTheme: DialogThemeData(
        backgroundColor: AppColors.darkCard,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        titleTextStyle: const TextStyle(
          color: AppColors.textPrimaryDark,
          fontWeight: FontWeight.w800,
          fontSize: 18,
          fontFamily: 'Inter',
        ),
        contentTextStyle: const TextStyle(
          color: AppColors.textPrimaryDark,
          fontSize: 16,
          fontFamily: 'Inter',
        ),
      ),
      
      // Snackbar
      snackBarTheme: SnackBarThemeData(
        backgroundColor: AppColors.darkCard,
        contentTextStyle: const TextStyle(color: AppColors.textPrimaryDark),
        actionTextColor: AppColors.skyBlue,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      
      // Divider
      dividerTheme: const DividerThemeData(
        color: AppColors.dividerDark,
        thickness: 1,
        space: 1,
      ),
    );
  }
  
  // ============================================================================
  // HELPER METHODS
  // ============================================================================
  
  /// Get theme based on brightness
  static ThemeData getTheme(bool isDark) {
    return isDark ? darkTheme() : lightTheme();
  }
  
  /// Create a color with opacity (helper for gradients and overlays)
  static Color tone(Color base, double opacity) {
    return AppColors.withOpacity(base, opacity);
  }
}
