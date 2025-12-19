/// Shadow constants for Mac-inspired design
/// Provides subtle depth and elevation throughout the app
library;

import 'package:flutter/material.dart';

class Shadows {
  Shadows._(); // Private constructor to prevent instantiation

  // Shadow colors (with opacity)
  static final Color shadowColor = Colors.black.withValues(alpha: 0.1);
  static final Color shadowColorLight = Colors.black.withValues(alpha: 0.05);
  static final Color shadowColorStrong = Colors.black.withValues(alpha: 0.15);
  
  // Subtle shadow (for cards, small elevations)
  static final List<BoxShadow> subtle = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.05),
      blurRadius: 10,
      offset: const Offset(0, 2),
      spreadRadius: 0,
    ),
  ];
  
  // Medium shadow (for modals, floating elements)
  static final List<BoxShadow> medium = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.1),
      blurRadius: 20,
      offset: const Offset(0, 4),
      spreadRadius: 0,
    ),
  ];
  
  // Strong shadow (for dialogs, important elements)
  static final List<BoxShadow> strong = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.15),
      blurRadius: 30,
      offset: const Offset(0, 8),
      spreadRadius: 0,
    ),
  ];
  
  // Layered shadow (for maximum depth)
  static final List<BoxShadow> layered = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.1),
      blurRadius: 10,
      offset: const Offset(0, 2),
      spreadRadius: 0,
    ),
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.05),
      blurRadius: 20,
      offset: const Offset(0, 4),
      spreadRadius: 0,
    ),
  ];
  
  // Semantic shadows (for specific use cases)
  static final List<BoxShadow> card = subtle;
  static final List<BoxShadow> button = subtle;
  static final List<BoxShadow> modal = medium;
  static final List<BoxShadow> dialog = strong;
  static final List<BoxShadow> fab = layered; // Floating Action Button
  
  // Elevation helpers (Material Design inspired)
  static List<BoxShadow> elevation(int level) {
    switch (level) {
      case 0:
        return [];
      case 1:
      case 2:
        return subtle;
      case 3:
      case 4:
        return medium;
      case 5:
      case 6:
        return strong;
      default:
        return layered;
    }
  }
}
