/// Border radius constants for Mac-inspired design
/// Provides consistent, rounded corners throughout the app
library;

import 'package:flutter/widgets.dart';

class BorderRadii {
  BorderRadii._(); // Private constructor to prevent instantiation

  // Base border radius scale
  static const double sm = 8.0;    // Small elements (chips, tags)
  static const double md = 12.0;   // Medium elements (cards, buttons)
  static const double lg = 16.0;   // Large elements (modals, sheets)
  static const double xl = 20.0;   // Extra large elements (hero cards)
  static const double full = 999.0; // Fully rounded (pills, avatars)
  
  // Semantic border radius (for specific use cases)
  static const double button = md;        // 12
  static const double card = md;          // 12
  static const double input = md;         // 12
  static const double modal = lg;         // 16
  static const double chip = full;        // 999 (pill shape)
  static const double avatar = full;      // 999 (circle)
  
  // BorderRadius helpers
  static const borderRadiusSM = BorderRadius.all(Radius.circular(sm));
  static const borderRadiusMD = BorderRadius.all(Radius.circular(md));
  static const borderRadiusLG = BorderRadius.all(Radius.circular(lg));
  static const borderRadiusXL = BorderRadius.all(Radius.circular(xl));
  static const borderRadiusFull = BorderRadius.all(Radius.circular(full));
  
  // Semantic BorderRadius helpers
  static const buttonBorderRadius = borderRadiusMD;
  static const cardBorderRadius = borderRadiusMD;
  static const inputBorderRadius = borderRadiusMD;
  static const modalBorderRadius = borderRadiusLG;
  static const chipBorderRadius = borderRadiusFull;
}
