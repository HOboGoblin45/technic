/// Spacing constants for Mac-inspired design
/// Provides consistent, generous spacing throughout the app
library;

import 'package:flutter/widgets.dart';

class Spacing {
  Spacing._(); // Private constructor to prevent instantiation

  // Base spacing scale (8pt grid system)
  static const double xs = 4.0;    // Extra small - tight spacing
  static const double sm = 8.0;    // Small - compact elements
  static const double md = 16.0;   // Medium - default spacing
  static const double lg = 24.0;   // Large - section spacing
  static const double xl = 32.0;   // Extra large - major sections
  static const double xxl = 48.0;  // Extra extra large - page margins
  
  // Semantic spacing (for specific use cases)
  static const double cardPadding = md;           // 16
  static const double screenPadding = lg;         // 24
  static const double sectionSpacing = xl;        // 32
  static const double elementSpacing = sm;        // 8
  static const double iconTextSpacing = sm;       // 8
  static const double buttonPadding = md;         // 16
  static const double inputPadding = md;          // 16
  
  // Vertical spacing
  static const double listItemSpacing = md;       // 16
  static const double paragraphSpacing = md;      // 16
  static const double headingSpacing = lg;        // 24
  
  // Horizontal spacing
  static const double buttonSpacing = md;         // 16
  static const double iconSpacing = lg;           // 24
  
  // Edge insets helpers
  static const edgeInsetsXS = EdgeInsets.all(xs);
  static const edgeInsetsSM = EdgeInsets.all(sm);
  static const edgeInsetsMD = EdgeInsets.all(md);
  static const edgeInsetsLG = EdgeInsets.all(lg);
  static const edgeInsetsXL = EdgeInsets.all(xl);
  static const edgeInsetsXXL = EdgeInsets.all(xxl);
  
  // Symmetric padding helpers
  static const horizontalPaddingSM = EdgeInsets.symmetric(horizontal: sm);
  static const horizontalPaddingMD = EdgeInsets.symmetric(horizontal: md);
  static const horizontalPaddingLG = EdgeInsets.symmetric(horizontal: lg);
  
  static const verticalPaddingSM = EdgeInsets.symmetric(vertical: sm);
  static const verticalPaddingMD = EdgeInsets.symmetric(vertical: md);
  static const verticalPaddingLG = EdgeInsets.symmetric(vertical: lg);
  
  // Screen-specific padding
  static const screenEdgeInsets = EdgeInsets.all(screenPadding);
  static const cardEdgeInsets = EdgeInsets.all(cardPadding);
}
