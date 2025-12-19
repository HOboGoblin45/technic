/// Animation constants for Mac-inspired design
/// Provides smooth, purposeful animations throughout the app
library;

import 'package:flutter/animation.dart';

class Animations {
  Animations._(); // Private constructor to prevent instantiation

  // Duration constants
  static const Duration instant = Duration(milliseconds: 0);
  static const Duration fast = Duration(milliseconds: 150);
  static const Duration normal = Duration(milliseconds: 250);
  static const Duration slow = Duration(milliseconds: 350);
  static const Duration verySlow = Duration(milliseconds: 500);
  
  // Semantic durations (for specific use cases)
  static const Duration button = fast;           // 150ms - Quick feedback
  static const Duration pageTransition = normal; // 250ms - Smooth navigation
  static const Duration modal = normal;          // 250ms - Modal appearance
  static const Duration tooltip = fast;          // 150ms - Quick info
  static const Duration loading = slow;          // 350ms - Loading states
  
  // Curve constants (Mac-style easing)
  static const Curve easeOut = Curves.easeOut;
  static const Curve easeIn = Curves.easeIn;
  static const Curve easeInOut = Curves.easeInOut;
  static const Curve easeOutCubic = Curves.easeOutCubic;
  static const Curve easeInCubic = Curves.easeInCubic;
  static const Curve easeInOutCubic = Curves.easeInOutCubic;
  
  // Semantic curves (for specific use cases)
  static const Curve defaultCurve = easeOut;
  static const Curve emphasizedCurve = easeOutCubic;
  static const Curve deceleratedCurve = easeOut;
  static const Curve acceleratedCurve = easeIn;
  static const Curve standardCurve = easeInOut;
  
  // Spring animations (for bouncy effects)
  static const Curve spring = Curves.elasticOut;
  static const Curve bounce = Curves.bounceOut;
  
  // Stagger delays (for sequential animations)
  static const Duration staggerShort = Duration(milliseconds: 50);
  static const Duration staggerMedium = Duration(milliseconds: 100);
  static const Duration staggerLong = Duration(milliseconds: 150);
}
