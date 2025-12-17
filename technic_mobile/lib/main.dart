/// Technic App - Entry Point
/// 
/// Quantitative trading companion with AI-powered insights.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app_shell.dart';
import 'providers/app_providers.dart';
import 'services/storage_service.dart';
import 'theme/app_theme.dart';
import 'screens/splash/splash_screen.dart';

// ============================================================================
// GLOBAL STATE
// ============================================================================

/// Theme mode notifier
final ValueNotifier<bool> themeIsDark = ValueNotifier<bool>(false);

/// Options mode preference: "stock_only" or "stock_plus_options"
final ValueNotifier<String> optionsMode = ValueNotifier<String>('stock_plus_options');

/// User ID
final ValueNotifier<String?> userId = ValueNotifier<String?>(null);

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

Future<void> main() async {
  // Initialize Flutter bindings
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize storage
  final storage = StorageService.instance;
  await storage.init();
  
  // Load theme preference
  final savedTheme = await storage.loadThemeMode();
  themeIsDark.value = savedTheme == 'dark';
  
  // Load options mode preference
  final savedOptionsMode = await storage.loadOptionsMode();
  if (savedOptionsMode != null) {
    optionsMode.value = savedOptionsMode;
  }
  
  // Load user ID
  userId.value = await storage.loadUser();
  
  // Run app with ProviderScope for state management
  runApp(
    const ProviderScope(
      child: TechnicApp(),
    ),
  );
}

// ============================================================================
// APP WIDGET
// ============================================================================

/// Main application widget
class TechnicApp extends ConsumerStatefulWidget {
  const TechnicApp({super.key});

  @override
  ConsumerState<TechnicApp> createState() => _TechnicAppState();
}

class _TechnicAppState extends ConsumerState<TechnicApp> {
  bool _showSplash = true;
  
  @override
  void initState() {
    super.initState();
    // Attempt auto-login on app start
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(authProvider.notifier).tryAutoLogin();
    });
  }

  void _onSplashComplete() {
    setState(() {
      _showSplash = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final isDark = ref.watch(themeModeProvider);
    
    return MaterialApp(
      title: 'Technic',
      debugShowCheckedModeBanner: false,
      
      // Theme configuration
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: isDark ? ThemeMode.dark : ThemeMode.light,
      
      // Home - Show splash first, then main app
      home: _showSplash
          ? SplashScreen(onComplete: _onSplashComplete)
          : const TechnicShell(),
    );
  }
}
