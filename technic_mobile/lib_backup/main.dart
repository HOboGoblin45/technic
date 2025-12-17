import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import 'screens/home_screen.dart';
import 'screens/scanner_screen.dart';
import 'screens/watchlist_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/scanner_test_screen.dart';
import 'theme/app_theme.dart';
import 'providers/scanner_provider.dart';

void main() {
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ScannerProvider()),
      ],
      child: const TechnicApp(),
    ),
  );
}

class TechnicApp extends StatelessWidget {
  const TechnicApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'Technic Scanner',
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.dark, // Default to dark theme (Technic aesthetic)
      routerConfig: _router,
      debugShowCheckedModeBanner: false,
    );
  }
}

// Router configuration
final _router = GoRouter(
  initialLocation: '/', // Start with home screen (original design)
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => const HomeScreen(),
    ),
    GoRoute(
      path: '/scanner',
      builder: (context, state) => const ScannerScreen(),
    ),
    GoRoute(
      path: '/watchlist',
      builder: (context, state) => const WatchlistScreen(),
    ),
    GoRoute(
      path: '/settings',
      builder: (context, state) => const SettingsScreen(),
    ),
    GoRoute(
      path: '/test',
      builder: (context, state) => const ScannerTestScreen(),
    ),
  ],
);
