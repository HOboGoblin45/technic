/// App Shell
///
/// Main navigation shell with bottom tab bar and page management.
library;

import 'package:flutter/material.dart';

import 'screens/scanner/scanner_page.dart';
import 'screens/ideas/ideas_page.dart';
import 'screens/copilot/copilot_page.dart';
import 'screens/watchlist/watchlist_page.dart';
import 'screens/settings/settings_page.dart';
import 'services/local_store.dart';
import 'theme/app_colors.dart';
import 'widgets/premium_bottom_nav.dart';
import 'widgets/premium_app_bar.dart';

/// Main app shell with tab navigation
class TechnicShell extends StatefulWidget {
  const TechnicShell({super.key});

  @override
  State<TechnicShell> createState() => _TechnicShellState();
}

class _TechnicShellState extends State<TechnicShell> {
  int _index = 0;

  final List<Widget> _pages = const [
    ScannerPage(),
    IdeasPage(),
    CopilotPage(),
    WatchlistPage(),
    SettingsPage(),
  ];


  @override
  void initState() {
    super.initState();
    _restoreTab();
  }

  Future<void> _restoreTab() async {
    final saved = await LocalStore.loadLastTab();
    if (saved != null && mounted) {
      setState(() {
        _index = saved.clamp(0, _pages.length - 1);
      });
    }
  }

  void setTab(int index) {
    setState(() {
      _index = index;
    });
    LocalStore.saveLastTab(index);
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      extendBodyBehindAppBar: true,
      // Premium App Bar (Phase 4.2)
      appBar: PremiumAppBar(
        title: 'technic',
        logoAsset: 'assets/logo_tq.svg',
        showSearch: true,
        searchHint: 'Search stocks...',
        onSearchChanged: (query) {
          // TODO: Implement global search
          debugPrint('Search query: $query');
        },
      ),
      
      // Body with gradient background
      body: Container(
        color: isDark ? AppColors.darkBackground : const Color(0xFFF9FAFB),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 250),
              child: _pages[_index],
            ),
          ),
        ),
      ),
      
      // Premium Bottom Navigation (Phase 4.1)
      bottomNavigationBar: PremiumBottomNav(
        currentIndex: _index,
        onTap: (index) {
          setState(() => _index = index);
          LocalStore.saveLastTab(index);
        },
        items: createTechnicNavItems(),
        enableHaptics: true,
      ),
    );
  }
}
