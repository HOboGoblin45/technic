/// App Shell
/// 
/// Main navigation shell with bottom tab bar and page management.
library;

import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

import 'screens/scanner/scanner_page.dart';
import 'screens/ideas/ideas_page.dart';
import 'screens/copilot/copilot_page.dart';
import 'screens/my_ideas/my_ideas_page.dart';
import 'screens/settings/settings_page.dart';
import 'services/local_store.dart';
import 'theme/app_colors.dart';
import 'utils/helpers.dart';

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
    MyIdeasPage(),
    SettingsPage(),
  ];

  final List<String> _titles = const [
    'Scanner',
    'Ideas',
    'Copilot',
    'My Ideas',
    'Settings',
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
    
    
    
    
    
    final navBackground = const Color(0xFF0F1C31);  // Dark blue for both modes

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(70),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: BoxDecoration(
            color: const Color(0xFF0F1C31),  // Dark blue for both modes
            border: Border(bottom: BorderSide(color: isDark ? AppColors.darkBorder : const Color(0xFFE5E7EB))),
          ),
          child: SafeArea(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Logo
                Container(
                  width: 34,
                  height: 34,
                  decoration: BoxDecoration(
                    color: AppColors.primaryBlue,  // Light blue background
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: AppColors.primaryBlue),
                    boxShadow: [
                      BoxShadow(
                        color: tone(Colors.black, 0.15),
                        blurRadius: 6,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  alignment: Alignment.center,
                  child: SvgPicture.asset(
                    'assets/logo_tq.svg',
                    width: 20,
                    height: 20,
                    colorFilter: const ColorFilter.mode(
                      AppColors.darkBackground,  // Dark blue lettering
                      BlendMode.srcIn,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                
                // Title
                const Text(
                  'technic',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.2,
                  ),
                ),
              ],
            ),
          ),
        ),
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
      
      // Bottom navigation bar
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: navBackground,
          border: const Border(top: BorderSide(color: Color(0xFF0F172A))),
          boxShadow: [
            BoxShadow(
              color: tone(Colors.black, 0.35),
              blurRadius: 12,
              offset: const Offset(0, -6),
            ),
          ],
        ),
        child: NavigationBar(
          selectedIndex: _index,
          backgroundColor: Colors.transparent,
          indicatorColor: tone(AppColors.primaryBlue, 0.18),
          height: 70,
          labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
          onDestinationSelected: (value) {
            setState(() => _index = value);
            LocalStore.saveLastTab(value);
          },
          destinations: const [
            NavigationDestination(
              icon: Icon(Icons.assessment_outlined),
              selectedIcon: Icon(Icons.assessment),
              label: 'Scan',
            ),
            NavigationDestination(
              icon: Icon(Icons.lightbulb_outline),
              selectedIcon: Icon(Icons.lightbulb),
              label: 'Ideas',
            ),
            NavigationDestination(
              icon: Icon(Icons.chat_bubble_outline),
              selectedIcon: Icon(Icons.chat_bubble),
              label: 'Copilot',
            ),
            NavigationDestination(
              icon: Icon(Icons.star_border),
              selectedIcon: Icon(Icons.star),
              label: 'My Ideas',
            ),
            NavigationDestination(
              icon: Icon(Icons.settings_outlined),
              selectedIcon: Icon(Icons.settings),
              label: 'Settings',
            ),
          ],
        ),
      ),
    );
  }
}
