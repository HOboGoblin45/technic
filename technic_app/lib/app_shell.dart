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
    
    final headerGradient = isDark
        ? const [
            Color(0xFF0F1C31),
            Color(0xFF0B1324),
            Color(0xFF081910),
          ]
        : const [
            Color(0xFFF5F7FB),
            Color(0xFFE8EDF7),
            Color(0xFFDDE6F5),
          ];
    
    final bodyGradient = isDark
        ? [AppColors.pineGrove, tone(AppColors.darkDeep, 0.85)]
        : [Colors.white, tone(AppColors.skyBlue, 0.07)];
    
    final navBackground = isDark ? tone(AppColors.darkDeep, 0.9) : Colors.white;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(70),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: headerGradient,
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: SafeArea(
            child: Row(
              children: [
                // Logo
                Container(
                  width: 34,
                  height: 34,
                  decoration: BoxDecoration(
                    color: const Color(0xFF0F1B33),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: const Color(0xFF1E3258)),
                    boxShadow: [
                      BoxShadow(
                        color: tone(Colors.black, 0.35),
                        blurRadius: 18,
                        offset: const Offset(0, 10),
                      ),
                    ],
                  ),
                  alignment: Alignment.center,
                  child: SvgPicture.asset(
                    'assets/logo_tq.svg',
                    width: 20,
                    height: 20,
                    colorFilter: const ColorFilter.mode(
                      Colors.white,
                      BlendMode.srcIn,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                
                // Title and subtitle
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      'technic',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 0.2,
                      ),
                    ),
                    Text(
                      _titles[_index],
                      style: TextStyle(
                        color: tone(Colors.white, 0.7),
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
                const Spacer(),
                
                // Live indicator
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: tone(Colors.white, 0.08),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: tone(Colors.white, 0.08)),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.circle, size: 10, color: Color(0xFFB6FF3B)),
                      SizedBox(width: 6),
                      Text(
                        'Live',
                        style: TextStyle(
                          color: Color(0xFFB6FF3B),
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      
      // Body with gradient background
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: bodyGradient,
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
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
          indicatorColor: tone(AppColors.skyBlue, 0.18),
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
