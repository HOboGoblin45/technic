import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

// Brand palette (refined)
const brandPrimary = Color(0xFF99BFFF);
const brandAccent = Color(0xFF001D51);
const brandBg = Color(0xFF213631);
const brandDeep = Color(0xFF0A1214);

const defaultTickers = <String>[
  'AAPL',
  'MSFT',
  'NVDA',
  'TSLA',
  'AMZN',
  'GOOGL',
  'META',
  'NFLX',
  'AMD',
  'INTC',
  'JPM',
  'GS',
  'XOM',
  'CVX',
  'BA',
  'CAT',
  'LMT',
  'KO',
  'PEP',
  'WMT',
];

// Share last scan results across tabs so we don't lose them on navigation
final ValueNotifier<List<ScanResult>> lastScanResults = ValueNotifier<List<ScanResult>>([]);
final ValueNotifier<List<MarketMover>> lastMoversStore = ValueNotifier<List<MarketMover>>([]);

Color tone(Color base, double opacity) =>
    base.withAlpha((opacity * 255).clamp(0, 255).round());

final technicApi = TechnicApi();
final GlobalKey<_TechnicShellState> _shellKey = GlobalKey<_TechnicShellState>();
final ValueNotifier<String?> copilotPrefill = ValueNotifier<String?>(null);
final ValueNotifier<String?> copilotStatus = ValueNotifier<String?>(null);
final ValueNotifier<ScanResult?> copilotContext = ValueNotifier<ScanResult?>(null);
final ValueNotifier<bool> themeIsDark = ValueNotifier<bool>(false);
final ValueNotifier<String?> userId = ValueNotifier<String?>(null);

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  userId.value = await LocalStore.loadUser();
  runApp(const TechnicApp());
}

class TechnicApp extends StatelessWidget {
  const TechnicApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: themeIsDark,
      builder: (context, isDark, _) {
        final baseTextColor = isDark ? Colors.white : Colors.black87;
        final bg = isDark ? brandBg : Colors.white;
        final surface = isDark ? brandBg : Colors.white;
        final cardColor = isDark ? tone(brandDeep, 0.9) : Colors.white;
        final inputFill =
            isDark ? tone(Colors.white, 0.03) : tone(Colors.black, 0.04);
        final inputBorder = OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide:
              BorderSide(color: tone(isDark ? Colors.white : Colors.black, 0.08)),
        );

        final textTheme = (isDark ? ThemeData.dark() : ThemeData.light())
            .textTheme
            .apply(
              fontFamily: 'Inter',
              bodyColor: baseTextColor,
              displayColor: baseTextColor,
            );

        final colorScheme = ColorScheme.fromSeed(
          seedColor: brandPrimary,
          brightness: isDark ? Brightness.dark : Brightness.light,
          surface: surface,
        );

        final buttonBg = isDark ? brandAccent : brandPrimary;
        final chipSelected = isDark ? tone(brandPrimary, 0.25) : tone(brandAccent, 0.12);
        final chipBg = isDark ? tone(Colors.white, 0.08) : tone(Colors.black, 0.04);
        final snackBg = isDark ? tone(brandDeep, 0.9) : tone(brandPrimary, 0.12);
        final snackText = isDark ? Colors.white : Colors.black87;
        final snackAction = isDark ? brandPrimary : brandAccent;

        return MaterialApp(
          title: 'technic',
          theme: ThemeData(
            brightness: isDark ? Brightness.dark : Brightness.light,
            scaffoldBackgroundColor: bg,
            colorScheme: colorScheme,
            textTheme: textTheme,
            dialogTheme: DialogThemeData(
              backgroundColor: isDark ? tone(brandDeep, 0.92) : Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              titleTextStyle: TextStyle(
                color: baseTextColor,
                fontWeight: FontWeight.w800,
                fontSize: 16,
                fontFamily: 'Inter',
              ),
              contentTextStyle: TextStyle(color: baseTextColor, fontFamily: 'Inter'),
            ),
            chipTheme: ChipThemeData(
              backgroundColor: chipBg,
              selectedColor: chipSelected,
              labelStyle: TextStyle(color: baseTextColor),
              secondaryLabelStyle: TextStyle(color: baseTextColor),
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
            ),
            cardTheme: CardThemeData(
              color: cardColor,
              elevation: 0,
              margin: const EdgeInsets.symmetric(vertical: 6),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18),
              ),
            ),
            inputDecorationTheme: InputDecorationTheme(
              filled: true,
              fillColor: inputFill,
              border: inputBorder,
              enabledBorder: inputBorder,
              focusedBorder: inputBorder.copyWith(
                borderSide: BorderSide(color: tone(brandPrimary, 0.55)),
              ),
              hintStyle: TextStyle(color: tone(baseTextColor, 0.6)),
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 14,
                vertical: 12,
              ),
            ),
            elevatedButtonTheme: ElevatedButtonThemeData(
              style: ElevatedButton.styleFrom(
                backgroundColor: buttonBg,
                foregroundColor: isDark ? Colors.white : Colors.black87,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              ),
            ),
            outlinedButtonTheme: OutlinedButtonThemeData(
              style: OutlinedButton.styleFrom(
                foregroundColor: isDark ? brandPrimary : brandAccent,
                side: BorderSide(
                  color: tone(isDark ? brandPrimary : brandAccent, 0.7),
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              ),
            ),
            textButtonTheme: TextButtonThemeData(
              style: TextButton.styleFrom(
                foregroundColor: isDark ? brandPrimary : brandAccent,
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              ),
            ),
            snackBarTheme: SnackBarThemeData(
              backgroundColor: snackBg,
              contentTextStyle: TextStyle(color: snackText),
              actionTextColor: snackAction,
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            navigationBarTheme: NavigationBarThemeData(
              indicatorColor: tone(brandPrimary, 0.18),
              backgroundColor: Colors.transparent,
              elevation: 0,
              labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
              height: 70,
              iconTheme: WidgetStateProperty.resolveWith(
                (states) => IconThemeData(
                  color: states.contains(WidgetState.selected)
                      ? brandPrimary
                      : (isDark ? Colors.white70 : Colors.black54),
                ),
              ),
              labelTextStyle: WidgetStateProperty.resolveWith(
                (states) => TextStyle(
                  color: states.contains(WidgetState.selected)
                      ? brandPrimary
                      : (isDark ? Colors.white70 : Colors.black54),
                  fontWeight: states.contains(WidgetState.selected)
                      ? FontWeight.w700
                      : FontWeight.w500,
                ),
              ),
            ),
            appBarTheme: AppBarTheme(
              backgroundColor: Colors.transparent,
              elevation: 0,
              foregroundColor: baseTextColor,
            ),
            useMaterial3: true,
          ),
          home: TechnicShell(key: _shellKey),
          debugShowCheckedModeBanner: false,
        );
      },
    );
  }
}

class TechnicShell extends StatefulWidget {
  const TechnicShell({super.key});

  @override
  State<TechnicShell> createState() => _TechnicShellState();
}

class _TechnicShellState extends State<TechnicShell> {
  int _index = 0;

  final List<Widget> _pages = [
    const ScannerPage(),
    const IdeasPage(),
    const CopilotPage(),
    const SettingsPage(),
  ];

  final List<String> _titles = const [
    'Scanner',
    'Ideas',
    'Copilot',
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
        ? [brandBg, tone(brandDeep, 0.85)]
        : [Colors.white, tone(brandPrimary, 0.07)];
    final navBackground = isDark ? tone(brandDeep, 0.9) : Colors.white;

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
                  child: Row(
                    children: const [
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
          indicatorColor: tone(brandPrimary, 0.18),
          height: 70,
          labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
          onDestinationSelected: (value) => setState(() => _index = value),
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
              icon: Icon(Icons.settings_outlined),
              selectedIcon: Icon(Icons.settings),
              label: 'Settings',
            ),
          ],
        ),
      ),
    );
  }

  void setTab(int index) {
    setState(() {
      _index = index;
    });
    LocalStore.saveLastTab(index);
  }
}

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage> with AutomaticKeepAliveClientMixin {
  late Future<ScannerBundle> _bundleFuture;
  int _maxSymbols = 50;
  double _lookbackDays = 90;
  double _minTechRating = 0;
  bool _allowShorts = true;
  bool _onlyTradeable = false;
  String _tradeStyle = 'Short-term swing';
  String _riskProfile = 'Balanced';
  String _timeHorizon = 'Swing';
  List<SavedScreen> _savedScreens = List.of(savedScreens);
  bool _advancedMode = false;
  bool _showOnboarding = true;
  String? _progressText;
  final List<String> _sectors = [
    'Information Technology',
    'Health Care',
    'Industrials',
    'Financials',
    'Energy',
    'Consumer Discretionary',
    'Consumer Staples',
    'Communication Services',
    'Materials',
    'Real Estate',
    'Utilities',
  ];
  List<String> _selectedSectors = [];
  String _industryFilter = '';
  Map<String, int> _sectorCounts = {};
  int _totalSymbolsCap = 6000;
  final Map<String, int> _subindustryCounts = {};
  final List<String> _selectedSubindustries = [];
  int _capForSelection() {
    if (_selectedSectors.isEmpty && _selectedSubindustries.isEmpty) {
      return _totalSymbolsCap;
    }
    final sectorSum = _selectedSectors
        .map((s) => _sectorCounts[s] ?? 0)
        .fold<int>(0, (a, b) => a + b);
    final subSum = _selectedSubindustries
        .map((s) => _subindustryCounts[s] ?? 0)
        .fold<int>(0, (a, b) => a + b);
    final sum = sectorSum + subSum;
    return sum > 0 ? sum : _totalSymbolsCap;
  }
  int _scanCount = 0;
  int _streakDays = 0;
  DateTime? _lastScanDate;
  final TextEditingController _searchCtrl = TextEditingController();
  List<ScanResult> _lastScans = mockScanResults;
  List<MarketMover> _lastMovers = mockMovers;
  List<String> _searchHints = List.of(defaultTickers);
  bool _randomizeSingleSector = false;
  final FocusNode _searchFocus = FocusNode();

  void _applyBasicProfile() {
    // Map risk profile + time horizon to underlying scan parameters.
    // Risk profile: conservative = tighter filters, aggressive = wider net.
    setState(() {
      switch (_riskProfile) {
        case 'Conservative':
          _maxSymbols = 40;
          _minTechRating = 25;
          _allowShorts = false;
          _onlyTradeable = true;
          break;
        case 'Aggressive':
          _maxSymbols = 100;
          _minTechRating = 15;
          _allowShorts = true;
          _onlyTradeable = false;
          break;
        case 'Balanced':
        default:
          _maxSymbols = 60;
          _minTechRating = 20;
          _allowShorts = false;
          _onlyTradeable = true;
      }

      switch (_timeHorizon) {
        case 'Short-term':
          _lookbackDays = 45;
          _tradeStyle = 'Short-term swing';
          break;
        case 'Position':
          _lookbackDays = 180;
          _tradeStyle = 'Multi-day';
          break;
        case 'Swing':
        default:
          _lookbackDays = 90;
          _tradeStyle = 'Short-term swing';
      }

      // Respect current sector caps.
      final cap = _capForSelection();
      if (_maxSymbols > cap) _maxSymbols = cap;
    });
  }

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    // Start with cached/notifier data; only run when user taps Run
    _bundleFuture = Future.value(
      ScannerBundle(
        movers: lastMoversStore.value.isNotEmpty ? lastMoversStore.value : mockMovers,
        scanResults: lastScanResults.value.isNotEmpty ? lastScanResults.value : mockScanResults,
        scoreboard: scoreboardSlices,
      ),
    );
    _hydrate();
    _loadUniverseStats();
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    _searchFocus.dispose();
    super.dispose();
  }

  Future<void> _hydrate() async {
    final state = await LocalStore.loadScannerState();
    if (state == null) return;
    final filters = Map<String, String>.from(state['filters'] as Map);
    final saved = (state['saved_screens'] as List<SavedScreen>?) ?? savedScreens;
    final lastScanStr = state['lastScan'] as String?;
    final cachedScans = state['last_scans'] as List<ScanResult>? ?? [];
    final cachedMovers = state['last_movers'] as List<MarketMover>? ?? [];
    setState(() {
      _maxSymbols = int.tryParse(filters['max_symbols'] ?? '') ?? _maxSymbols;
      _lookbackDays =
          double.tryParse(filters['lookback_days'] ?? '') ?? _lookbackDays;
      _minTechRating =
          double.tryParse(filters['min_tech_rating'] ?? '') ?? _minTechRating;
      _allowShorts = (filters['allow_shorts'] ?? 'true') == 'true';
      _onlyTradeable = (filters['only_tradeable'] ?? 'false') == 'true';
      _tradeStyle = filters['trade_style'] ?? _tradeStyle;
      _selectedSectors = (filters['sectors'] ?? '')
          .split(',')
          .where((e) => e.trim().isNotEmpty)
          .toList();
      _searchHints = {
        ...defaultTickers,
        ..._searchHints,
        ...cachedScans.map((e) => e.ticker),
        ...cachedMovers.map((e) => e.ticker),
      }.toList()..sort();
      _savedScreens = List.of(saved);
      _scanCount = state['scanCount'] as int? ?? 0;
      _streakDays = state['streakDays'] as int? ?? 0;
      _lastScanDate =
          lastScanStr != null ? DateTime.tryParse(lastScanStr) : _lastScanDate;
      _advancedMode = state['advancedMode'] as bool? ?? _advancedMode;
      _showOnboarding = state['showOnboarding'] as bool? ?? _showOnboarding;
      // Do NOT auto-run a scan on load; wait for user action
      _bundleFuture = Future.value(
        ScannerBundle(
          movers: cachedMovers.isNotEmpty
              ? cachedMovers
              : (lastMoversStore.value.isNotEmpty ? lastMoversStore.value : mockMovers),
          scanResults: cachedScans.isNotEmpty
              ? cachedScans
              : (lastScanResults.value.isNotEmpty ? lastScanResults.value : mockScanResults),
          scoreboard: scoreboardSlices,
        ),
      );
      if (cachedScans.isNotEmpty) {
        lastScanResults.value = cachedScans;
      }
      if (cachedMovers.isNotEmpty) {
        lastMoversStore.value = cachedMovers;
      }
    });
  }

  Future<void> _refresh() async {
    setState(() {
      _progressText = 'Starting scan...';
      _bundleFuture = technicApi.fetchScannerBundle(params: _filtersMap());
    });
    final bundle = await _bundleFuture;
    // Save for reuse across tabs
    lastScanResults.value = bundle.scanResults;
    lastMoversStore.value = bundle.movers;
    _progressText = bundle.progress ?? _progressText;
    await LocalStore.saveLastBundle(bundle);
    _incrementProgress();
    if (_showOnboarding) {
      setState(() => _showOnboarding = false);
    }
    _persistState();
  }

  // Keep for future lazy-load; currently unused to avoid auto-scanning on launch.
  // ignore: unused_element
  Future<void> _refreshIfEmpty() async {
    if (lastScanResults.value.isNotEmpty) return;
    setState(() {
      _bundleFuture = technicApi.fetchScannerBundle(
        params: _filtersMap(),
      );
    });
    final bundle = await _bundleFuture;
    lastScanResults.value = bundle.scanResults;
    lastMoversStore.value = bundle.movers;
    _progressText = bundle.progress ?? _progressText;
    await LocalStore.saveLastBundle(bundle);
  }

  Map<String, String> _filtersMap() {
    return {
      'max_symbols': _maxSymbols.toString(),
      'lookback_days': _lookbackDays.round().toString(),
      'min_tech_rating': _minTechRating.round().toString(),
      'allow_shorts': _allowShorts.toString(),
      'only_tradeable': _onlyTradeable.toString(),
      'trade_style': _tradeStyle,
      if (_selectedSectors.isNotEmpty) 'sectors': _selectedSectors.join(','),
      if (_selectedSubindustries.isNotEmpty) 'subindustries': _selectedSubindustries.join(','),
      if (_industryFilter.trim().isNotEmpty) 'industry': _industryFilter.trim(),
    };
  }

Future<void> _loadUniverseStats() async {
  try {
    final stats = await technicApi.fetchUniverseStats();
    if (!mounted || stats == null) return;
    setState(() {
      _sectorCounts = stats.sectors;
      _totalSymbolsCap = stats.total > 0 ? stats.total : _totalSymbolsCap;
      // _subindustryCounts is final; update its contents instead of reassigning.
      _subindustryCounts.clear();
      _subindustryCounts.addAll(stats.subindustries);
      final cap = _capForSelection();
      if (_maxSymbols > cap) _maxSymbols = cap;
    });
  } catch (_) {}
}

  void _applyQuickAction(QuickAction action) {
    // Map quick actions to preset filter sets; adjust as needed.
    var shouldRefresh = true;
    switch (action.label) {
      case 'Fast scan':
        _maxSymbols = 50;
        _lookbackDays = 60;
        _minTechRating = 10;
        _allowShorts = true;
        _onlyTradeable = false;
        _tradeStyle = 'Short-term swing';
        break;
      case 'Filters':
        _maxSymbols = 100;
        _lookbackDays = 180;
        _minTechRating = 20;
        _allowShorts = false;
        _onlyTradeable = true;
        _tradeStyle = 'Weekly';
        break;
      case 'Refresh':
        // Keep filters, just rerun
        break;
      case 'Save as preset':
        _savePreset();
        shouldRefresh = false;
        break;
      case 'Layout':
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Layout density toggled (coming soon).'),
            duration: Duration(seconds: 1),
          ),
        );
        shouldRefresh = false;
        break;
      case 'Send to Copilot':
        copilotPrefill.value = 'Summarize my latest scan results';
        _shellKey.currentState?.setTab(2);
        shouldRefresh = false;
        break;
      case 'Randomize':
        _randomizeScan();
        shouldRefresh = false;
        break;
      default:
        break;
    }
    if (shouldRefresh) {
      setState(() {
        _bundleFuture = technicApi.fetchScannerBundle(params: _filtersMap());
      });
      _persistState();
    }
  }

  void _randomizeScan() {
    final random = math.Random();
    _tradeStyle = ['Short-term swing', 'Weekly', 'Multi-day', 'Momentum'][random.nextInt(4)];
    _lookbackDays = [30, 60, 90, 180, 365][random.nextInt(5)].toDouble();
    _minTechRating = [0, 10, 20, 30, 40, 50][random.nextInt(6)].toDouble();
    _allowShorts = random.nextBool();
    _onlyTradeable = random.nextBool();
    final shuffled = List<String>.from(_sectors)..shuffle(random);
    final pickCount =
        _randomizeSingleSector ? 1 : random.nextInt(_sectors.length) + 1;
    _selectedSectors = shuffled.take(pickCount).toList();
    _bundleFuture = technicApi.fetchScannerBundle(params: _filtersMap());
    _incrementProgress();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Random scan started with surprise sectors.'),
        duration: Duration(seconds: 1),
      ),
    );
  }

  void _incrementProgress() {
    _scanCount += 1;
    final today = DateTime.now();
    if (_lastScanDate == null ||
        today.difference(_lastScanDate!).inDays > 1) {
      _streakDays = 1;
    } else if (today.difference(_lastScanDate!).inDays == 1) {
      _streakDays += 1;
    }
    _lastScanDate = today;
    _persistState();
  }

  Future<void> _persistState() async {
    await LocalStore.saveScannerState(
      filters: _filtersMap(),
      savedScreens: _savedScreens,
      scanCount: _scanCount,
      streakDays: _streakDays,
      lastScan: _lastScanDate,
      advancedMode: _advancedMode,
      showOnboarding: _showOnboarding,
    );
  }

  void _savePreset() {
    final newPreset = SavedScreen(
      'Preset ${_savedScreens.length + 1}',
      'Custom filters',
      _tradeStyle,
      false,
      params: _filtersMap(),
    );
    setState(() {
      _savedScreens = [..._savedScreens, newPreset];
    });
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Preset saved. Long-press a preset to apply.'),
        duration: Duration(seconds: 1),
      ),
    );
    _persistState();
  }

  void _applyPresetFromScreen(SavedScreen screen) {
    if (screen.params == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No filter data saved for this preset.'),
          duration: Duration(seconds: 1),
        ),
      );
      return;
    }
    final p = screen.params!;
    setState(() {
      _maxSymbols = int.tryParse(p['max_symbols'] ?? '') ?? _maxSymbols;
      _lookbackDays = double.tryParse(p['lookback_days'] ?? '') ?? _lookbackDays;
      _minTechRating = double.tryParse(p['min_tech_rating'] ?? '') ?? _minTechRating;
      _allowShorts = (p['allow_shorts'] ?? 'true') == 'true';
      _onlyTradeable = (p['only_tradeable'] ?? 'false') == 'true';
      _tradeStyle = p['trade_style'] ?? _tradeStyle;
      _bundleFuture = technicApi.fetchScannerBundle(params: _filtersMap());
      _setActivePreset(screen);
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Applied preset "${screen.name}"'),
        duration: const Duration(seconds: 1),
      ),
    );
    _persistState();
  }

  void _setActivePreset(SavedScreen active) {
    _savedScreens = _savedScreens
        .map(
          (s) => s == active ? s.copyWith(isActive: true) : s.copyWith(isActive: false),
        )
        .toList();
  }

  Widget _filterPanel(BuildContext context) {
    return _infoCard(
      title: 'Scan settings',
      subtitle: 'Tune the feed before running a scan',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Profile',
                style: TextStyle(fontWeight: FontWeight.w800),
              ),
              Row(
                children: [
                  FilterChip(
                    label: const Text('Save as preset'),
                    selected: false,
                    tooltip: 'Save current filters as a preset',
                    onSelected: (_) => _savePreset(),
                  ),
                  const SizedBox(width: 6),
                  FilterChip(
                    label: Text(_advancedMode ? 'Advanced' : 'Basic'),
                    selected: _advancedMode,
                    tooltip: 'Toggle between basic and advanced controls',
                    onSelected: (_) => setState(() => _advancedMode = !_advancedMode),
                  ),
                  const SizedBox(width: 6),
                  ActionChip(
                    label: Text(
                      'Active filters',
                      style: TextStyle(color: tone(Colors.white, 0.7)),
                    ),
                    avatar: Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: brandPrimary,
                        shape: BoxShape.circle,
                      ),
                    ),
                    onPressed: () {},
                    backgroundColor: tone(Colors.white, 0.06),
                    visualDensity: VisualDensity.compact,
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 6),
          if (!_advancedMode) ...[
            const Text(
              'Step 1 – Pick your risk profile',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            Wrap(
              spacing: 8,
              runSpacing: 6,
              children: ['Conservative', 'Balanced', 'Aggressive']
                  .map(
                    (p) => ChoiceChip(
                      label: Text(p),
                      selected: _riskProfile == p,
                      onSelected: (_) {
                        setState(() => _riskProfile = p);
                        _applyBasicProfile();
                      },
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 12),
            const Text(
              'Step 2 – Pick your time horizon',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            Wrap(
              spacing: 8,
              runSpacing: 6,
              children: ['Short-term', 'Swing', 'Position']
                  .map(
                    (h) => ChoiceChip(
                      label: Text(h),
                      selected: _timeHorizon == h,
                      onSelected: (_) {
                        setState(() => _timeHorizon = h);
                        _applyBasicProfile();
                      },
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 12),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ElevatedButton.icon(
                  onPressed: () {
                    _applyBasicProfile();
                    _refresh();
                  },
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Run scan'),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Technic will pull a fresh scan with these defaults. Switch to Advanced for full control.',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.grey[600]),
                  ),
                ),
              ],
            ),
          ] else ...[
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Randomize single sector'),
              subtitle: const Text('Random scans will pick one sector only'),
              value: _randomizeSingleSector,
              onChanged: (v) => setState(() => _randomizeSingleSector = v),
            ),
            const SizedBox(height: 6),
            _sliderRow(
              label: 'Lookback (days)',
              value: _lookbackDays,
              min: 30,
              max: 365,
              divisions: 335,
              formatter: (v) => v.round().toString(),
              onChanged: (v) => setState(() => _lookbackDays = v),
            ),
            _sliderRow(
              label: 'Min TechRating',
              value: _minTechRating,
              min: 0,
              max: 100,
              divisions: 20,
              formatter: (v) => v.round().toString(),
              onChanged: (v) => setState(() => _minTechRating = v),
            ),
            _sliderRow(
              label: 'Max symbols',
              value: _maxSymbols.toDouble(),
              min: 10,
              max: _capForSelection().toDouble().clamp(10.0, 6000.0),
              divisions: 119,
              formatter: (v) => v.round().toString(),
              onChanged: (v) => setState(() => _maxSymbols = v.round()),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 4, bottom: 8),
              child: Text(
                'Selection size: ${_capForSelection()} symbols based on chosen sectors/subindustries',
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: Colors.grey[600]),
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              decoration: const InputDecoration(
                labelText: 'Industry contains',
                hintText: 'e.g., Semiconductors, Banks, Software',
              ),
              onChanged: (v) => setState(() => _industryFilter = v),
            ),
            const SizedBox(height: 10),
            if (_subindustryCounts.isNotEmpty)
              SizedBox(
                height: 140,
                child: SingleChildScrollView(
                  child: Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: _subindustryCounts.entries
                        .map(
                          (e) => FilterChip(
                            label: Text('${e.key} (${e.value})'),
                            selected: _selectedSubindustries.contains(e.key),
                            onSelected: (v) => setState(() {
                              if (v) {
                                _selectedSubindustries.add(e.key);
                              } else {
                                _selectedSubindustries.remove(e.key);
                              }
                              final cap = _capForSelection();
                              if (_maxSymbols > cap) _maxSymbols = cap;
                            }),
                          ),
                        )
                        .toList(),
                  ),
                ),
              ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 10,
              runSpacing: 8,
              children: [
                FilterChip(
                  label: const Text('Allow shorts'),
                  selected: _allowShorts,
                  onSelected: (v) => setState(() => _allowShorts = v),
                ),
                FilterChip(
                  label: const Text('Only tradeable'),
                  selected: _onlyTradeable,
                  onSelected: (v) => setState(() => _onlyTradeable = v),
                ),
                DropdownButton<String>(
                  value: _tradeStyle,
                  dropdownColor: tone(brandDeep, 0.95),
                  items: const [
                    'Short-term swing',
                    'Weekly',
                    'Multi-day',
                    'Momentum',
                  ].map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
                  onChanged: (v) {
                    if (v != null) setState(() => _tradeStyle = v);
                  },
                ),
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: _sectors
                      .map(
                        (s) => FilterChip(
                          label: Text(s),
                          selected: _selectedSectors.contains(s),
                          onSelected: (v) => setState(() {
                            if (v) {
                              _selectedSectors.add(s);
                            } else {
                              _selectedSectors.remove(s);
                            }
                            final cap = _capForSelection();
                            if (_maxSymbols > cap) _maxSymbols = cap;
                          }),
                        ),
                      )
                      .toList(),
                ),
                Tooltip(
                  message: 'Current filter summary',
                  child: Chip(
                    label: Text(
                      '${_filtersMap()['lookback_days']}d • min TR ${_filtersMap()['min_tech_rating']} • max ${_filtersMap()['max_symbols']}',
                      style: TextStyle(color: tone(Colors.white, 0.8)),
                    ),
                    backgroundColor: tone(Colors.white, 0.05),
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _refresh,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Apply & run'),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _sliderRow({
    required String label,
    required double value,
    required double min,
    required double max,
    int? divisions,
    required String Function(double) formatter,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontWeight: FontWeight.w700)),
            Text(formatter(value), style: const TextStyle(color: Colors.white70)),
          ],
        ),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          label: formatter(value),
          onChanged: onChanged,
        ),
        const SizedBox(height: 6),
      ],
    );
  }

  Future<void> _analyzeWithCopilot(ScanResult r) async {
    final prompt =
        'Analyze ${r.ticker}: ${r.signal}, entry ${r.entry}, stop ${r.stop}, target ${r.target}. Note: ${r.note}';
    try {
      final reply = await technicApi.sendCopilot(prompt);
      if (!mounted) return;
      await showModalBottomSheet(
        context: context,
        backgroundColor: tone(brandDeep, 0.95),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        builder: (_) => Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Copilot on ${r.ticker}',
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 10),
              Text(reply.body, style: const TextStyle(color: Colors.white)),
              if (reply.meta != null) ...[
                const SizedBox(height: 8),
                Text(
                  reply.meta!,
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ],
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      copilotStatus.value = e.toString();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text(
            'Copilot unavailable. Cached guidance will appear until service recovers.',
          ),
          action: SnackBarAction(
            label: 'Open Copilot',
            onPressed: () {
              copilotPrefill.value = 'Retry the last analysis';
              _shellKey.currentState?.setTab(2);
            },
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return RefreshIndicator(
      onRefresh: _refresh,
      child: FutureBuilder<ScannerBundle>(
        future: _bundleFuture,
        builder: (context, snapshot) {
          if (snapshot.hasData) {
          }
          final bundle = snapshot.data;
          final movers = bundle?.movers ?? mockMovers;
          final scans = bundle?.scanResults ?? mockScanResults;
          final scoreboard = bundle?.scoreboard ?? scoreboardSlices;
          final loading = snapshot.connectionState == ConnectionState.waiting;
          final hasError = snapshot.hasError;
          final hasScans = scans.isNotEmpty;
          final hasMovers = movers.isNotEmpty;
          final hasScoreboard = scoreboard.isNotEmpty;
          _lastScans = scans;
          _lastMovers = movers;
          lastScanResults.value = scans;
          lastMoversStore.value = movers;
          // Expand search hints with live data.
          _searchHints = {
            ...defaultTickers,
            ..._searchHints,
            ...scans.map((e) => e.ticker),
            ...movers.map((e) => e.ticker),
          }.toList()
            ..sort();
          if (_searchHints.length > 20) {
            _searchHints = _searchHints.take(20).toList();
          }
          final staleScan = _lastScanDate != null &&
              DateTime.now().difference(_lastScanDate!).inDays >= 7;
          final progressLine = bundle?.progress ?? _progressText;

          return ListView(
            children: [
              if (_showOnboarding) _onboardingCard(),
          if (loading)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: _infoCard(
                title: 'Scanning?',
                subtitle:
                    'Working through $_maxSymbols symbols${_selectedSectors.isNotEmpty ? " in ${_selectedSectors.join(", ")}" : ""}',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    LinearProgressIndicator(
                      minHeight: 4,
                      color: brandPrimary,
                      backgroundColor: Colors.white24,
                      value: null,
                    ),
                    const SizedBox(height: 6),
                    if (progressLine != null)
                      Text(
                        'Now scanning: $progressLine',
                        style: const TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                  ],
                ),
              ),
            ),
              ValueListenableBuilder<String?>(
                valueListenable: copilotStatus,
                builder: (context, status, _) {
                  if (status == null) return const SizedBox.shrink();
                  return Column(
                    children: [
                      _infoCard(
                        title: 'Copilot offline',
                        subtitle:
                            'Cached guidance will display until service recovers.',
                        child: Row(
                          children: [
                            Expanded(
                              child: Text(
                                status,
                                style: const TextStyle(color: Colors.white70),
                              ),
                            ),
                            const SizedBox(width: 12),
                            OutlinedButton(
                              onPressed: () {
                                copilotPrefill.value =
                                    'Retry the last Copilot question';
                                _shellKey.currentState?.setTab(2);
                              },
                              child: const Text('Open Copilot'),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 8),
                    ],
                  );
                },
              ),
              if (staleScan)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: _infoCard(
                    title: 'It?s been a while',
                    subtitle: 'Run a fresh scan to refresh your signals.',
                    child: ElevatedButton.icon(
                      onPressed: _refresh,
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Run scan now'),
                    ),
                  ),
                ),
              _heroBanner(
                context,
                title: 'Unified Scanner',
                subtitle:
                    'Saved presets, global search, and inline Copilot to reduce clicks.',
                badge: 'Pro ready',
                trailing: ElevatedButton.icon(
                  onPressed: _refresh,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Run scan'),
                ),
                child: Column(
                  children: [
                    RawAutocomplete<String>(
                      textEditingController: _searchCtrl,
                      focusNode: _searchFocus,
                      optionsBuilder: (text) {
                        final query = text.text.trim().toUpperCase();
                        if (query.isEmpty) return const Iterable<String>.empty();
                        return _searchHints.where(
                          (h) => h.toUpperCase().startsWith(query),
                        );
                      },
                      optionsViewBuilder: (context, onSelected, options) {
                        return Align(
                          alignment: Alignment.topLeft,
                          child: Material(
                            color: tone(brandDeep, 0.95),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                              side: BorderSide(color: tone(Colors.white, 0.08)),
                            ),
                            child: ListView(
                              padding: const EdgeInsets.symmetric(vertical: 6),
                              shrinkWrap: true,
                              children: options
                                  .map(
                                    (o) => ListTile(
                                      dense: true,
                                      title: Text(o),
                                      onTap: () {
                                        onSelected(o);
                                        _quickSearch(o);
                                      },
                                    ),
                                  )
                                  .toList(),
                            ),
                          ),
                        );
                      },
                      fieldViewBuilder:
                          (context, controller, focusNode, onFieldSubmitted) {
                        return TextField(
                          controller: controller,
                          focusNode: focusNode,
                          decoration: InputDecoration(
                            hintText: 'Search tickers, news, or features',
                            prefixIcon: const Icon(Icons.search),
                            suffixIcon: IconButton(
                              icon: const Icon(Icons.tune),
                              onPressed: () {},
                              tooltip: 'Advanced search (coming soon)',
                            ),
                          ),
                          onSubmitted: (v) {
                            onFieldSubmitted();
                            _quickSearch(v);
                          },
                        );
                      },
                ),
                const SizedBox(height: 6),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Wrap(
                    spacing: 8,
                    runSpacing: 6,
                    children: _searchHints
                        .map(
                          (t) => ActionChip(
                            label: Text(t),
                            onPressed: () => _quickSearch(t),
                            backgroundColor: tone(Colors.white, 0.05),
                            visualDensity: VisualDensity.compact,
                          ),
                        )
                        .toList(),
                  ),
                ),
                const SizedBox(height: 6),
                if (_selectedSectors.isNotEmpty || _selectedSubindustries.isNotEmpty || _industryFilter.isNotEmpty)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Active filters: ${[
                        if (_selectedSectors.isNotEmpty) 'Sectors: ${_selectedSectors.join(", ")}',
                        if (_selectedSubindustries.isNotEmpty) 'Subindustries: ${_selectedSubindustries.join(", ")}',
                        if (_industryFilter.isNotEmpty) 'Industry contains: $_industryFilter',
                      ].join(" | ")}',
                      style: const TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                  ),
                const SizedBox(height: 12),
                    Row(
                      children: const [
                        _PulseBadge(label: 'Volume surge +18%'),
                        SizedBox(width: 8),
                        _PulseBadge(label: 'Breadth improving'),
                        SizedBox(width: 8),
                        _PulseBadge(label: 'Earnings heavy week'),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              _filterPanel(context),
              const SizedBox(height: 8),
              _activeFilterSummary(),
              _sinceLastCard(),
              if (hasError) ...[
                const SizedBox(height: 10),
                _infoCard(
                  title: 'Live scan unavailable',
                  subtitle: snapshot.error.toString(),
                  child: TextButton(
                    onPressed: _refresh,
                    child: const Text('Retry'),
                  ),
                ),
              ],
              if (loading) ...[
                const SizedBox(height: 12),
                _infoCard(
                  title: 'Fetching live data...',
                  subtitle: 'Connecting to Streamlit API',
                  child: const LinearProgressIndicator(
                    minHeight: 4,
                    color: brandPrimary,
                    backgroundColor: Colors.white24,
                  ),
                ),
              ],
              const SizedBox(height: 16),
              SectionHeader(
                'Quick actions',
                caption: 'Guided workflows',
                trailing: TextButton(
                  onPressed: () {},
                  child: const Text('Customize'),
                ),
              ),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: quickActions.map((a) {
                  return Tooltip(
                    message: a.hint,
                    child: InkResponse(
                      onTap: () => _applyQuickAction(a),
                      borderRadius: BorderRadius.circular(12),
                      splashColor: tone(brandPrimary, 0.25),
                      highlightShape: BoxShape.rectangle,
                      child: MouseRegion(
                        cursor: SystemMouseCursors.click,
                        child: _pillButton(a.icon, a.label, a.hint),
                      ),
                    ),
                  );
                }).toList(),
              ),
              const SizedBox(height: 20),
              SectionHeader(
                'Saved screens',
                caption:
                    'Jump back into your favorite filters (${_savedScreens.length} saved)',
                trailing: TextButton(
                  onPressed: _openPresetManager,
                  child: const Text('Manage'),
                ),
              ),
              SizedBox(
                height: 160,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: _savedScreens.length,
                  separatorBuilder: (_, _) => const SizedBox(width: 12),
                  itemBuilder: (context, index) =>
                      _savedScreenCard(
                        context,
                        _savedScreens[index],
                        () => _applyPresetFromScreen(_savedScreens[index]),
                      ),
                ),
              ),
              const SizedBox(height: 20),
              SectionHeader(
                'Market pulse',
                caption: 'Live snapshot',
                trailing: TextButton(
                  onPressed: () {},
                  child: const Text('View snapshot'),
                ),
              ),
              if (!hasMovers)
                _infoCard(
                  title: 'No movers yet',
                  subtitle: 'Refresh or run a scan to populate market pulse.',
                  child: ElevatedButton.icon(
                    onPressed: _refresh,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Refresh movers'),
                  ),
                )
              else
                _marketPulseCard(context, movers),
              SectionHeader(
                'Scan results',
                caption: 'Ranked by technic score',
                trailing: TextButton(
                  onPressed: () {},
                  child: const Text('View all'),
                ),
              ),
              if (!hasScans)
                _infoCard(
                  title: 'No results yet',
                  subtitle: 'Adjust filters or run a scan to populate results.',
                  child: ElevatedButton.icon(
                    onPressed: _refresh,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Run scan'),
                  ),
                )
              else
                ...scans.map(
                  (r) => _scanResultCard(context, r, () {
                        copilotContext.value = r;
                        copilotPrefill.value =
                            'Explain the ${r.signal.toLowerCase()} setup in ${r.ticker} and outline an example trade using today\'s Technic scan metrics.';
                        _shellKey.currentState?.setTab(2);
                      }),
                ),
              if (hasScans)
                _infoCard(
                  title: 'More like ${scans.first.ticker}',
                  subtitle: 'We spotted similar setups based on your last view.',
                  child: Text(
                    'Try comparing ${scans.first.ticker} to peers in ${_selectedSectors.isEmpty ? 'top sectors' : _selectedSectors.join(', ')} or ask Copilot for ?similar momentum names?.',
                    style: const TextStyle(color: Colors.white70),
                  ),
                ),
              const SizedBox(height: 8),
              SectionHeader(
                'Scoreboard',
                caption: 'Track by strategy and horizon',
                trailing: TextButton(
                  onPressed: () {},
                  child: const Text('See breakdown'),
                ),
              ),
              if (!hasScoreboard)
                _infoCard(
                  title: 'No scoreboard data',
                  subtitle: 'Tag trades or run scans to see performance here.',
                  child: ElevatedButton.icon(
                    onPressed: _refresh,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Refresh'),
                  ),
                )
              else
                _scoreboardCard(scoreboard),
              const SizedBox(height: 8),
              SectionHeader(
                'Copilot inline',
                caption: 'Ask while you scan',
                trailing: TextButton(
                  onPressed: () {},
                  child: const Text('Open chat'),
                ),
              ),
              _copilotInlineCard(context),
            ],
          );
        },
      ),
    );
  }

  Widget _activeFilterSummary() {
    final f = _filtersMap();
    return _infoCard(
      title: 'Active filters',
      subtitle: 'Quick view of your current scan profile',
      child: Wrap(
        spacing: 8,
        runSpacing: 6,
        children: [
          Tooltip(
            message: 'Lookback window',
            child: Chip(
              label: Text('Lookback: ${f['lookback'] ?? 'N/A'}'),
              backgroundColor: tone(Colors.white, 0.05),
              visualDensity: VisualDensity.compact,
            ),
          ),
          Tooltip(
            message: 'Minimum Technic rating',
            child: Chip(
              label: Text('Active filters summary'),
              backgroundColor: tone(Colors.white, 0.05),
              visualDensity: VisualDensity.compact,
            ),
          ),
          Tooltip(
            message: 'Result cap',
            child: Chip(
              label: Text('Active filters summary'),
              backgroundColor: tone(Colors.white, 0.05),
              visualDensity: VisualDensity.compact,
            ),
          ),
          Tooltip(
            message: _allowShorts ? 'Including short setups' : 'Long-only results',
            child: Chip(
              label: Text(_allowShorts ? 'Shorts allowed' : 'Long only'),
              backgroundColor:
                  _allowShorts ? tone(brandPrimary, 0.15) : tone(Colors.white, 0.05),
              visualDensity: VisualDensity.compact,
            ),
          ),
          Tooltip(
            message: _onlyTradeable
                ? 'Filtered to liquid/tradeable symbols'
                : 'Including less liquid symbols',
            child: Chip(
              label: Text(_onlyTradeable ? 'Tradeable only' : 'Include illiquid'),
              backgroundColor: tone(Colors.white, 0.05),
              visualDensity: VisualDensity.compact,
            ),
          ),
          Tooltip(
            message: 'Trade style profile',
            child: Chip(
              label: Text(_tradeStyle),
              backgroundColor: tone(Colors.white, 0.05),
              visualDensity: VisualDensity.compact,
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _openPresetManager() async {
    await showModalBottomSheet(
      context: context,
      backgroundColor: tone(brandDeep, 0.95),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Saved presets',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
              ),
              const SizedBox(height: 8),
              if (_savedScreens.isEmpty)
                const Text(
                  'No presets yet. Save a scan to reuse it later.',
                  style: TextStyle(color: Colors.white70),
                )
              else
                ..._savedScreens.map(
                  (s) => ListTile(
                    dense: true,
                    title: Text(s.name),
                    subtitle: Text('${s.description} ? ${s.horizon}'),
                    trailing: OutlinedButton(
                      onPressed: () {
                        Navigator.pop(ctx);
                        _applyPresetFromScreen(s);
                      },
                      child: const Text('Apply'),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _sinceLastCard() {
    if (_lastScanDate == null) return const SizedBox.shrink();
    final fav = _selectedSectors.isEmpty ? 'All sectors' : _selectedSectors.join(', ');
    return _infoCard(
      title: 'Since your last scan',
      subtitle: 'Last run: ${_fmtLocalTime(_lastScanDate!)} | Focus: $fav',
      child: Row(
        children: [
          Chip(
            label: Text('Scans: $_scanCount'),
            backgroundColor: tone(Colors.white, 0.05),
            visualDensity: VisualDensity.compact,
          ),
          const SizedBox(width: 8),
          Chip(
            label: Text('Streak: $_streakDays d'),
            backgroundColor: tone(brandPrimary, 0.18),
            visualDensity: VisualDensity.compact,
          ),
          const SizedBox(width: 8),
          Chip(
            label: Text('Max $_maxSymbols | Lookback ${_lookbackDays.round()}d | TR >= ${_minTechRating.round()}'),
            backgroundColor: tone(Colors.white, 0.08),
            visualDensity: VisualDensity.compact,
          ),
          const Spacer(),
          OutlinedButton.icon(
            onPressed: _refresh,
            icon: const Icon(Icons.refresh),
            label: const Text('Refresh'),
          ),
        ],
      ),
    );
  }
void _quickSearch(String query) async {
    final q = query.trim().toUpperCase();
    if (q.isEmpty) return;
    try {
      final scanHit = _lastScans.firstWhere(
        (s) => s.ticker.toUpperCase() == q,
        orElse: () => _lastScans.firstWhere(
          (s) => s.ticker.toUpperCase().contains(q),
          orElse: () => _lastScans.isEmpty ? mockScanResults.first : _lastScans.first,
        ),
      );
      if (scanHit.ticker.toUpperCase() == q) {
        _showScanDetail(context, scanHit, () => _analyzeWithCopilot(scanHit));
        return;
      }
      final moverHit = _lastMovers.firstWhere(
        (m) => m.ticker.toUpperCase() == q,
        orElse: () => _lastMovers.isNotEmpty ? _lastMovers.first : mockMovers.first,
      );
      if (moverHit.ticker.toUpperCase() == q) {
        _showMoverDetail(context, moverHit);
        return;
      }
    } catch (_) {}

    // Fallback: fetch a single-symbol scan from the API
    try {
      final remote = await technicApi.fetchSymbolScan(q);
      if (remote.isNotEmpty) {
        final hit = remote.first;
        if (!mounted) return;
        _showScanDetail(context, hit, () => _analyzeWithCopilot(hit));
        return;
      }
    } catch (_) {}

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('No direct match for $query. Run a scan to discover more.')),
    );
  }

  Widget _heroBanner(
    BuildContext context, {
    required String title,
    required String subtitle,
    required String badge,
    required Widget trailing,
    required Widget child,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [tone(brandPrimary, 0.12), tone(brandDeep, 0.9)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: tone(Colors.white, 0.08)),
        boxShadow: [
          BoxShadow(
            color: tone(Colors.black, 0.35),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.08),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: tone(Colors.white, 0.1)),
                ),
                child: Text(
                  badge,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              const Spacer(),
              trailing,
            ],
          ),
          const SizedBox(height: 12),
          Text(
            title,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }

  Widget _onboardingCard() {
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Welcome to technic',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: () => setState(() => _showOnboarding = false),
                ),
              ],
            ),
            const SizedBox(height: 8),
            const Text(
              'Pick your style and focus. We?ll personalize defaults and run your first scan.',
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: 'Trading style'),
              initialValue: _tradeStyle,
              items: const [
                'Short-term swing',
                'Weekly',
                'Multi-day',
                'Momentum',
              ].map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
              onChanged: (v) => setState(() => _tradeStyle = v ?? _tradeStyle),
            ),
            const SizedBox(height: 8),
            _sliderRow(
              label: 'Comfort with risk (min TechRating)',
              value: _minTechRating,
              min: 0,
              max: 50,
              divisions: 10,
              formatter: (v) => v.round().toString(),
              onChanged: (v) => setState(() => _minTechRating = v),
            ),
            const SizedBox(height: 4),
            const Text('Favorite sectors'),
            Wrap(
              spacing: 8,
              runSpacing: 6,
              children: _sectors
                  .map(
                    (s) => FilterChip(
                      label: Text(s),
                      selected: _selectedSectors.contains(s),
                      onSelected: (v) => setState(() {
                        if (v) {
                          _selectedSectors.add(s);
                        } else {
                          _selectedSectors.remove(s);
                        }
                      }),
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                ElevatedButton.icon(
                  onPressed: () {
                    setState(() => _showOnboarding = false);
                    _refresh();
                  },
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Apply & run'),
                ),
                const SizedBox(width: 8),
                OutlinedButton.icon(
                  onPressed: () {
                    setState(() => _showOnboarding = false);
                    _randomizeScan();
                  },
                  icon: const Icon(Icons.shuffle),
                  label: const Text('Randomize scan'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class IdeasPage extends StatefulWidget {
  const IdeasPage({super.key});

  @override
  State<IdeasPage> createState() => _IdeasPageState();
}

class _IdeasPageState extends State<IdeasPage> with AutomaticKeepAliveClientMixin {
  late Future<List<Idea>> _ideasFuture;

  @override
  void initState() {
    super.initState();
    _ideasFuture = _loadIdeasFromLastScan();
  }

  @override
  bool get wantKeepAlive => true;

  Future<void> _refresh() async {
    setState(() {
      _ideasFuture = _loadIdeasFromLastScan();
    });
    await _ideasFuture;
  }

  Future<List<Idea>> _loadIdeasFromLastScan() async {
    // Derive ideas from last scan results; if empty, try pulling from API as fallback
    final scans = lastScanResults.value;
    if (scans.isNotEmpty) {
      return scans.map((s) {
        final plan = 'Entry ${s.entry.isNotEmpty ? s.entry : "-"}, Stop ${s.stop.isNotEmpty ? s.stop : "-"}, Target ${s.target.isNotEmpty ? s.target : "-"}';
        final why = '${s.signal} setup based on blended trend, momentum, volume, and risk scores.';
        return Idea(s.signal, s.ticker, why, plan, s.sparkline);
      }).toList();
    }
    try {
      return await technicApi.fetchIdeas();
    } catch (_) {
      return mockIdeas;
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return RefreshIndicator(
      onRefresh: _refresh,
      child: FutureBuilder<List<Idea>>(
        future: _ideasFuture,
        builder: (context, snapshot) {
          final ideas = snapshot.data ?? mockIdeas;
          final loading = snapshot.connectionState == ConnectionState.waiting;
          final hasError = snapshot.hasError;
          final hasIdeas = ideas.isNotEmpty;

          return ListView(
            children: [
              _heroBanner(
                context,
                title: 'Personal Quant ideas',
                subtitle:
                    'Card stack with sparkline previews and quick execute hooks.',
                badge: 'Live feed',
                trailing: OutlinedButton.icon(
                  onPressed: _refresh,
                  icon: const Icon(Icons.filter_alt),
                  label: const Text('Filter ideas'),
                ),
                child: Row(
                  children: const [
                    _PulseBadge(label: 'Copilot ready'),
                    SizedBox(width: 8),
                    _PulseBadge(label: 'Time horizons mixed'),
                    SizedBox(width: 8),
                    _PulseBadge(label: 'Risk tuned to 1%'),
                  ],
                ),
              ),
              if (hasError) ...[
                const SizedBox(height: 10),
                _infoCard(
                  title: 'Ideas feed unavailable',
                  subtitle: snapshot.error.toString(),
                  child: TextButton(
                    onPressed: _refresh,
                    child: const Text('Retry'),
                  ),
                ),
              ],
              if (loading) ...[
                const SizedBox(height: 12),
                _infoCard(
                  title: 'Fetching ideas...',
                  subtitle: 'Pulling from Streamlit feed',
                  child: const LinearProgressIndicator(
                    minHeight: 4,
                    color: brandPrimary,
                    backgroundColor: Colors.white24,
                  ),
                ),
              ],
              const SizedBox(height: 16),
              ValueListenableBuilder<String?>(
                valueListenable: copilotStatus,
                builder: (context, status, _) {
                  if (status == null) return const SizedBox.shrink();
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: _infoCard(
                      title: 'Copilot offline',
                      subtitle:
                          'Cached guidance will display until service recovers.',
                      child: Row(
                        children: [
                          Expanded(
                            child: Text(
                              status,
                              style: const TextStyle(color: Colors.white70),
                            ),
                          ),
                          const SizedBox(width: 12),
                          OutlinedButton(
                            onPressed: () {
                              copilotPrefill.value =
                                  'Summarize these trade ideas when back online';
                              _shellKey.currentState?.setTab(2);
                            },
                            child: const Text('Open Copilot'),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
              if (!hasIdeas)
                _infoCard(
                  title: 'No ideas yet',
                  subtitle: 'Adjust filters or refresh to load ideas.',
                  child: ElevatedButton.icon(
                    onPressed: _refresh,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Refresh ideas'),
                  ),
                )
              else
                ...ideas.map((i) => _ideaCard(i, () => _analyzeIdeaWithCopilot(i))),
            ],
          );
        },
      ),
    );
  }

  Future<void> _analyzeIdeaWithCopilot(Idea idea) async {
    final prompt = StringBuffer()
      ..writeln("Explain why ${idea.ticker} is selected and the trade plan.")
      ..writeln("Signal: ${idea.title}. Plan: ${idea.plan}. Meta: ${idea.meta}.")
      ..writeln("Give a concise recommendation in plain language.");
    if (idea.option != null) {
      prompt.writeln(
          "Option idea: ${idea.option?["contract_type"]} ${idea.option?["strike"]} exp ${idea.option?["expiration"]}, delta ${idea.option?["delta"]}.");
    }
    try {
      final reply = await technicApi.sendCopilot(prompt.toString());
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(reply.body, maxLines: 4, overflow: TextOverflow.ellipsis)),
      );
    } catch (e) {
      if (!mounted) return;
      copilotStatus.value = e.toString();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Copilot unavailable: $e')),
      );
    }
  }
}

class CopilotPage extends StatefulWidget {
  const CopilotPage({super.key});

  @override
  State<CopilotPage> createState() => _CopilotPageState();
}

class _CopilotPageState extends State<CopilotPage> with AutomaticKeepAliveClientMixin {
  final TextEditingController _controller = TextEditingController();
  final List<CopilotMessage> _messages = List.of(copilotMessages);
  bool _sending = false;
  bool _copilotError = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  bool get wantKeepAlive => true;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final prefill = copilotPrefill.value;
    if (prefill != null && _controller.text.isEmpty) {
      _controller.text = prefill;
      copilotPrefill.value = null;
    }
  }

  Future<void> _sendPrompt([String? prompt]) async {
    final text = prompt ?? _controller.text.trim();
    if (text.isEmpty || _sending) return;
    setState(() {
      _sending = true;
      _messages.add(CopilotMessage('user', text));
      _controller.clear();
    });

    try {
      final reply = await technicApi.sendCopilot(text);
      if (!mounted) return;
      setState(() {
        _messages.add(reply);
        _sending = false;
        _copilotError = false;
      });
      copilotStatus.value = null;
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _sending = false;
        _copilotError = true;
      });
      copilotStatus.value = e.toString();
      _messages.add(
        const CopilotMessage(
          'assistant',
          'Copilot is temporarily offline. Showing cached guidance instead.',
        ),
      );
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Copilot unavailable: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return ListView(
      children: [
        ValueListenableBuilder<String?>(
          valueListenable: copilotStatus,
          builder: (context, status, _) {
            if (status == null) return const SizedBox.shrink();
            return _infoCard(
              title: 'Copilot offline',
              subtitle: 'Cached guidance will display until service recovers.',
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      status,
                      style: const TextStyle(color: Colors.white70),
                    ),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton(
                    onPressed: () {
                      copilotPrefill.value = 'Retry the last question';
                      _shellKey.currentState?.setTab(2);
                    },
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          },
        ),
        _heroBanner(
          context,
          title: 'Quant Copilot',
          subtitle: 'Context-aware chat with structured answers.',
          badge: 'Conversational',
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              OutlinedButton.icon(
                onPressed: _sending ? null : () => _sendPrompt('Voice request'),
                icon: const Icon(Icons.mic_none),
                label: const Text('Voice'),
              ),
              const SizedBox(width: 8),
              TextButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.note_alt_outlined),
                label: const Text('Notes'),
              ),
            ],
          ),
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: copilotPrompts
                  .map(
                    (p) => Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ActionChip(
                        backgroundColor: tone(Colors.white, 0.04),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                          side: BorderSide(color: tone(Colors.white, 0.06)),
                        ),
                        label:
                            Text(p, style: const TextStyle(color: Colors.white)),
                        onPressed: () => _sendPrompt(p),
                      ),
                    ),
                  )
                  .toList(),
            ),
          ),
        ),
        const SizedBox(height: 12),
        ValueListenableBuilder<ScanResult?>(
          valueListenable: copilotContext,
          builder: (context, ctx, _) {
            if (ctx == null) return const SizedBox.shrink();
            return Card(
              margin: const EdgeInsets.symmetric(horizontal: 0, vertical: 8),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              ctx.ticker,
                              style: const TextStyle(
                                  fontSize: 18, fontWeight: FontWeight.w800),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              '${ctx.signal} • ${ctx.playStyle ?? "Swing"}',
                              style: const TextStyle(fontSize: 12, color: Colors.white70),
                            ),
                            if (ctx.institutionalCoreScore != null)
                              Text(
                                'ICS ${ctx.institutionalCoreScore!.toStringAsFixed(0)}/100'
                                '${ctx.icsTier != null ? " (${ctx.icsTier})" : ""}',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: tone(brandPrimary, 0.9),
                                ),
                              ),
                          ],
                        ),
                        IconButton(
                          icon: const Icon(Icons.clear),
                          onPressed: () => copilotContext.value = null,
                          tooltip: 'Clear context',
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Win ~${ctx.winProb10d != null ? (ctx.winProb10d! * 100).toStringAsFixed(0) : "--"}% • '
                      'Quality ${ctx.qualityScore?.toStringAsFixed(1) ?? "--"} • '
                      'ATR ${(ctx.atrPct != null ? (ctx.atrPct! * 100).toStringAsFixed(1) : "--")}%',
                      style: const TextStyle(fontSize: 11, color: Colors.white70),
                    ),
                    if ((ctx.eventSummary ?? ctx.eventFlags ?? '').isNotEmpty) ...[
                      const SizedBox(height: 4),
                      Text(
                        ctx.eventSummary ?? ctx.eventFlags ?? '',
                        style: TextStyle(fontSize: 11, color: tone(brandAccent, 0.9)),
                      ),
                    ],
                  ],
                ),
              ),
            );
          },
        ),
        const SizedBox(height: 16),
        if (_copilotError)
          _infoCard(
            title: 'Copilot unavailable',
            subtitle: 'Check your connection or retry shortly.',
            child: TextButton.icon(
              onPressed: _sending ? null : () => _sendPrompt(),
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ),
        if (_messages.isEmpty && !_copilotError)
          _infoCard(
            title: 'Start a Copilot session',
            subtitle: 'Ask a question to begin. Voice coming soon.',
            child: const Text(
              'Examples: "Summarize today\'s scan", "Explain risk on NVDA setup", "Compare TSLA vs AAPL momentum".',
              style: TextStyle(color: Colors.white70),
            ),
          ),
        _infoCard(
          title: 'Conversation',
          subtitle:
              'Persist responses with tables, bullets, and calls to action.',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ..._messages.map((m) => _messageBubble(m)),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.02),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: tone(Colors.white, 0.05)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    ValueListenableBuilder<String?>(
                      valueListenable: copilotPrefill,
                      builder: (context, suggestion, _) {
                        if (suggestion == null || suggestion.isEmpty) {
                          return const SizedBox.shrink();
                        }
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Row(
                            children: [
                              Expanded(
                                child: Text(
                                  suggestion,
                                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                                ),
                              ),
                              TextButton(
                                onPressed: () {
                                  _controller.text = suggestion;
                                  copilotPrefill.value = null;
                                },
                                child: const Text('Use suggestion'),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                    TextField(
                      controller: _controller,
                      maxLines: 4,
                      minLines: 2,
                      decoration: InputDecoration(
                        hintText: 'Type your question...',
                        border: InputBorder.none,
                        suffixIcon: _sending
                            ? const Padding(
                                padding: EdgeInsets.all(10),
                                child: SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                ),
                              )
                            : IconButton(
                                icon: const Icon(Icons.send),
                                onPressed: _sending ? null : () => _sendPrompt(),
                              ),
                      ),
                      onSubmitted: (_) => _sendPrompt(),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Copilot explains what Technic sees in this setup and outlines an example trade. '
                      'Responses are educational, not financial advice.',
                      style: TextStyle(color: tone(Colors.white, 0.6), fontSize: 12),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.memory,
                              size: 14,
                              color: tone(Colors.white, 0.6),
                            ),
                            const SizedBox(width: 6),
                            Text(
                              'Session memory on',
                              style: TextStyle(color: tone(Colors.white, 0.7)),
                            ),
                          ],
                        ),
                        ElevatedButton.icon(
                          onPressed: _sending ? null : _sendPrompt,
                          icon: _sending
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Icon(Icons.send),
                          label:
                              Text(_sending ? 'Sending...' : 'Send to Copilot'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      children: [
        ValueListenableBuilder<String?>(
          valueListenable: userId,
          builder: (context, uid, _) {
            return _infoCard(
              title: uid == null ? 'Sign in to sync' : 'Signed in as $uid',
              subtitle: 'Sync presets, streaks, and preferences across devices.',
              child: Row(
                children: [
                  ElevatedButton.icon(
                    onPressed: () async {
                      final messenger = ScaffoldMessenger.of(context);
                      userId.value = 'google_user';
                      await LocalStore.saveUser(userId.value!);
                      messenger.showSnackBar(
                        const SnackBar(content: Text('Signed in with Google (stub)')),
                      );
                    },
                    icon: const Icon(Icons.login),
                    label: const Text('Google'),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: () async {
                      final messenger = ScaffoldMessenger.of(context);
                      userId.value = 'apple_user';
                      await LocalStore.saveUser(userId.value!);
                      messenger.showSnackBar(
                        const SnackBar(content: Text('Signed in with Apple (stub)')),
                      );
                    },
                    icon: const Icon(Icons.apple),
                    label: const Text('Apple'),
                  ),
                  const SizedBox(width: 8),
                  if (uid != null)
                    TextButton(
                      onPressed: () async {
                        final messenger = ScaffoldMessenger.of(context);
                        userId.value = null;
                        final prefs = await SharedPreferences.getInstance();
                        await prefs.remove('user_id');
                        messenger.showSnackBar(
                          const SnackBar(content: Text('Signed out')),
                        );
                      },
                      child: const Text('Sign out'),
                    ),
                ],
              ),
            );
          },
        ),
        ValueListenableBuilder<String?>(
          valueListenable: copilotStatus,
          builder: (context, status, _) {
            if (status == null) return const SizedBox.shrink();
            return _infoCard(
              title: 'Copilot offline',
              subtitle:
                  'Cached guidance will display until the service recovers.',
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      status,
                      style: const TextStyle(color: Colors.white70),
                    ),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton(
                    onPressed: () {
                      copilotPrefill.value = 'Retry Copilot with the last question.';
                      _shellKey.currentState?.setTab(2);
                    },
                    child: const Text('Open Copilot'),
                  ),
                ],
              ),
            );
          },
        ),
        _heroBanner(
          context,
          title: 'Profile and preferences',
          subtitle: 'Preserve every setting across devices.',
          badge: 'Synced',
          trailing: TextButton.icon(
            onPressed: () {},
            icon: const Icon(Icons.edit_outlined),
            label: const Text('Edit profile'),
          ),
          child: Row(
            children: const [
              _PulseBadge(label: 'Advanced view on'),
              SizedBox(width: 8),
              _PulseBadge(label: 'Alerts enabled'),
              SizedBox(width: 8),
              _PulseBadge(label: 'Sessions persist'),
            ],
          ),
        ),
        const SizedBox(height: 16),
        SectionHeader('Profile', caption: 'Mode, risk, and universe'),
        _infoCard(
          title: 'Account',
          subtitle: 'Connect your profile and preferences',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 42,
                    height: 42,
                    decoration: BoxDecoration(
                      color: tone(brandPrimary, 0.12),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.person_outline, color: Colors.white70),
                  ),
                  const SizedBox(width: 10),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text('Primary workspace',
                          style: TextStyle(fontWeight: FontWeight.w700)),
                      Text('Synced across devices',
                          style: TextStyle(color: Colors.white70, fontSize: 12)),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const _ProfileRow(label: 'Mode', value: 'Swing / Long-term'),
              const _ProfileRow(label: 'Risk per trade', value: '1.0%'),
              const _ProfileRow(label: 'Universe', value: 'US Equities'),
              const _ProfileRow(label: 'Experience', value: 'Advanced'),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 6,
                children: const [
                  _PulseBadge(label: 'Dark mode'),
                  _PulseBadge(label: 'Advanced view'),
                  _PulseBadge(label: 'Session memory'),
                ],
              ),
              const SizedBox(height: 8),
              const Text(
                'Data sources: Polygon/rest API; Copilot: OpenAI.',
                style: TextStyle(color: Colors.white70, fontSize: 12),
              ),
              const SizedBox(height: 4),
              const Text(
                'Your API keys are stored locally (not uploaded).',
                style: TextStyle(color: Colors.white70, fontSize: 12),
              ),
            ],
          ),
        ),
        SectionHeader('Appearance', caption: 'Dark mode with Technic accent'),
        _infoCard(
          title: 'Theme',
          subtitle: 'Institutional minimal with optional high contrast',
          child: Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  gradient: const LinearGradient(
                    colors: [Color(0xFFB6FF3B), Color(0xFF5EEAD4)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              const Text('Technic Dark', style: TextStyle(color: Colors.white)),
              const Spacer(),
              Row(
                children: [
                  const Text('Dark mode', style: TextStyle(color: Colors.white70)),
                  const SizedBox(width: 6),
                  Switch(
                    value: themeIsDark.value,
                    onChanged: (v) {
                      themeIsDark.value = v;
                    },
                    thumbColor: WidgetStatePropertyAll(brandPrimary),
                    trackColor: WidgetStatePropertyAll(tone(brandPrimary, 0.3)),
                  ),
                ],
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.06),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Text(
                  'Sync across devices',
                  style: TextStyle(color: Colors.white70),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        _infoCard(
          title: 'Display options',
          subtitle: 'Toggle modes and accessibility presets',
          child: Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              Chip(
                label: const Text('Dark mode'),
                avatar: const Icon(Icons.dark_mode, size: 16, color: Colors.white70),
                backgroundColor: tone(Colors.white, 0.05),
              ),
              Chip(
                label: const Text('Light mode'),
                avatar:
                    const Icon(Icons.light_mode_outlined, size: 16, color: Colors.white70),
                backgroundColor: tone(brandPrimary, Theme.of(context).brightness == Brightness.dark ? 0.05 : 0.12),
              ),
              Chip(
                label: const Text('High contrast'),
                avatar: const Icon(Icons.contrast, size: 16, color: Colors.white70),
                backgroundColor: tone(Colors.white, 0.05),
              ),
            ],
          ),
        ),
        SectionHeader('Data & alerts', caption: 'Control intensity'),
        _infoCard(
          title: 'Notifications',
          subtitle: 'Goal progress, alerts, and data refresh cadence',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const _ProfileRow(label: 'Goal tracking', value: 'On'),
              const _ProfileRow(label: 'Scanner refresh', value: 'Every 60s'),
              const _ProfileRow(label: 'Haptics', value: 'Subtle'),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 6,
                children: [
                  ActionChip(
                    label: const Text('Mute alerts'),
                    onPressed: () {},
                    backgroundColor: tone(Colors.white, 0.05),
                  ),
                  ActionChip(
                    label: const Text('Set refresh to 30s'),
                    onPressed: () {},
                    backgroundColor: tone(Colors.white, 0.05),
                  ),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        _infoCard(
          title: 'Data & trust',
          subtitle: 'How scores and keys are handled',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const [
              Text('Scores: trend, momentum, volatility, and risk signals combined.'),
              SizedBox(height: 4),
              Text('Data sources: Polygon/rest API; Copilot: OpenAI.'),
              SizedBox(height: 4),
              Text('Your API keys are stored locally (not uploaded).'),
            ],
          ),
        ),
        const SizedBox(height: 12),
        FutureBuilder<Map<String, dynamic>?>(
          future: LocalStore.loadScannerState(),
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const LinearProgressIndicator(minHeight: 2);
            }
            final data = snapshot.data;
            if (data == null) return const SizedBox.shrink();
            final scanCount = data['scanCount'] as int? ?? 0;
            final streak = data['streakDays'] as int? ?? 0;
            final savedPresets =
                (data['saved_screens'] as List<SavedScreen>?)?.length ?? 0;
            final filters = Map<String, String>.from(data['filters'] as Map);
            final sectors = (filters['sectors'] ?? '')
                .split(',')
                .where((e) => e.trim().isNotEmpty)
                .toList();
            final lastScanStr = data['lastScan'] as String?;
            final lastScan =
                lastScanStr != null ? DateTime.tryParse(lastScanStr) : null;
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _infoCard(
                  title: 'Your month in technic',
                  subtitle: 'Activity recap',
                  child: Row(
                    children: [
                      Chip(
                        label: Text('Scans: $scanCount'),
                        backgroundColor: tone(Colors.white, 0.05),
                      ),
                      const SizedBox(width: 8),
                      Chip(
                        label: Text('Streak: $streak d'),
                        backgroundColor: tone(brandPrimary, 0.15),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          sectors.isEmpty
                              ? 'Top sector: All'
                              : 'Top sectors: ${sectors.join(', ')}',
                          style: const TextStyle(color: Colors.white70),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Chip(
                    label: Text('Presets: $savedPresets'),
                    backgroundColor: tone(Colors.white, 0.05),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                if (lastScan != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    child: Text(
                      'Last scan: ${lastScan.toLocal().toString().split('.').first} ? ${DateTime.now().difference(lastScan).inDays}d ago',
                      style: const TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                  ),
                if (lastScan != null) const SizedBox(height: 8),
                _infoCard(
                  title: 'Achievements',
                  subtitle: 'Celebrate streaks and progress',
                  child: Wrap(
                spacing: 8,
                runSpacing: 6,
                children: [
                  Chip(
                    label: Text(
                        scanCount >= 5 ? 'Starter: 5 scans' : 'Next: 5 scans'),
                    backgroundColor: scanCount >= 5
                        ? tone(brandPrimary, 0.2)
                        : tone(Colors.white, 0.05),
                  ),
                  Chip(
                    label: Text(
                        scanCount >= 10 ? 'Builder: 10 scans' : 'Next: 10 scans'),
                    backgroundColor: scanCount >= 10
                        ? tone(brandPrimary, 0.2)
                        : tone(Colors.white, 0.05),
                  ),
                  Chip(
                    label: Text(
                        streak >= 3 ? 'Streak 3 days' : 'Keep a 3-day streak'),
                    backgroundColor: streak >= 3
                        ? tone(brandPrimary, 0.2)
                        : tone(Colors.white, 0.05),
                  ),
                  Chip(
                    label: Text(
                        streak >= 7 ? 'Streak 7 days' : 'Keep a 7-day streak'),
                    backgroundColor: streak >= 7
                        ? tone(brandPrimary, 0.2)
                        : tone(Colors.white, 0.05),
                  ),
                  Chip(
                    label: Text(savedPresets >= 3
                        ? 'Preset pro: 3 saved'
                        : 'Next: save 3 presets'),
                    backgroundColor: savedPresets >= 3
                        ? tone(brandPrimary, 0.2)
                        : tone(Colors.white, 0.05),
                  ),
                ],
              ),
            ),
              ],
            );
          },
        ),
        const SizedBox(height: 12),
        ValueListenableBuilder<String?>(
          valueListenable: copilotStatus,
          builder: (context, status, _) {
            return _infoCard(
              title: 'Copilot status',
              subtitle: status == null
                  ? 'Online. Answers will return live.'
                  : 'Offline. Showing cached guidance until service recovers.',
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Flexible(
                    child: Text(
                      status ?? 'All systems go.',
                      style: const TextStyle(color: Colors.white70),
                    ),
                  ),
                  const SizedBox(width: 12),
                  ElevatedButton.icon(
                    onPressed: () {
                      copilotPrefill.value = 'Check Copilot status';
                      _shellKey.currentState?.setTab(2);
                    },
                    icon: const Icon(Icons.chat_bubble_outline),
                    label: const Text('Open Copilot'),
                  ),
                ],
              ),
            );
          },
        ),
      ],
    );
  }
}

class MarketMover {
  final String ticker;
  final String delta;
  final String note;
  final bool isPositive;
  final List<double> sparkline;
  const MarketMover(
    this.ticker,
    this.delta,
    this.note,
    this.isPositive, [
    this.sparkline = const [],
  ]);

  factory MarketMover.fromJson(Map<String, dynamic> json) {
    final deltaRaw = json['delta']?.toString() ?? '';
    final delta = deltaRaw.isEmpty
        ? ''
        : deltaRaw.startsWith('+') || deltaRaw.startsWith('-')
        ? deltaRaw
        : '+$deltaRaw';
    final isPositiveField = json['isPositive'] ?? json['is_positive'];
    final isPos = isPositiveField is bool
        ? isPositiveField
        : delta.startsWith('+') ||
              (json['change']?.toString().startsWith('+') ?? false);
    final rawSpark = json['sparkline'] ?? json['spark'] ?? [];
    final spark = rawSpark is List
        ? rawSpark.map((e) => double.tryParse(e.toString()) ?? 0).toList()
        : <double>[];
    return MarketMover(
      json['ticker']?.toString() ?? '',
      delta,
      json['note']?.toString() ?? json['label']?.toString() ?? '',
      isPos,
      spark,
    );
  }

  Map<String, dynamic> toJson() => {
        'ticker': ticker,
        'delta': delta,
        'note': note,
        'isPositive': isPositive,
        'sparkline': sparkline,
      };
}

class UniverseStats {
  final int total;
  final Map<String, int> sectors;
  final Map<String, int> subindustries;
  const UniverseStats(this.total, this.sectors, this.subindustries);

  factory UniverseStats.fromJson(Map<String, dynamic> json) {
    final total = json['total'] is num ? (json['total'] as num).toInt() : 0;
    final sectorsRaw = json['sectors'] as List<dynamic>? ?? [];
    final sectors = <String, int>{};
    for (final s in sectorsRaw) {
      if (s is Map<String, dynamic>) {
        final name = s['name']?.toString() ?? '';
        final count = s['count'] is num ? (s['count'] as num).toInt() : 0;
        if (name.isNotEmpty) sectors[name] = count;
      }
    }
    final subsRaw = json['subindustries'] as List<dynamic>? ?? [];
    final subs = <String, int>{};
    for (final s in subsRaw) {
      if (s is Map<String, dynamic>) {
        final name = s['name']?.toString() ?? '';
        final count = s['count'] is num ? (s['count'] as num).toInt() : 0;
        if (name.isNotEmpty) subs[name] = count;
      }
    }
    return UniverseStats(total, sectors, subs);
  }
}

class ScanResult {
  final String ticker;
  final String signal;
  final String rrr;
  final String entry;
  final String stop;
  final String target;
  final String note;
  final List<double> sparkline;
  final double? institutionalCoreScore;
  final String? icsTier;
  final double? winProb10d;
  final double? qualityScore;
  final String? playStyle;
  final bool? isUltraRisky;
  final String? profileName;
  final String? profileLabel;
  final String? sector;
  final String? industry;
  final double? techRating;
  final double? alphaScore;
  final double? atrPct;
  final String? eventSummary;
  final String? eventFlags;
  final String? fundamentalSnapshot;
  final List<dynamic>? optionStrategies;
  const ScanResult(
    this.ticker,
    this.signal,
    this.rrr,
    this.entry,
    this.stop,
    this.target,
    this.note, [
    this.sparkline = const [],
    this.institutionalCoreScore,
    this.icsTier,
    this.winProb10d,
    this.qualityScore,
    this.playStyle,
    this.isUltraRisky,
    this.profileName,
    this.profileLabel,
    this.sector,
    this.industry,
    this.techRating,
    this.alphaScore,
    this.atrPct,
    this.eventSummary,
    this.eventFlags,
    this.fundamentalSnapshot,
    this.optionStrategies,
  ]);

  factory ScanResult.fromJson(Map<String, dynamic> json) {
    String num(dynamic v) => v == null ? '' : v.toString();
    double? dbl(dynamic v) => v == null ? null : double.tryParse(v.toString());
    bool? bl(dynamic v) {
      if (v == null) return null;
      if (v is bool) return v;
      final s = v.toString().toLowerCase();
      return s == 'true' || s == '1';
    }
    final rawSpark = json['sparkline'] ?? json['spark'] ?? [];
    final spark = rawSpark is List
        ? rawSpark.map((e) => double.tryParse(e.toString()) ?? 0).toList()
        : <double>[];
    return ScanResult(
      json['ticker']?.toString() ?? '',
      json['signal']?.toString() ?? '',
      json['rrr']?.toString() ?? json['rr']?.toString() ?? '',
      num(json['entry']),
      num(json['stop']),
      num(json['target']),
      json['note']?.toString() ?? '',
      spark,
      dbl(json['InstitutionalCoreScore'] ?? json['ics'] ?? json['ICS']),
      json['ICS_Tier']?.toString() ?? json['Tier']?.toString(),
      dbl(json['win_prob_10d']),
      dbl(json['QualityScore'] ?? json['fundamental_quality_score']),
      json['PlayStyle']?.toString(),
      bl(json['IsUltraRisky']),
      json['Profile']?.toString(),
      json['ProfileLabel']?.toString(),
      json['Sector']?.toString(),
      json['Industry']?.toString(),
      dbl(json['TechRating']),
      dbl(json['AlphaScore']),
      dbl(json['ATR14_pct']),
      json['EventSummary']?.toString(),
      json['EventFlags']?.toString(),
      json['FundamentalSnapshot']?.toString(),
      json['OptionStrategies'] as List<dynamic>?,
    );
  }

  Map<String, dynamic> toJson() => {
        'ticker': ticker,
        'signal': signal,
        'rrr': rrr,
        'entry': entry,
        'stop': stop,
        'target': target,
        'note': note,
        'sparkline': sparkline,
        'InstitutionalCoreScore': institutionalCoreScore,
        'ICS_Tier': icsTier,
        'win_prob_10d': winProb10d,
        'QualityScore': qualityScore,
        'PlayStyle': playStyle,
        'IsUltraRisky': isUltraRisky,
        'Profile': profileName,
        'ProfileLabel': profileLabel,
        'Sector': sector,
        'Industry': industry,
        'TechRating': techRating,
        'AlphaScore': alphaScore,
        'ATR14_pct': atrPct,
        'EventSummary': eventSummary,
        'EventFlags': eventFlags,
        'FundamentalSnapshot': fundamentalSnapshot,
        'OptionStrategies': optionStrategies,
      };
}

class Idea {
  final String title;
  final String ticker;
  final String meta;
  final String plan;
  final List<double> sparkline;
  final Map<String, dynamic>? option;
  const Idea(this.title, this.ticker, this.meta, this.plan, this.sparkline, {this.option});

  factory Idea.fromJson(Map<String, dynamic> json) {
    final rawSpark = json['sparkline'] ?? json['spark'] ?? [];
    final spark = rawSpark is List
        ? rawSpark.map((e) => double.tryParse(e.toString()) ?? 0).toList()
        : <double>[];
    return Idea(
      json['title']?.toString() ?? '',
      json['ticker']?.toString() ?? '',
      json['meta']?.toString() ?? '',
      json['plan']?.toString() ?? '',
      spark,
      option: json['option'] as Map<String, dynamic>?,
    );
  }
}

class QuickAction {
  final IconData icon;
  final String label;
  final String hint;
  const QuickAction(this.icon, this.label, this.hint);
}

class SavedScreen {
  final String name;
  final String description;
  final String horizon;
  final bool isActive;
  final Map<String, String>? params;
  const SavedScreen(
    this.name,
    this.description,
    this.horizon,
    this.isActive, {
    this.params,
  });

  Map<String, dynamic> toJson() => {
        'name': name,
        'description': description,
        'horizon': horizon,
        'isActive': isActive,
        'params': params,
      };

  factory SavedScreen.fromJson(Map<String, dynamic> json) => SavedScreen(
        json['name']?.toString() ?? '',
        json['description']?.toString() ?? '',
        json['horizon']?.toString() ?? '',
        json['isActive'] == true,
        params: (json['params'] as Map?)?.map(
              (k, v) => MapEntry(k.toString(), v.toString()),
            ) ??
            const {},
      );

  SavedScreen copyWith({
    String? name,
    String? description,
    String? horizon,
    bool? isActive,
    Map<String, String>? params,
  }) {
    return SavedScreen(
      name ?? this.name,
      description ?? this.description,
      horizon ?? this.horizon,
      isActive ?? this.isActive,
      params: params ?? this.params,
    );
  }
}

class ScoreboardSlice {
  final String label;
  final String pnl;
  final String winRate;
  final String horizon;
  final Color accent;
  const ScoreboardSlice(
    this.label,
    this.pnl,
    this.winRate,
    this.horizon,
    this.accent,
  );

  factory ScoreboardSlice.fromJson(Map<String, dynamic> json) {
    final accentStr = json['accent']?.toString();
    return ScoreboardSlice(
      json['label']?.toString() ?? '',
      json['pnl']?.toString() ?? '',
      json['winRate']?.toString() ?? json['win_rate']?.toString() ?? '',
      json['horizon']?.toString() ?? '',
      accentStr != null ? _colorFromHex(accentStr) : brandPrimary,
    );
  }
}

class CopilotMessage {
  final String role;
  final String body;
  final String? meta;
  const CopilotMessage(this.role, this.body, {this.meta});

  factory CopilotMessage.fromJson(Map<String, dynamic> json) {
    return CopilotMessage(
      json['role']?.toString() ?? 'assistant',
      json['body']?.toString() ?? json['message']?.toString() ?? '',
      meta: json['meta']?.toString(),
    );
  }
}

class ScannerBundle {
  final List<MarketMover> movers;
  final List<ScanResult> scanResults;
  final List<ScoreboardSlice> scoreboard;
  final String? progress;
  const ScannerBundle({
    required this.movers,
    required this.scanResults,
    required this.scoreboard,
    this.progress,
  });
}

class ApiConfig {
  final String baseUrl;
  final String moversPath;
  final String scanPath;
  final String ideasPath;
  final String scoreboardPath;
  final String copilotPath;
  final String universeStatsPath;
  final String symbolPath;

  const ApiConfig({
    required this.baseUrl,
    required this.moversPath,
    required this.scanPath,
    required this.ideasPath,
    required this.scoreboardPath,
    required this.copilotPath,
    required this.universeStatsPath,
    required this.symbolPath,
  });

  factory ApiConfig.fromEnv() {
    final rawBase = const String.fromEnvironment(
      'TECHNIC_API_BASE',
      defaultValue: 'http://localhost:8502',
    );
    final normalizedBase = _normalizeBaseForPlatform(rawBase);
    // Default Copilot path; can be overridden by dart-define if needed
    return ApiConfig(
      baseUrl: normalizedBase,
      moversPath: const String.fromEnvironment(
        'TECHNIC_API_MOVERS',
        defaultValue: '/api/movers',
      ),
      scanPath: const String.fromEnvironment(
        'TECHNIC_API_SCANNER',
        defaultValue: '/api/scanner',
      ),
      ideasPath: const String.fromEnvironment(
        'TECHNIC_API_IDEAS',
        defaultValue: '/api/ideas',
      ),
      scoreboardPath: const String.fromEnvironment(
        'TECHNIC_API_SCOREBOARD',
        defaultValue: '/api/scoreboard',
      ),
      copilotPath: const String.fromEnvironment(
        'TECHNIC_API_COPILOT',
        defaultValue: '/api/copilot',
      ),
      universeStatsPath: const String.fromEnvironment(
        'TECHNIC_API_UNIVERSE',
        defaultValue: '/api/universe_stats',
      ),
      symbolPath: const String.fromEnvironment(
        'TECHNIC_API_SYMBOL',
        defaultValue: '/api/symbol',
      ),
    );
  }

  Uri _uri(String path) {
    if (path.startsWith('http')) return Uri.parse(path);
    return Uri.parse('$baseUrl$path');
  }

  Uri moversUri() => _uri(moversPath);
  Uri scanUri() => _uri(scanPath);
  Uri ideasUri() => _uri(ideasPath);
  Uri scoreboardUri() => _uri(scoreboardPath);
  Uri copilotUri() => _uri(copilotPath);
  Uri universeStatsUri() => _uri(universeStatsPath);
  Uri symbolUri() => _uri(symbolPath);
}

String _normalizeBaseForPlatform(String base) {
  final isAndroid = !kIsWeb && defaultTargetPlatform == TargetPlatform.android;
  if (isAndroid) {
    const localHosts = {
      'http://localhost:8501',
      'http://127.0.0.1:8501',
      'http://localhost:8502',
      'http://127.0.0.1:8502',
    };
    if (localHosts.contains(base)) {
      final port = base.split(':').last;
      return 'http://10.0.2.2:$port';
    }
  }
  return base;
}

class ScanResultsPayload {
  final List<ScanResult> results;
  final String? progress;
  const ScanResultsPayload(this.results, this.progress);
}

class TechnicApi {
  TechnicApi({http.Client? client, ApiConfig? config})
    : _client = client ?? http.Client(),
      _config = config ?? ApiConfig.fromEnv();

  final http.Client _client;
  final ApiConfig _config;

  Future<ScannerBundle> fetchScannerBundle({Map<String, String>? params}) async {
    final results = await Future.wait([
      _safe(() => fetchMovers(params: params), mockMovers),
      _safe(() => fetchScanResults(params: params), ScanResultsPayload(mockScanResults, null)),
      _safe(() => fetchScoreboard(), scoreboardSlices),
    ]);
    return ScannerBundle(
      movers: results[0] as List<MarketMover>,
      scanResults: (results[1] as ScanResultsPayload).results,
      scoreboard: results[2] as List<ScoreboardSlice>,
      progress: (results[1] as ScanResultsPayload).progress,
    );
  }

  Future<List<MarketMover>> fetchMovers({Map<String, String>? params}) async {
    return _getList(
      _config.moversUri(),
      (json) => MarketMover.fromJson(json),
      params: params,
    );
  }

  Future<ScanResultsPayload> fetchScanResults({Map<String, String>? params}) async {
    final targetUri = params == null
        ? _config.scanUri()
        : _config.scanUri().replace(queryParameters: {
            ..._config.scanUri().queryParameters,
            ...params,
          });
    final res = await _client.get(
      targetUri,
      headers: {'Accept': 'application/json'},
    );
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      // Handle {"results": [...], "progress": "..."} shape
      if (decoded is Map<String, dynamic>) {
        final list = decoded['results'] as List<dynamic>? ?? [];
        final progress = decoded['progress']?.toString();
        final parsed = list
            .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
        return ScanResultsPayload(parsed, progress);
      }
      // Fallback: assume bare list
      if (decoded is List) {
        final parsed = decoded
            .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
        return ScanResultsPayload(parsed, null);
      }
    }
    return ScanResultsPayload(mockScanResults, null);
  }

  Future<List<ScanResult>> fetchSymbolScan(String symbol) async {
    final uri = _config.symbolUri().replace(
      queryParameters: {
        ..._config.symbolUri().queryParameters,
        'symbol': symbol,
      },
    );
    final res = await _client.get(uri, headers: {'Accept': 'application/json'});
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      List<dynamic>? list;
      if (decoded is List) {
        list = decoded;
      } else if (decoded is Map && decoded['results'] is List) {
        list = decoded['results'] as List;
      }
      if (list != null) {
        return list
            .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
      }
    }
    return [];
  }

  Future<List<Idea>> fetchIdeas({Map<String, String>? params}) async {
    return _getList(
      _config.ideasUri(),
      (json) => Idea.fromJson(json),
      params: params,
    );
  }

  Future<List<ScoreboardSlice>> fetchScoreboard() async {
    return _getList(
      _config.scoreboardUri(),
      (json) => ScoreboardSlice.fromJson(json),
    );
  }

  Future<UniverseStats?> fetchUniverseStats() async {
    final res = await _client.get(
      _config.universeStatsUri(),
      headers: {'Accept': 'application/json'},
    );
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      if (decoded is Map<String, dynamic>) {
        return UniverseStats.fromJson(decoded);
      }
    }
    return null;
  }

  Future<List<ScanResult>> fetchSymbolScanFor(String symbol) async {
    return _getList(
      _config.symbolUri(),
      (json) => ScanResult.fromJson(json),
      params: {'symbol': symbol},
    );
  }

  Future<CopilotMessage> sendCopilot(String prompt) async {
    final res = await _client.post(
      _config.copilotUri(),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'prompt': prompt}),
    );
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      if (decoded is Map<String, dynamic>) {
        return CopilotMessage.fromJson(decoded);
      }
      if (decoded is List && decoded.isNotEmpty) {
        return CopilotMessage.fromJson(
          Map<String, dynamic>.from(decoded.first as Map),
        );
      }
      return CopilotMessage('assistant', decoded.toString());
    }
    throw Exception('HTTP ${res.statusCode}');
  }

  Future<List<T>> _getList<T>(
    Uri uri,
    T Function(Map<String, dynamic>) parser, {
    Map<String, String>? params,
  }) async {
    final targetUri = params == null
        ? uri
        : uri.replace(
            queryParameters: {
              ...uri.queryParameters,
              ...params,
            },
          );
    final res = await _client.get(
      targetUri,
      headers: {'Accept': 'application/json'},
    );
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      final list = decoded is List
          ? decoded
          : (decoded is Map && decoded['data'] is List)
          ? decoded['data'] as List
          : null;
      if (list != null) {
        return list
            .map((e) => parser(Map<String, dynamic>.from(e as Map)))
            .toList();
      }
    }
    throw Exception('Failed to load $uri');
  }

  dynamic _decode(String body) {
    try {
      return jsonDecode(body);
    } catch (_) {
      return body;
    }
  }

  Future<T> _safe<T>(Future<T> Function() task, T fallback) async {
    try {
      return await task();
    } catch (e) {
      debugPrint('Falling back to mock for $task: $e');
      return fallback;
    }
  }
}

Color _colorFromHex(String hex) {
  final buffer = StringBuffer();
  if (hex.length == 6 || hex.length == 7) buffer.write('ff');
  buffer.write(hex.replaceFirst('#', ''));
  return Color(int.parse(buffer.toString(), radix: 16));
}

const mockMovers = <MarketMover>[
  MarketMover('AAPL', '+1.3%', 'Momentum higher', true, [0.2, 0.25, 0.28, 0.27]),
  MarketMover('NVDA', '+2.1%', 'Breakout pushing', true, [0.4, 0.42, 0.45, 0.44]),
  MarketMover('TSLA', '-0.8%', 'Cooling after run', false, [0.5, 0.48, 0.46, 0.47]),
];

const mockScanResults = <ScanResult>[
  ScanResult(
    'ADBE',
    'Breakout long',
    'R:R 2.0',
    '598',
    '586',
    '620',
    'Volume +18% vs avg',
    [0.2, 0.25, 0.3, 0.27, 0.35],
  ),
  ScanResult(
    'MSFT',
    'Momentum swing',
    'R:R 2.1',
    '412',
    '404',
    '430',
    'Copilot suggests staggered exits',
    [0.4, 0.42, 0.38, 0.44, 0.5],
  ),
  ScanResult(
    'NVDA',
    'Pullback buy',
    'R:R 1.8',
    '488',
    '472',
    '520',
    'Watching semis breadth',
    [0.3, 0.28, 0.32, 0.35, 0.36],
  ),
];

const mockIdeas = <Idea>[
  Idea(
    'Breakout long',
    'ADBE',
    'R:R 2.0 | TechRating 82',
    'Entry 598 | Stop 586 | Target 620',
    [0.2, 0.25, 0.3, 0.27, 0.35],
  ),
  Idea(
    'Swing momentum',
    'MSFT',
    'R:R 2.1 | TechRating 79',
    'Entry 412 | Stop 404 | Target 430',
    [0.4, 0.42, 0.38, 0.44, 0.5],
  ),
  Idea(
    'Pullback buy',
    'NVDA',
    'R:R 1.8 | TechRating 76',
    'Entry 488 | Stop 472 | Target 520',
    [0.35, 0.3, 0.32, 0.36, 0.4],
  ),
];

const quickActions = <QuickAction>[
  QuickAction(Icons.bolt, 'Fast scan', 'Run presets'),
  QuickAction(Icons.tune, 'Filters', 'Adjust criteria'),
  QuickAction(Icons.refresh, 'Refresh', 'Pull latest data'),
  QuickAction(Icons.save_alt, 'Save as preset', 'Reuse later'),
  QuickAction(Icons.dashboard_customize, 'Layout', 'Switch density'),
  QuickAction(Icons.analytics, 'Send to Copilot', 'Explain results'),
  QuickAction(Icons.shuffle, 'Randomize', 'Run a surprise scan'),
];

const savedScreens = <SavedScreen>[
  SavedScreen('Tech growth', 'High momentum + volume', '1D swing', true),
  SavedScreen('Value / quality', 'Low volatility + cash flow', 'Weekly', true),
  SavedScreen('Event driven', 'Earnings and catalysts', 'Multi-day', false),
];

const scoreboardSlices = <ScoreboardSlice>[
  ScoreboardSlice(
    'Day trades',
    '+4.2% MTM',
    '64% win',
    'Avg hold 4h',
    Color(0xFF5EEAD4),
  ),
  ScoreboardSlice(
    'Swing',
    '+8.5% MTD',
    '58% win',
    'Avg hold 3d',
    Color(0xFFB6FF3B),
  ),
  ScoreboardSlice(
    'Long-term',
    '+12.4% YTD',
    '52% win',
    'Avg hold 6m',
    brandPrimary,
  ),
];

const copilotPrompts = <String>[
  "Summarize today's scan",
  'Explain risk on NVDA setup',
  'Compare TSLA vs AAPL momentum',
  'What moved semis this week?',
];

const copilotMessages = <CopilotMessage>[
  CopilotMessage('user', 'What changed in semis today?'),
  CopilotMessage(
    'assistant',
    'Semis led by NVDA (+2.1%) and AMD (+1.6%). Breadth improved across the group with above-average volume. Resistance at 510 is being tested.',
    meta: 'Scan basis: Volume surge +16%, trend > 50d',
  ),
  CopilotMessage('user', 'Is the risk worth it if I enter NVDA now?'),
  CopilotMessage(
    'assistant',
    '- Entry 488, stop 472, target 520 (R:R 1.8)\n'
        '- Key risks: earnings in 10d, semis vol cluster near 500\n'
        '- Suggest staggered exits at 505 / 516 if intraday volatility stays elevated.',
    meta: 'Personalized to Swing profile, risk 1%',
  ),
];
Widget _heroBanner(
  BuildContext context, {
  required String title,
  required String subtitle,
  required String badge,
  required Widget trailing,
  required Widget child,
}) {
  return Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      gradient: LinearGradient(
        colors: [tone(brandPrimary, 0.12), tone(brandDeep, 0.9)],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      borderRadius: BorderRadius.circular(20),
      border: Border.all(color: tone(Colors.white, 0.08)),
      boxShadow: [
        BoxShadow(
          color: tone(Colors.black, 0.35),
          blurRadius: 18,
          offset: const Offset(0, 12),
        ),
      ],
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: tone(Colors.white, 0.08),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: tone(Colors.white, 0.1)),
              ),
              child: Text(
                badge,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            const Spacer(),
            trailing,
          ],
        ),
        const SizedBox(height: 12),
        Text(
          title,
          style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 4),
        Text(
          subtitle,
          style: const TextStyle(color: Colors.white70, fontSize: 13),
        ),
        const SizedBox(height: 12),
        child,
      ],
    ),
  );
}

class SectionHeader extends StatelessWidget {
  final String title;
  final String? caption;
  final Widget? trailing;

  const SectionHeader(this.title, {super.key, this.caption, this.trailing});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10, top: 6),
      child: Row(
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                ),
              ),
              if (caption != null)
                Text(
                  caption!,
                  style: TextStyle(
                    color: tone(Colors.white, 0.7),
                    fontSize: 12,
                  ),
                ),
            ],
          ),
          const Spacer(),
          if (trailing != null) trailing!,
        ],
      ),
    );
  }
}

class _PulseBadge extends StatelessWidget {
  final String label;
  const _PulseBadge({required this.label});

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: label,
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: tone(Colors.white, 0.06),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: tone(Colors.white, 0.06)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  color: Color(0xFFB6FF3B),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              Text(label, style: const TextStyle(color: Colors.white)),
            ],
          ),
        ),
      ),
    );
  }
}

Widget _pillButton(IconData icon, String label, String hint) {
  return Container(
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
    decoration: BoxDecoration(
      color: tone(const Color(0xFF081910), 0.85),
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: tone(Colors.white, 0.08)),
    ),
    child: Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 18, color: const Color(0xFFAFC9FF)),
        const SizedBox(width: 8),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w700,
              ),
            ),
            Text(hint, style: const TextStyle(color: Colors.white70)),
          ],
        ),
      ],
    ),
  );
}

Widget _infoCard({
  required String title,
  required String subtitle,
  required Widget child,
}) {
  final isDark = themeIsDark.value;
  final cardBg = isDark ? tone(brandDeep, 0.82) : Colors.white;
  final borderColor = isDark ? tone(Colors.white, 0.05) : tone(Colors.black, 0.05);
  final titleStyle = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w700,
    color: isDark ? Colors.white : Colors.black87,
  );
  final subtitleStyle = TextStyle(
    color: isDark ? Colors.white70 : Colors.black54,
    fontSize: 12,
  );
  return Container(
    margin: const EdgeInsets.only(bottom: 12),
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: cardBg,
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: borderColor),
      boxShadow: [
        BoxShadow(
          color: tone(Colors.black, 0.35),
          blurRadius: 18,
          offset: const Offset(0, 12),
        ),
      ],
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: titleStyle,
        ),
        const SizedBox(height: 4),
        Text(
          subtitle,
          style: subtitleStyle,
        ),
        const SizedBox(height: 12),
        child,
      ],
    ),
  );
}

Widget _listRow(BuildContext context, MarketMover mover) {
  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 6),
    child: InkWell(
      onTap: () => _showMoverDetail(context, mover),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: mover.isPositive
                  ? tone(brandPrimary, 0.15)
                  : tone(brandAccent, 0.15),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              _fmtField(mover.delta),
              style: TextStyle(
                color: mover.isPositive ? brandPrimary : brandAccent,
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            mover.ticker,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _fmtField(mover.note),
                  style: const TextStyle(color: Colors.white70),
                  overflow: TextOverflow.ellipsis,
                ),
                if (mover.sparkline.isNotEmpty)
                  SizedBox(
                    height: 20,
                    child: Sparkline(
                      data: mover.sparkline,
                      positive: mover.sparkline.last >= mover.sparkline.first,
                    ),
                  ),
              ],
            ),
          ),
          Icon(
            mover.isPositive ? Icons.north_east : Icons.south_east,
            color: mover.isPositive ? brandPrimary : Colors.redAccent,
            size: 16,
          ),
        ],
      ),
    ),
  );
}

Widget _marketPulseCard(BuildContext context, List<MarketMover> movers) {
  return Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      gradient: LinearGradient(
        colors: [tone(brandDeep, 0.9), tone(brandDeep, 0.78)],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: tone(Colors.white, 0.05)),
      boxShadow: [
        BoxShadow(
          color: tone(Colors.black, 0.35),
          blurRadius: 12,
          offset: const Offset(0, 8),
        ),
      ],
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.trending_up, color: Colors.white70, size: 18),
            const SizedBox(width: 8),
            const Text(
              'Top movers',
              style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14),
            ),
            const Spacer(),
            Chip(
              label: Text(
                '${movers.length} symbols',
                style: TextStyle(color: tone(Colors.white, 0.7)),
              ),
              backgroundColor: tone(Colors.white, 0.05),
              visualDensity: VisualDensity.compact,
            ),
          ],
        ),
        const SizedBox(height: 10),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: movers.map((m) => _listRow(context, m)).toList(),
        ),
      ],
    ),
  );
}

void _showMoverDetail(BuildContext context, MarketMover mover) {
  showModalBottomSheet(
    context: context,
    backgroundColor: tone(brandDeep, 0.95),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    builder: (ctx) {
      return Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text(
                  mover.ticker,
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
                ),
                const SizedBox(width: 8),
                Chip(
                  label: Text(_fmtField(mover.delta)),
                  backgroundColor: mover.isPositive
                      ? tone(brandPrimary, 0.15)
                      : tone(brandAccent, 0.15),
                  visualDensity: VisualDensity.compact,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              _fmtField(mover.note),
              style: const TextStyle(color: Colors.white70),
            ),
            if (mover.sparkline.isNotEmpty) ...[
              const SizedBox(height: 10),
              SizedBox(
                height: 40,
                child: Sparkline(
                  data: mover.sparkline,
                  positive: mover.sparkline.last >= mover.sparkline.first,
                ),
              ),
            ],
          ],
        ),
      );
    },
  );
}

Widget _savedScreenCard(
  BuildContext context,
  SavedScreen screen,
  VoidCallback onApply,
) {
  return Tooltip(
    message: 'Tap to apply preset, long-press to auto-apply',
    child: MouseRegion(
      cursor: SystemMouseCursors.click,
      child: InkWell(
        onTap: onApply,
        onLongPress: () {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Applied preset "${screen.name}"'),
              duration: const Duration(seconds: 1),
            ),
          );
          onApply();
        },
        borderRadius: BorderRadius.circular(16),
        child: Container(
          width: 240,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: tone(Colors.white, 0.03),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: screen.isActive
                  ? tone(brandPrimary, 0.4)
                  : tone(Colors.white, 0.06),
            ),
            boxShadow: [
              BoxShadow(
                color: tone(Colors.black, 0.25),
                blurRadius: 10,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    screen.isActive ? Icons.check_circle : Icons.circle_outlined,
                    color: screen.isActive ? brandPrimary : Colors.white38,
                    size: 18,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    screen.horizon,
                    style: TextStyle(color: tone(Colors.white, 0.7), fontSize: 12),
                  ),
                  const Spacer(),
                  Chip(
                    label: Text(
                      screen.isActive ? 'Active' : 'Preset',
                      style: const TextStyle(color: Colors.white, fontSize: 11),
                    ),
                    backgroundColor:
                        screen.isActive ? tone(brandPrimary, 0.2) : tone(Colors.white, 0.08),
                    visualDensity: VisualDensity.compact,
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                screen.name,
                style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w800),
              ),
              const SizedBox(height: 4),
              Text(screen.description, style: const TextStyle(color: Colors.white70)),
              const Spacer(),
              Align(
                alignment: Alignment.bottomRight,
                child: OutlinedButton(
                  onPressed: onApply,
                  style: OutlinedButton.styleFrom(
                    foregroundColor:
                        screen.isActive ? brandPrimary : tone(Colors.white, 0.8),
                    side: BorderSide(
                      color: screen.isActive
                          ? tone(brandPrimary, 0.7)
                          : tone(Colors.white, 0.15),
                    ),
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: const Text('Open'),
                ),
              ),
            ],
          ),
        ),
      ),
    ),
  );
}

Widget _scanResultCard(BuildContext context, ScanResult r, VoidCallback onAnalyze) {
  final theme = Theme.of(context);
  final isDark = theme.brightness == Brightness.dark;
  final muted = theme.textTheme.bodySmall?.copyWith(
    color: isDark ? Colors.white70 : Colors.black54,
  );

  final ics = r.institutionalCoreScore;
  String tierLabel = r.icsTier ?? r.profileLabel ?? r.profileName ?? '';
  if (tierLabel.isEmpty && ics != null) {
    if (ics >= 80) {
      tierLabel = 'Tier 1';
    } else if (ics >= 65) {
      tierLabel = 'Tier 2';
    } else {
      tierLabel = 'Watch';
    }
  }
  Color tierColor;
  if (ics != null && ics >= 80) {
    tierColor = tone(brandPrimary, 0.25);
  } else if (ics != null && ics >= 65) {
    tierColor = tone(brandAccent, 0.18);
  } else {
    tierColor = tone(Colors.grey, isDark ? 0.25 : 0.18);
  }

  final winProb = r.winProb10d;
  final quality = r.qualityScore;
  double? atrPct = r.atrPct;
  if (atrPct != null && atrPct < 1) atrPct *= 100;

  final desc = [
    r.signal,
    if (r.playStyle != null && r.playStyle!.isNotEmpty) '• ${r.playStyle}',
    if (ics != null && winProb != null)
      '• ICS ${ics.toStringAsFixed(0)}/100, win ~${(winProb * 100).toStringAsFixed(0)}%',
  ].where((e) => e.isNotEmpty).join(' ');

  final eventChips = <Widget>[];
  if ((r.eventSummary ?? '').isNotEmpty) {
    eventChips.add(
      FilterChip(
        label: Text(r.eventSummary!),
        selected: false,
        visualDensity: VisualDensity.compact,
        backgroundColor: tone(brandPrimary, 0.12),
        onSelected: (_) {},
      ),
    );
  } else if ((r.eventFlags ?? '').isNotEmpty) {
    eventChips.add(
      FilterChip(
        label: Text(r.eventFlags!),
        selected: false,
        visualDensity: VisualDensity.compact,
        backgroundColor: tone(Colors.orange, 0.12),
        onSelected: (_) {},
      ),
    );
  }
  if (r.optionStrategies != null && r.optionStrategies!.isNotEmpty) {
    eventChips.add(
      FilterChip(
        label: const Text('Options idea available'),
        selected: false,
        visualDensity: VisualDensity.compact,
        backgroundColor: tone(brandAccent, 0.14),
        onSelected: (_) {},
      ),
    );
  }

  return Card(
    margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 0),
    child: Padding(
      padding: const EdgeInsets.all(14.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      r.ticker,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 0.1,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '${r.sector ?? ''}${(r.sector != null && r.industry != null) ? '  •  ' : ''}${r.industry ?? ''}',
                      style: muted?.copyWith(fontSize: 11),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
              if (tierLabel.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: tierColor,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${tierLabel}${ics != null ? ' • ${ics.toStringAsFixed(0)}/100' : ''}',
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      color: isDark ? Colors.white : Colors.black87,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 12,
            runSpacing: 8,
            children: [
              _miniStat('TechRating', r.techRating),
              _miniStat('AlphaScore', r.alphaScore),
              if (winProb != null) _miniStat('Win prob 10d', winProb * 100, suffix: '%'),
              if (quality != null) _miniStat('Quality', quality),
              if (atrPct != null) _miniStat('ATR%', atrPct, suffix: '%'),
              if (r.profileLabel != null || r.profileName != null)
                _miniStat('Profile', null,
                    labelOnly: r.profileLabel ?? r.profileName ?? ''),
              if (r.isUltraRisky == true)
                Chip(
                  label: const Text('ULTRA-RISKY'),
                  backgroundColor: tone(Colors.red, 0.18),
                  visualDensity: VisualDensity.compact,
                ),
            ],
          ),
          if (eventChips.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: eventChips,
            ),
          ],
          const SizedBox(height: 10),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      desc.isNotEmpty ? desc : _fmtField(r.note),
                      style: muted,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 6,
                      children: [
                        Chip(
                          label: Text('Entry ${r.entry}'),
                          backgroundColor: tone(Colors.greenAccent, 0.12),
                          visualDensity: VisualDensity.compact,
                        ),
                        Chip(
                          label: Text('Target ${r.target}'),
                          backgroundColor: tone(brandPrimary, 0.14),
                          visualDensity: VisualDensity.compact,
                        ),
                        Chip(
                          label: Text('Stop ${r.stop}'),
                          backgroundColor: tone(Colors.redAccent, 0.12),
                          visualDensity: VisualDensity.compact,
                        ),
                        if (r.rrr.isNotEmpty)
                          Chip(
                            label: Text(r.rrr),
                            backgroundColor: tone(Colors.orange, 0.12),
                            visualDensity: VisualDensity.compact,
                          ),
                      ],
                    ),
                  ],
                ),
              ),
              Column(
                children: [
                  IconButton(
                    icon: const Icon(Icons.chat_bubble_outline),
                    onPressed: onAnalyze,
                    tooltip: 'Ask Copilot',
                  ),
                  IconButton(
                    icon: const Icon(Icons.info_outline),
                    onPressed: () => _showScanDetail(context, r, onAnalyze),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    ),
  );
}

Widget _miniStat(String label, double? value, {String suffix = '', String? labelOnly}) {
  final display = labelOnly ??
      (value != null
          ? '${value.toStringAsFixed(value.abs() >= 100 ? 0 : 1)}$suffix'
          : '--');
  return Chip(
    label: Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w700)),
        Text(display, style: const TextStyle(fontSize: 12)),
      ],
    ),
    visualDensity: VisualDensity.compact,
    backgroundColor: tone(Colors.white, 0.06),
  );
}

Widget _ideaCard(Idea idea, VoidCallback onCopilot) {
  return Container(
    margin: const EdgeInsets.only(bottom: 12),
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: tone(const Color(0xFF081910), 0.85),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: tone(Colors.white, 0.07)),
      boxShadow: [
        BoxShadow(
          color: tone(Colors.black, 0.25),
          blurRadius: 10,
          offset: const Offset(0, 6),
        ),
      ],
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              idea.ticker,
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
            ),
            const SizedBox(width: 8),
            Chip(
              label: Text(idea.title),
              backgroundColor: tone(Colors.white, 0.08),
              visualDensity: VisualDensity.compact,
            ),
            const Spacer(),
            Icon(Icons.trending_up, color: tone(Colors.white, 0.6)),
          ],
        ),
        const SizedBox(height: 6),
        Text(
          '${idea.ticker}: ${_fmtField(idea.plan) == "?" ? "No plan text provided" : idea.plan}',
          style: const TextStyle(color: Colors.white70),
        ),
        if (_fmtField(idea.meta) != '?') ...[
          const SizedBox(height: 6),
          Text(
            'Why this was picked: ${_fmtField(idea.meta)}',
            style: const TextStyle(color: Colors.white60, fontSize: 12),
          ),
        ],
        if (idea.option != null) ...[
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: [
              Chip(
                label: Text(
                  '${(idea.option?["contract_type"] ?? "call").toString().toUpperCase()} ${_fmtField(idea.option?["strike"]?.toString())}',
                ),
                backgroundColor: tone(brandPrimary, 0.18),
              ),
              if (idea.option?["expiration"] != null)
                Chip(
                  label: Text('Exp ${idea.option?["expiration"]}'),
                  backgroundColor: tone(Colors.white, 0.08),
                ),
              if (idea.option?["delta"] != null)
                Chip(
                  label: Text('? ${_fmtField(idea.option?["delta"]?.toString())}'),
                  backgroundColor: tone(Colors.white, 0.08),
                ),
              if (idea.option?["bid"] != null || idea.option?["ask"] != null)
                Chip(
                  label: Text(
                    'Bid ${_fmtField(idea.option?["bid"]?.toString())} / Ask ${_fmtField(idea.option?["ask"]?.toString())}',
                  ),
                  backgroundColor: tone(Colors.white, 0.08),
                ),
            ],
          ),
        ],
        if (idea.sparkline.isNotEmpty) ...[
          const SizedBox(height: 10),
          SizedBox(
            height: 36,
            child: Sparkline(
              data: idea.sparkline,
              positive: idea.sparkline.last >= idea.sparkline.first,
            ),
          ),
        ],
        const SizedBox(height: 8),
        Row(
          children: [
            Icon(
              Icons.chat_bubble_outline,
              size: 16,
              color: tone(Colors.white, 0.7),
            ),
            const SizedBox(width: 6),
            Expanded(
              child: Text(
                'Ask Copilot about this idea for a tailored breakdown.',
                style: TextStyle(color: tone(Colors.white, 0.7)),
              ),
            ),
            const SizedBox(width: 12),
            OutlinedButton.icon(
              onPressed: onCopilot,
              icon: const Icon(Icons.auto_awesome),
              label: const Text('Ask Copilot'),
            ),
          ],
        ),
      ],
    ),
  );
}

Widget _scoreboardCard(List<ScoreboardSlice> slices) {
  return Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      gradient: LinearGradient(
        colors: [tone(brandDeep, 0.92), tone(brandDeep, 0.78)],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      borderRadius: BorderRadius.circular(18),
      border: Border.all(color: tone(Colors.white, 0.06)),
      boxShadow: [
        BoxShadow(
          color: tone(Colors.black, 0.35),
          blurRadius: 16,
          offset: const Offset(0, 8),
        ),
      ],
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: const [
            Icon(Icons.flag_outlined, size: 18, color: Colors.white70),
            SizedBox(width: 8),
            Text(
              'Performance scoreboard',
              style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14),
            ),
          ],
        ),
        const SizedBox(height: 10),
        ...slices.map(
          (s) => Tooltip(
                message: 'Tap to view details and definitions',
                child: Container(
            margin: const EdgeInsets.symmetric(vertical: 6),
            padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
            decoration: BoxDecoration(
              color: tone(Colors.white, 0.02),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: tone(s.accent, 0.25)),
            ),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 48,
                  decoration: BoxDecoration(
                    color: tone(s.accent, 0.45),
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Tooltip(
                        message: 'Strategy label ? tagged when you log trades',
                        child: Text(
                          s.label,
                          style: const TextStyle(
                            fontWeight: FontWeight.w800,
                            fontSize: 13,
                          ),
                        ),
                      ),
                      const SizedBox(height: 2),
                      Tooltip(
                        message: 'Holding horizon and open/total trade count',
                        child: Text(
                          s.horizon,
                          style:
                              const TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                      ),
                    ],
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Tooltip(
                        message: 'P&L performance for this strategy/category',
                        child: Text(
                          s.pnl,
                          style: TextStyle(
                            color: s.pnl.startsWith('-')
                                ? Colors.redAccent
                              : brandPrimary,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                    const SizedBox(height: 2),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: tone(Colors.white, 0.06),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Tooltip(
                        message: 'Win rate (closed positions only)',
                        child: Text(
                          s.winRate,
                          style: const TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          )),
        ),
      ],
    ),
  );
}

Widget _copilotInlineCard(BuildContext context) {
  return Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: tone(Colors.white, 0.03),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: tone(Colors.white, 0.05)),
      boxShadow: [
        BoxShadow(
          color: tone(Colors.black, 0.25),
          blurRadius: 10,
          offset: const Offset(0, 6),
        ),
      ],
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.chat_bubble_outline, color: Colors.white70, size: 16),
            const SizedBox(width: 6),
            const Text(
              'Need context?',
              style: TextStyle(fontWeight: FontWeight.w800),
            ),
            const Spacer(),
            OutlinedButton.icon(
              onPressed: () {
                copilotPrefill.value = 'Summarize this scan and key risks.';
                _shellKey.currentState?.setTab(2);
              },
              icon: const Icon(Icons.play_arrow),
              label: const Text('Open Copilot'),
            ),
          ],
        ),
        const SizedBox(height: 6),
        const Text(
          'Tap to ask Copilot while keeping your scanner list in view.',
          style: TextStyle(color: Colors.white70),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: const [
            _PulseBadge(label: 'Explain this screen'),
            _PulseBadge(label: 'Highlight risks'),
            _PulseBadge(label: 'Suggest targets'),
          ],
        ),
        const SizedBox(height: 8),
        ValueListenableBuilder<String?>(
          valueListenable: copilotStatus,
          builder: (context, status, _) {
            if (status == null) return const SizedBox.shrink();
            return Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: tone(Colors.white, 0.05),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: tone(Colors.white, 0.08)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.info_outline, size: 16, color: Colors.white70),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      'Copilot offline: $status\nCached guidance will display until the service recovers.',
                      style: const TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                  ),
                  TextButton(
                    onPressed: () {
                      copilotPrefill.value = 'Retry Copilot with the last question.';
                      _shellKey.currentState?.setTab(2);
                    },
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          },
        ),
      ],
    ),
  );
}

class Sparkline extends StatelessWidget {
  final List<double> data;
  final bool positive;

  const Sparkline({super.key, required this.data, required this.positive});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _SparklinePainter(
        data,
        positive ? brandPrimary : Colors.redAccent,
      ),
      size: const Size(double.infinity, double.infinity),
    );
  }
}

class _SparklinePainter extends CustomPainter {
  final List<double> data;
  final Color color;
  const _SparklinePainter(this.data, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;
    final minValue = data.reduce(math.min);
    final maxValue = data.reduce(math.max);
    final range = (maxValue - minValue).abs() < 0.0001
        ? 1.0
        : maxValue - minValue;

    final points = <Offset>[];
    for (var i = 0; i < data.length; i++) {
      final x = size.width * (i / math.max(1, data.length - 1));
      final normalized = (data[i] - minValue) / range;
      final y = size.height - (normalized * size.height);
      points.add(Offset(x, y));
    }

    final path = Path()..moveTo(points.first.dx, points.first.dy);
    for (var i = 1; i < points.length; i++) {
      path.lineTo(points[i].dx, points[i].dy);
    }

    final paint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final fillPaint = Paint()
      ..shader = LinearGradient(
        colors: [tone(color, 0.25), Colors.transparent],
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height))
      ..style = PaintingStyle.fill;

    final fillPath = Path.from(path)
      ..lineTo(points.last.dx, size.height)
      ..lineTo(points.first.dx, size.height)
      ..close();

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _ProfileRow extends StatelessWidget {
  final String label;
  final String value;
  const _ProfileRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Colors.white70,
              fontWeight: FontWeight.w600,
            ),
          ),
          const Spacer(),
          Text(value, style: const TextStyle(color: Colors.white)),
        ],
      ),
    );
  }
}

Widget _messageBubble(CopilotMessage message) {
  final isUser = message.role == 'user';
  return Align(
    alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
    child: Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(12),
      constraints: const BoxConstraints(maxWidth: 320),
      decoration: BoxDecoration(
        color: isUser ? tone(brandPrimary, 0.22) : tone(Colors.white, 0.05),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: isUser ? tone(brandPrimary, 0.6) : tone(Colors.white, 0.1),
        ),
      ),
      child: Column(
        crossAxisAlignment: isUser
            ? CrossAxisAlignment.end
            : CrossAxisAlignment.start,
        children: [
          Text(message.body, style: const TextStyle(color: Colors.white)),
          if (message.meta != null) ...[
            const SizedBox(height: 6),
            Text(
              message.meta!,
              style: const TextStyle(color: Colors.white70, fontSize: 12),
            ),
          ],
        ],
      ),
    ),
  );
}

String _fmtField(String? v) {
  if (v == null) return '?';
  final trimmed = v.trim();
  if (trimmed.isEmpty) return '?';
  if (trimmed.toLowerCase() == 'nan') return '?';
  return trimmed;
}

String _fmtLocalTime(DateTime dt) {
  final local = dt.toLocal();
  final today = DateTime.now();
  final daysDiff = DateTime(today.year, today.month, today.day)
      .difference(DateTime(local.year, local.month, local.day))
      .inDays;
  final hour12 = local.hour % 12 == 0 ? 12 : local.hour % 12;
  final mm = local.minute.toString().padLeft(2, '0');
  final ampm = local.hour >= 12 ? 'p' : 'a';
  String dayLabel;
  if (daysDiff == 0) {
    dayLabel = 'Today';
  } else if (daysDiff == 1) {
    dayLabel = 'Yesterday';
  } else {
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    final month = months[local.month - 1];
    final yearSuffix = local.year == today.year ? '' : ' ${local.year}';
    dayLabel = '$month ${local.day}$yearSuffix';
  }
  return '$dayLabel $hour12:$mm$ampm';
}

void _showScanDetail(
  BuildContext context,
  ScanResult r,
  VoidCallback onAnalyze,
) {
  showModalBottomSheet(
    context: context,
    backgroundColor: tone(brandDeep, 0.95),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    builder: (ctx) {
      return Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text(
                  r.ticker,
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
                ),
                const SizedBox(width: 8),
                Chip(
                  label: Text(r.signal),
                  backgroundColor: tone(Colors.white, 0.08),
                  visualDensity: VisualDensity.compact,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              _fmtField(r.note),
              style: const TextStyle(color: Colors.white70),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 6,
              children: [
                Chip(
                  label: Text('Entry: ${r.entry}'),
                  backgroundColor: tone(Colors.white, 0.05),
                  visualDensity: VisualDensity.compact,
                ),
                Chip(
                  label: Text('Target: ${r.target}'),
                  backgroundColor: tone(Colors.white, 0.05),
                  visualDensity: VisualDensity.compact,
                ),
                Chip(
                  label: Text('Stop: ${r.stop}'),
                  backgroundColor: tone(Colors.redAccent, 0.12),
                  visualDensity: VisualDensity.compact,
                ),
            Chip(
              label: Text(r.rrr),
              backgroundColor: tone(Colors.greenAccent, 0.12),
              visualDensity: VisualDensity.compact,
            ),
              ],
            ),
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerRight,
              child: ElevatedButton.icon(
                onPressed: onAnalyze,
                icon: const Icon(Icons.chat_bubble_outline),
                label: const Text('Ask Copilot'),
              ),
            ),
          ],
        ),
      );
    },
  );
}

class LocalStore {
  static Future<SharedPreferences> _prefs() => SharedPreferences.getInstance();

  static Future<String?> loadUser() async {
    final p = await _prefs();
    return p.getString('user_id');
  }

  static Future<void> saveUser(String user) async {
    final p = await _prefs();
    await p.setString('user_id', user);
  }

  static Future<int?> loadLastTab() async {
    final p = await _prefs();
    return p.getInt('last_tab');
  }

  static Future<void> saveLastTab(int index) async {
    final p = await _prefs();
    await p.setInt('last_tab', index);
  }

  static Future<Map<String, dynamic>?> loadScannerState() async {
    final p = await _prefs();
    final filters = p.getString('filters');
    if (filters == null) return null;
    final saved = p.getString('saved_screens');
    final lastScans = p.getString('last_scans');
    final lastMovers = p.getString('last_movers');
    return {
      'filters': Map<String, String>.from(jsonDecode(filters)),
      'saved_screens': saved != null
          ? (jsonDecode(saved) as List)
              .map((e) => SavedScreen.fromJson(Map<String, dynamic>.from(e)))
              .toList()
          : savedScreens,
      'scanCount': p.getInt('scan_count') ?? 0,
      'streakDays': p.getInt('streak_days') ?? 0,
      'lastScan': p.getString('last_scan'),
      'advancedMode': p.getBool('advanced_mode') ?? true,
      'showOnboarding': p.getBool('show_onboarding') ?? true,
      'last_scans': lastScans != null
          ? (jsonDecode(lastScans) as List)
              .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e)))
              .toList()
          : <ScanResult>[],
      'last_movers': lastMovers != null
          ? (jsonDecode(lastMovers) as List)
              .map((e) => MarketMover.fromJson(Map<String, dynamic>.from(e)))
              .toList()
          : <MarketMover>[],
    };
  }

  static Future<void> saveScannerState({
    required Map<String, String> filters,
    required List<SavedScreen> savedScreens,
    required int scanCount,
    required int streakDays,
    required DateTime? lastScan,
    required bool advancedMode,
    required bool showOnboarding,
    List<ScanResult>? lastScans,
    List<MarketMover>? lastMovers,
  }) async {
    final p = await _prefs();
    await p.setString('filters', jsonEncode(filters));
    await p.setString(
      'saved_screens',
      jsonEncode(savedScreens.map((e) => e.toJson()).toList()),
    );
    await p.setInt('scan_count', scanCount);
    await p.setInt('streak_days', streakDays);
    if (lastScan != null) {
      await p.setString('last_scan', lastScan.toIso8601String());
    }
    await p.setBool('advanced_mode', advancedMode);
    await p.setBool('show_onboarding', showOnboarding);
    if (lastScans != null) {
      await p.setString('last_scans', jsonEncode(lastScans.map((e) => e.toJson()).toList()));
    }
    if (lastMovers != null) {
      await p.setString('last_movers', jsonEncode(lastMovers.map((e) => e.toJson()).toList()));
    }
  }

  static Future<void> saveLastBundle(ScannerBundle bundle) async {
    final p = await _prefs();
    await p.setString(
      'last_scans',
      jsonEncode(bundle.scanResults.map((e) => e.toJson()).toList()),
    );
    await p.setString(
      'last_movers',
      jsonEncode(bundle.movers.map((e) => e.toJson()).toList()),
    );
  }
}
