/// Scanner Page
/// 
/// Main scanner interface with quantitative stock analysis.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/scanner_bundle.dart';
import '../../models/scan_result.dart';
import '../../models/market_mover.dart';
import '../../models/saved_screen.dart';
import '../../providers/app_providers.dart';
import '../../services/local_store.dart';
import '../../theme/app_colors.dart'; // Using tone from helpers.dart
import '../../utils/helpers.dart';
import '../../widgets/section_header.dart';
import 'widgets/widgets.dart';
import '../symbol_detail/symbol_detail_page.dart';

class ScannerPage extends ConsumerStatefulWidget {
  const ScannerPage({super.key});

  @override
  ConsumerState<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends ConsumerState<ScannerPage>
    with AutomaticKeepAliveClientMixin {
  late Future<ScannerBundle> _bundleFuture;
  Map<String, String> _filters = {};
  List<SavedScreen> _savedScreens = [];
  int _scanCount = 0;
  int _streakDays = 0;
  DateTime? _lastScan;
  bool _advancedMode = false;
  bool _showOnboarding = true;

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    _loadState();
    _bundleFuture = _fetchBundle();
  }

  Future<void> _loadState() async {
    final state = await LocalStore.loadScannerState();
    if (mounted) {
      setState(() {
        final filtersMap = state?['filters'] as Map?;
        _filters = filtersMap?.map(
          (k, v) => MapEntry(k.toString(), v.toString()),
        ) ?? {};
        
        final screensList = state?['saved_screens'] as List?;
        _savedScreens = screensList
          ?.map((e) => SavedScreen.fromJson(e as Map<String, dynamic>))
          .toList() ?? [];
        
        _scanCount = (state?['scan_count'] as int?) ?? 0;
        _streakDays = (state?['streak_days'] as int?) ?? 0;
        _lastScan = state?['last_scan'] as DateTime?;
        _advancedMode = (state?['advanced_mode'] as bool?) ?? false;
        _showOnboarding = (state?['show_onboarding'] as bool?) ?? true;
      });
    }
  }

  Future<void> _saveState() async {
    await LocalStore.saveScannerState(
      filters: _filters,
      savedScreens: _savedScreens,
      scanCount: _scanCount,
      streakDays: _streakDays,
      lastScan: _lastScan,
      advancedMode: _advancedMode,
      showOnboarding: _showOnboarding,
    );
  }

  Future<ScannerBundle> _fetchBundle() async {
    try {
      final apiService = ref.read(apiServiceProvider);
      final bundle = await apiService.fetchScannerBundle(
        params: _filters.isNotEmpty ? _filters : null,
      );

      // Update scan stats
      setState(() {
        _scanCount++;
        final now = DateTime.now();
        if (_lastScan != null &&
            now.difference(_lastScan!).inHours < 24 &&
            now.day != _lastScan!.day) {
          _streakDays++;
        } else if (_lastScan == null ||
            now.difference(_lastScan!).inHours >= 48) {
          _streakDays = 1;
        }
        _lastScan = now;
      });
      _saveState();

      // Save to local store for offline access
      await LocalStore.saveLastBundle(bundle);

      return bundle;
    } catch (e) {
      // Try to load cached data
      final state = await LocalStore.loadScannerState();
      final scansList = state?['last_scans'] as List?;
      final lastScans = scansList
        ?.map((e) => ScanResult.fromJson(e as Map<String, dynamic>))
        .toList() ?? [];
      
      final moversList = state?['last_movers'] as List?;
      final lastMovers = moversList
        ?.map((e) => MarketMover.fromJson(e as Map<String, dynamic>))
        .toList() ?? [];

      if (lastScans.isNotEmpty) {
        return ScannerBundle(
          scanResults: lastScans,
          movers: lastMovers,
          scoreboard: [],
          progress: 'Loaded from cache (offline mode)',
        );
      }

      rethrow;
    }
  }

  void _refresh() {
    setState(() {
      _bundleFuture = _fetchBundle();
    });
  }

  void _applyProfile(String profile) {
    setState(() {
      switch (profile) {
        case 'conservative':
          _filters = {
            'trade_style': 'Position',
            'min_tech_rating': '7.0',
            'lookback_days': '180',
            'sector': '',
          };
          break;
        case 'moderate':
          _filters = {
            'trade_style': 'Swing',
            'min_tech_rating': '5.0',
            'lookback_days': '90',
            'sector': '',
          };
          break;
        case 'aggressive':
          _filters = {
            'trade_style': 'Day',
            'min_tech_rating': '3.0',
            'lookback_days': '30',
            'sector': '',
          };
          break;
      }
    });
    _saveState();
    _refresh();
  }

  void _randomize() {
    final sectors = ['', 'Technology', 'Healthcare', 'Financial Services', 'Energy'];
    final styles = ['Day', 'Swing', 'Position'];
    final random = DateTime.now().millisecondsSinceEpoch;

    setState(() {
      _filters = {
        'trade_style': styles[random % styles.length],
        'sector': sectors[random % sectors.length],
        'min_tech_rating': ((random % 8) + 2).toString(),
        'lookback_days': ((random % 12) * 30 + 30).toString(),
      };
    });
    _saveState();
    _refresh();
  }

  void _showFilterPanel() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => FilterPanel(
        filters: _filters,
        onFiltersChanged: (newFilters) {
          setState(() {
            _filters = newFilters;
          });
          _saveState();
        },
      ),
    ).then((result) {
      if (result != null) {
        _refresh();
      }
    });
  }

  void _showPresetManager() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => PresetManager(
        presets: _savedScreens,
        onLoad: (preset) {
          setState(() {
            _filters = preset.params ?? {};
          });
          _saveState();
          _refresh();
        },
        onDelete: (name) {
          setState(() {
            _savedScreens.removeWhere((s) => s.name == name);
          });
          _saveState();
        },
        onSaveNew: _saveCurrentAsPreset,
      ),
    );
  }

  void _saveCurrentAsPreset() {
    final controller = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Save Preset'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: 'Preset Name',
            hintText: 'e.g., My Tech Swing Strategy',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              if (controller.text.isNotEmpty) {
                setState(() {
                  _savedScreens.add(
                    SavedScreen(
                      controller.text,
                      'Custom preset',
                      _filters['trade_style'] ?? 'Swing',
                      true,
                      params: Map.from(_filters),
                    ),
                  );
                });
                _saveState();
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Saved "${controller.text}"')),
                );
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);

    return Scaffold(
      body: RefreshIndicator(
        onRefresh: () async {
          _refresh();
          await _bundleFuture;
        },
        child: CustomScrollView(
          slivers: [
            // App Bar
            SliverAppBar(
              floating: true,
              snap: true,
              title: Row(
                children: [
                  const Text(
                    'Scanner',
                    style: TextStyle(fontWeight: FontWeight.w800),
                  ),
                  const SizedBox(width: 12),
                  if (_scanCount > 0)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: tone(AppColors.primaryBlue, 0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '$_scanCount scans',
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  if (_streakDays > 1) ...[
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: tone(Colors.orange, 0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.local_fire_department,
                              size: 14, color: Colors.orange),
                          const SizedBox(width: 4),
                          Text(
                            '$_streakDays days',
                            style: const TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: Colors.orange,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
              actions: [
                IconButton(
                  onPressed: _showFilterPanel,
                  icon: Badge(
                    isLabelVisible: _filters.isNotEmpty,
                    label: Text(_filters.length.toString()),
                    child: const Icon(Icons.tune),
                  ),
                  tooltip: 'Filters',
                ),
                IconButton(
                  onPressed: _showPresetManager,
                  icon: Badge(
                    isLabelVisible: _savedScreens.isNotEmpty,
                    label: Text(_savedScreens.length.toString()),
                    child: const Icon(Icons.bookmark_outline),
                  ),
                  tooltip: 'Presets',
                ),
              ],
            ),

            // Content
            SliverPadding(
              padding: const EdgeInsets.all(16),
              sliver: FutureBuilder<ScannerBundle>(
                future: _bundleFuture,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return SliverFillRemaining(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            CircularProgressIndicator(
                              color: AppColors.primaryBlue,
                            ),
                            const SizedBox(height: 16),
                            const Text(
                              'Scanning markets...',
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.white70,
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  }

                  if (snapshot.hasError) {
                    return SliverFillRemaining(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.error_outline,
                              size: 64,
                              color: Colors.red.withValues(alpha: 0.5),
                            ),
                            const SizedBox(height: 16),
                            const Text(
                              'Failed to load scan results',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              snapshot.error.toString(),
                              style: const TextStyle(
                                fontSize: 14,
                                color: Colors.white60,
                              ),
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 24),
                            ElevatedButton.icon(
                              onPressed: _refresh,
                              icon: const Icon(Icons.refresh),
                              label: const Text('Retry'),
                            ),
                          ],
                        ),
                      ),
                    );
                  }

                  final bundle = snapshot.data!;

                  return SliverList(
                    delegate: SliverChildListDelegate([
                      // Onboarding
                      if (_showOnboarding)
                        OnboardingCard(
                          onDismiss: () {
                            setState(() {
                              _showOnboarding = false;
                            });
                            _saveState();
                          },
                        ),

                      // Quick Actions
                      QuickActions(
                        onConservative: () => _applyProfile('conservative'),
                        onModerate: () => _applyProfile('moderate'),
                        onAggressive: () => _applyProfile('aggressive'),
                        onRandomize: _randomize,
                        advancedMode: _advancedMode,
                        onAdvancedModeChanged: (value) {
                          setState(() {
                            _advancedMode = value;
                          });
                          _saveState();
                        },
                      ),

                      // Market Pulse
                      if (bundle.movers.isNotEmpty)
                        MarketPulseCard(movers: bundle.movers),

                      // Scoreboard
                      if (bundle.scoreboard.isNotEmpty)
                        ScoreboardCard(slices: bundle.scoreboard),

                      // Results Header
                      SectionHeader(
                        'Scan Results',
                        caption: '${bundle.scanResults.length} opportunities',
                        trailing: IconButton(
                          onPressed: _refresh,
                          icon: const Icon(Icons.refresh, size: 20),
                          tooltip: 'Refresh',
                        ),
                      ),

                      // Results List
                      if (bundle.scanResults.isEmpty)
                        Center(
                          child: Padding(
                            padding: const EdgeInsets.all(40),
                            child: Column(
                              children: [
                                Icon(
                                  Icons.search_off,
                                  size: 64,
                                  color: Colors.white38,
                                ),
                                const SizedBox(height: 16),
                                const Text(
                                  'No results found',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.w700,
                                    color: Colors.white70,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                const Text(
                                  'Try adjusting your filters or profile',
                                  style: TextStyle(
                                    fontSize: 14,
                                    color: Colors.white38,
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                              ],
                            ),
                          ),
                        )
                      else
                        ...bundle.scanResults.map(
                          (result) => ScanResultCard(
                            result: result,
                            onTap: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => SymbolDetailPage(
                                    ticker: result.ticker,
                                    scanResult: result,
                                  ),
                                ),
                              );
                            },
                          ),
                        ),

                      // Bottom spacing
                      const SizedBox(height: 80),
                    ]),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
