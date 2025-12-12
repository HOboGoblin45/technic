/// Ideas Page
/// 
/// Displays curated trade ideas derived from scanner results.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/idea.dart';
import '../../models/scan_result.dart';
import '../../providers/app_providers.dart';
import '../../theme/app_colors.dart'; // Using tone from helpers.dart
import '../../utils/helpers.dart';
import '../../utils/mock_data.dart';
import '../../widgets/info_card.dart';
import '../../widgets/pulse_badge.dart';
import 'widgets/idea_card.dart';

class IdeasPage extends ConsumerStatefulWidget {
  const IdeasPage({super.key});

  @override
  ConsumerState<IdeasPage> createState() => _IdeasPageState();
}

class _IdeasPageState extends ConsumerState<IdeasPage>
    with AutomaticKeepAliveClientMixin {
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
    final scans = ref.read(lastScanResultsProvider);
    
    if (scans.isNotEmpty) {
      return scans.map((s) {
        final plan =
            'Entry ${s.entry.isNotEmpty ? s.entry : "-"}, Stop ${s.stop.isNotEmpty ? s.stop : "-"}, Target ${s.target.isNotEmpty ? s.target : "-"}';
        final why =
            '${s.signal} setup based on blended trend, momentum, volume, and risk scores.';
        return Idea(s.signal, s.ticker, why, plan, s.sparkline);
      }).toList();
    }
    
    try {
      final apiService = ref.read(apiServiceProvider);
      return await apiService.fetchIdeas();
    } catch (_) {
      return mockIdeas;
    }
  }

  Widget _buildHeroBanner({
    required String title,
    required String subtitle,
    required String badge,
    required Widget trailing,
    required Widget child,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: tone(AppColors.darkCard, 0.5),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: tone(Colors.white, 0.08)),
        boxShadow: [
          BoxShadow(
            color: tone(Colors.black, 0.15),
            blurRadius: 6,
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
              // Hero Banner
              _buildHeroBanner(
                title: 'Personal Quant ideas',
                subtitle:
                    'Card stack with sparkline previews and quick execute hooks.',
                badge: 'Live feed',
                trailing: OutlinedButton.icon(
                  onPressed: _refresh,
                  icon: const Icon(Icons.filter_alt),
                  label: const Text('Filter ideas'),
                ),
                child: const Row(
                  children: [
                    PulseBadge(
                      text: 'Copilot ready',
                      color: AppColors.successGreen,
                    ),
                    SizedBox(width: 8),
                    PulseBadge(
                      text: 'Time horizons mixed',
                      color: AppColors.successGreen,
                    ),
                    SizedBox(width: 8),
                    PulseBadge(
                      text: 'Risk tuned to 1%',
                      color: AppColors.successGreen,
                    ),
                  ],
                ),
              ),

              // Error State
              if (hasError) ...[
                const SizedBox(height: 10),
                InfoCard(
                  title: 'Ideas feed unavailable',
                  subtitle: snapshot.error.toString(),
                  child: TextButton(
                    onPressed: _refresh,
                    child: const Text('Retry'),
                  ),
                ),
              ],

              // Loading State
              if (loading && !hasIdeas) ...[
                const SizedBox(height: 20),
                const Center(
                  child: CircularProgressIndicator(),
                ),
                const SizedBox(height: 20),
              ],

              // Empty State
              if (!loading && !hasIdeas && !hasError) ...[
                const SizedBox(height: 10),
                const InfoCard(
                  title: 'No ideas yet',
                  subtitle: 'Run a scan to generate trade ideas.',
                  child: Text(
                    'Ideas are derived from your latest scan results. Pull to refresh or navigate to Scanner to run a new scan.',
                    style: TextStyle(color: Colors.white70),
                  ),
                ),
              ],

              // Ideas List
              if (hasIdeas) ...[
                const SizedBox(height: 16),
                ...ideas.map((idea) {
                  return IdeaCard(
                    idea: idea,
                    onAskCopilot: () {
                      // Set context and navigate to Copilot
                      // Find matching scan result if available
                      final scans = ref.read(lastScanResultsProvider);
                      final matchingScan = scans.firstWhere(
                        (s) => s.ticker == idea.ticker,
                        orElse: () => ScanResult(
                          idea.ticker,
                          idea.title,
                          '',
                          '',
                          '',
                          '',
                          '',
                          idea.sparkline,
                        ),
                      );
                      
                      ref.read(copilotContextProvider.notifier).state =
                          matchingScan;
                      ref.read(copilotPrefillProvider.notifier).state =
                          'Explain the ${idea.title} setup for ${idea.ticker}';
                      ref.read(currentTabProvider.notifier).setTab(2);
                    },
                    onSave: () {
                      // Add to watchlist
                      ref.read(watchlistProvider.notifier).add(
                            idea.ticker,
                            note: idea.title,
                          );
                      
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text('${idea.ticker} saved to My Ideas'),
                        ),
                      );
                    },
                  );
                }),
              ],

              const SizedBox(height: 20),
            ],
          );
        },
      ),
    );
  }
}
