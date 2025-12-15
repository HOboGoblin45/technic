/// Watchlist Page
/// 
/// Displays user's saved symbols with quick actions and navigation.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/app_providers.dart';
import '../../models/watchlist_item.dart';
import '../../theme/app_colors.dart';
import '../../utils/helpers.dart';
import '../symbol_detail/symbol_detail_page.dart';
import '../auth/login_page.dart';
import 'widgets/add_note_dialog.dart';
import 'widgets/tag_selector.dart';

class WatchlistPage extends ConsumerStatefulWidget {
  const WatchlistPage({super.key});

  @override
  ConsumerState<WatchlistPage> createState() => _WatchlistPageState();
}

class _WatchlistPageState extends ConsumerState<WatchlistPage> {
  final _addSymbolController = TextEditingController();
  final _searchController = TextEditingController();
  final List<String> _selectedTagFilters = [];
  String _searchQuery = '';

  @override
  void dispose() {
    _addSymbolController.dispose();
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _showAddSymbolDialog() async {
    _addSymbolController.clear();
    
    await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.darkCard,
        title: const Text('Add to Watchlist'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _addSymbolController,
              autofocus: true,
              textCapitalization: TextCapitalization.characters,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                labelText: 'Symbol',
                labelStyle: const TextStyle(color: Colors.white70),
                hintText: 'e.g., AAPL',
                hintStyle: const TextStyle(color: Colors.white38),
                prefixIcon: const Icon(Icons.search, color: Colors.white70),
                filled: true,
                fillColor: const Color(0xFF1A1F3A),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: AppColors.primaryBlue, width: 2),
                ),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final symbol = _addSymbolController.text.trim().toUpperCase();
              if (symbol.isNotEmpty) {
                ref.read(watchlistProvider.notifier).add(
                  symbol,
                  signal: 'Manual add',
                );
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('$symbol added to watchlist'),
                    backgroundColor: AppColors.successGreen,
                  ),
                );
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryBlue,
            ),
            child: const Text('Add'),
          ),
        ],
      ),
    );
  }

  Future<void> _removeSymbol(String ticker) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.darkCard,
        title: const Text('Remove from Watchlist'),
        content: Text('Remove $ticker from your watchlist?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
            ),
            child: const Text('Remove'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await ref.read(watchlistProvider.notifier).remove(ticker);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('$ticker removed from watchlist'),
          ),
        );
      }
    }
  }

  Future<void> _editNote(String ticker, String? currentNote) async {
    await showAddNoteDialog(
      context: context,
      ticker: ticker,
      initialNote: currentNote,
      onSave: (note) async {
        await ref.read(watchlistProvider.notifier).updateNote(ticker, note);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(note == null ? 'Note removed' : 'Note updated'),
              backgroundColor: AppColors.successGreen,
            ),
          );
        }
      },
    );
  }

  Future<void> _editTags(String ticker, List<String> currentTags) async {
    await showTagSelectorDialog(
      context: context,
      ticker: ticker,
      initialTags: currentTags,
      onSave: (tags) async {
        await ref.read(watchlistProvider.notifier).updateTags(ticker, tags);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Tags updated'),
              backgroundColor: AppColors.successGreen,
            ),
          );
        }
      },
    );
  }

  void _toggleTagFilter(String tag) {
    setState(() {
      if (_selectedTagFilters.contains(tag)) {
        _selectedTagFilters.remove(tag);
      } else {
        _selectedTagFilters.add(tag);
      }
    });
  }

  void _clearFilters() {
    setState(() {
      _selectedTagFilters.clear();
      _searchQuery = '';
      _searchController.clear();
    });
  }

  List<WatchlistItem> _getFilteredWatchlist(List<WatchlistItem> watchlist) {
    var filtered = watchlist;

    // Apply tag filters
    if (_selectedTagFilters.isNotEmpty) {
      filtered = ref.read(watchlistProvider.notifier).filterByTags(_selectedTagFilters);
    }

    // Apply search query
    if (_searchQuery.isNotEmpty) {
      filtered = ref.read(watchlistProvider.notifier).search(_searchQuery);
    }

    return filtered;
  }

  Widget _buildWatchlistItem(WatchlistItem item) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: tone(AppColors.darkCard, 0.5),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: tone(Colors.white, 0.08)),
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () {
            // Navigate to symbol detail page
            Navigator.of(context).push(
              MaterialPageRoute(
                builder: (context) => SymbolDetailPage(ticker: item.ticker),
              ),
            );
          },
          borderRadius: BorderRadius.circular(16),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                // Symbol Icon
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: tone(AppColors.primaryBlue, 0.2),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: AppColors.primaryBlue.withValues(alpha: 0.3),
                      width: 2,
                    ),
                  ),
                  child: Center(
                    child: Text(
                      item.ticker.substring(0, item.ticker.length > 2 ? 2 : 1),
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),

                // Symbol Info
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item.ticker,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 4),
                      if (item.hasSignal)
                        Text(
                          item.signal!,
                          style: TextStyle(
                            fontSize: 13,
                            color: AppColors.successGreen,
                          ),
                        )
                      else
                        Text(
                          'Added ${item.daysSinceAdded} ${item.daysSinceAdded == 1 ? "day" : "days"} ago',
                          style: const TextStyle(
                            fontSize: 13,
                            color: Colors.white70,
                          ),
                        ),
                      if (item.hasNote) ...[
                        const SizedBox(height: 4),
                        Text(
                          item.note!,
                          style: const TextStyle(
                            fontSize: 12,
                            color: Colors.white60,
                            fontStyle: FontStyle.italic,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ],
                  ),
                ),

                // Actions
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_forward_ios, size: 16),
                      color: Colors.white70,
                      onPressed: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (context) => SymbolDetailPage(ticker: item.ticker),
                          ),
                        );
                      },
                    ),
                    IconButton(
                      icon: const Icon(Icons.delete_outline),
                      color: Colors.red,
                      onPressed: () => _removeSymbol(item.ticker),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.bookmark_border,
              size: 80,
              color: Colors.white.withValues(alpha: 0.3),
            ),
            const SizedBox(height: 24),
            const Text(
              'Your Watchlist is Empty',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              'Add symbols to track your favorite stocks',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white.withValues(alpha: 0.7),
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: _showAddSymbolDialog,
              icon: const Icon(Icons.add),
              label: const Text('Add Symbol'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primaryBlue,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 16,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUnauthenticatedState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.lock_outline,
              size: 80,
              color: Colors.white.withValues(alpha: 0.3),
            ),
            const SizedBox(height: 24),
            const Text(
              'Sign In Required',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              'Sign in to save and sync your watchlist across devices',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white.withValues(alpha: 0.7),
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => const LoginPage(),
                  ),
                );
              },
              icon: const Icon(Icons.login),
              label: const Text('Sign In'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primaryBlue,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 16,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);
    final watchlist = ref.watch(watchlistProvider);

    return Scaffold(
      backgroundColor: const Color(0xFF0A0E27),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'Watchlist',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
        actions: [
          if (authState.isAuthenticated && watchlist.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.add),
              onPressed: _showAddSymbolDialog,
              tooltip: 'Add Symbol',
            ),
        ],
      ),
      body: !authState.isAuthenticated
          ? _buildUnauthenticatedState()
          : watchlist.isEmpty
              ? _buildEmptyState()
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    // Header Stats
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: tone(AppColors.darkCard, 0.5),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: tone(Colors.white, 0.08)),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          Column(
                            children: [
                              Text(
                                '${watchlist.length}',
                                style: const TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                              const Text(
                                'Symbols',
                                style: TextStyle(
                                  fontSize: 13,
                                  color: Colors.white70,
                                ),
                              ),
                            ],
                          ),
                          Container(
                            width: 1,
                            height: 40,
                            color: tone(Colors.white, 0.1),
                          ),
                          Column(
                            children: [
                              Text(
                                '${watchlist.where((item) => item.hasSignal).length}',
                                style: TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                  color: AppColors.successGreen,
                                ),
                              ),
                              const Text(
                                'With Signals',
                                style: TextStyle(
                                  fontSize: 13,
                                  color: Colors.white70,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 24),

                    // Watchlist Items
                    ...watchlist.map((item) => _buildWatchlistItem(item)),
                  ],
                ),
      floatingActionButton: authState.isAuthenticated && watchlist.isNotEmpty
          ? FloatingActionButton(
              onPressed: _showAddSymbolDialog,
              backgroundColor: AppColors.primaryBlue,
              child: const Icon(Icons.add),
            )
          : null,
    );
  }
}
