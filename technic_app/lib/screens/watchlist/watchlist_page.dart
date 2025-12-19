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
import 'widgets/add_alert_dialog.dart';
import '../../providers/alert_provider.dart';

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

  Future<void> _addAlert(String ticker) async {
    await showAddAlertDialog(
      context: context,
      ticker: ticker,
      onSave: (alert) async {
        await ref.read(alertProvider.notifier).addAlert(alert);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Alert set for $ticker'),
              backgroundColor: AppColors.warningOrange,
            ),
          );
        }
      },
    );
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
            Navigator.of(context).push(
              MaterialPageRoute(
                builder: (context) => SymbolDetailPage(ticker: item.ticker),
              ),
            );
          },
          borderRadius: BorderRadius.circular(16),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
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

                // Note
                if (item.hasNote) ...[
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: tone(Colors.white, 0.05),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.note, size: 14, color: Colors.white.withValues(alpha: 0.5)),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            item.note!,
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.white60,
                              fontStyle: FontStyle.italic,
                            ),
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],

                // Tags
                if (item.hasTags) ...[
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: item.tags.map((tag) {
                      return Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: AppColors.primaryBlue.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: AppColors.primaryBlue.withValues(alpha: 0.3),
                          ),
                        ),
                        child: Text(
                          tag,
                          style: TextStyle(
                            fontSize: 11,
                            color: AppColors.primaryBlue,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ],

                // Alerts Indicator
                Builder(
                  builder: (context) {
                    final alerts = ref.watch(alertProvider);
                    final tickerAlerts = alerts.where((a) => a.ticker == item.ticker && a.isActive).toList();
                    
                    if (tickerAlerts.isNotEmpty) {
                      return Column(
                        children: [
                          const SizedBox(height: 8),
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: AppColors.warningOrange.withValues(alpha: 0.2),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: AppColors.warningOrange.withValues(alpha: 0.3),
                              ),
                            ),
                            child: Row(
                              children: [
                                Icon(
                                  Icons.notifications_active,
                                  size: 14,
                                  color: AppColors.warningOrange,
                                ),
                                const SizedBox(width: 6),
                                Expanded(
                                  child: Text(
                                    '${tickerAlerts.length} active ${tickerAlerts.length == 1 ? "alert" : "alerts"}',
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: AppColors.warningOrange,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      );
                    }
                    return const SizedBox.shrink();
                  },
                ),

                // Action Buttons
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () => _editNote(item.ticker, item.note),
                        icon: Icon(
                          item.hasNote ? Icons.edit_note : Icons.note_add,
                          size: 16,
                        ),
                        label: Text(item.hasNote ? 'Edit Note' : 'Add Note'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.white70,
                          side: BorderSide(color: tone(Colors.white, 0.2)),
                          padding: const EdgeInsets.symmetric(vertical: 8),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () => _editTags(item.ticker, item.tags),
                        icon: Icon(
                          item.hasTags ? Icons.label : Icons.label_outline,
                          size: 16,
                        ),
                        label: Text(item.hasTags ? 'Edit Tags' : 'Add Tags'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.white70,
                          side: BorderSide(color: tone(Colors.white, 0.2)),
                          padding: const EdgeInsets.symmetric(vertical: 8),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  onPressed: () => _addAlert(item.ticker),
                  icon: const Icon(Icons.notifications_active, size: 16),
                  label: const Text('Set Alert'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppColors.warningOrange,
                    side: BorderSide(color: AppColors.warningOrange.withValues(alpha: 0.5)),
                    padding: const EdgeInsets.symmetric(vertical: 8),
                  ),
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
          if (authState.isAuthenticated && watchlist.isNotEmpty) ...[
            // Search Button
            IconButton(
              icon: const Icon(Icons.search),
              onPressed: () {
                showDialog(
                  context: context,
                  builder: (context) => AlertDialog(
                    backgroundColor: AppColors.darkCard,
                    title: const Text('Search Watchlist'),
                    content: TextField(
                      controller: _searchController,
                      autofocus: true,
                      style: const TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        hintText: 'Search by ticker or note...',
                        hintStyle: const TextStyle(color: Colors.white38),
                        prefixIcon: const Icon(Icons.search, color: Colors.white70),
                        filled: true,
                        fillColor: const Color(0xFF1A1F3A),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide.none,
                        ),
                      ),
                      onChanged: (value) {
                        setState(() {
                          _searchQuery = value;
                        });
                      },
                    ),
                    actions: [
                      TextButton(
                        onPressed: () {
                          setState(() {
                            _searchQuery = '';
                            _searchController.clear();
                          });
                          Navigator.pop(context);
                        },
                        child: const Text('Clear'),
                      ),
                      ElevatedButton(
                        onPressed: () => Navigator.pop(context),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primaryBlue,
                        ),
                        child: const Text('Search'),
                      ),
                    ],
                  ),
                );
              },
              tooltip: 'Search',
            ),
            // Filter Button
            IconButton(
              icon: Icon(
                Icons.filter_list,
                color: _selectedTagFilters.isNotEmpty ? AppColors.primaryBlue : null,
              ),
              onPressed: () async {
                final allTags = ref.read(watchlistProvider.notifier).getAllTags();
                await showDialog(
                  context: context,
                  builder: (dialogContext) => StatefulBuilder(
                    builder: (context, setDialogState) => AlertDialog(
                      backgroundColor: AppColors.darkCard,
                      title: const Text('Filter by Tags'),
                      content: SingleChildScrollView(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: allTags.isEmpty
                              ? [
                                  const Padding(
                                    padding: EdgeInsets.all(16.0),
                                    child: Text(
                                      'No tags available.\nAdd tags to watchlist items first.',
                                      style: TextStyle(color: Colors.white70),
                                      textAlign: TextAlign.center,
                                    ),
                                  ),
                                ]
                              : allTags.map((tag) {
                                  return CheckboxListTile(
                                    title: Text(tag),
                                    value: _selectedTagFilters.contains(tag),
                                    onChanged: (checked) {
                                      setState(() {
                                        if (checked == true) {
                                          _selectedTagFilters.add(tag);
                                        } else {
                                          _selectedTagFilters.remove(tag);
                                        }
                                      });
                                      setDialogState(() {});
                                    },
                                    activeColor: AppColors.primaryBlue,
                                  );
                                }).toList(),
                        ),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () {
                            setState(() {
                              _selectedTagFilters.clear();
                            });
                            setDialogState(() {});
                          },
                          child: const Text('Clear All'),
                        ),
                        ElevatedButton(
                          onPressed: () => Navigator.pop(dialogContext),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.primaryBlue,
                          ),
                          child: const Text('Done'),
                        ),
                      ],
                    ),
                  ),
                );
              },
              tooltip: 'Filter by Tags',
            ),
            IconButton(
              icon: const Icon(Icons.add),
              onPressed: _showAddSymbolDialog,
              tooltip: 'Add Symbol',
            ),
          ],
        ],
      ),
      body: !authState.isAuthenticated
          ? _buildUnauthenticatedState()
          : watchlist.isEmpty
              ? _buildEmptyState()
              : Builder(
                  builder: (context) {
                    final filteredWatchlist = _getFilteredWatchlist(watchlist);
                    final hasActiveFilters = _selectedTagFilters.isNotEmpty || _searchQuery.isNotEmpty;
                    
                    return ListView(
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

                        // Active Filters Indicator
                        if (hasActiveFilters) ...[
                          Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: AppColors.primaryBlue.withValues(alpha: 0.2),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(
                                color: AppColors.primaryBlue.withValues(alpha: 0.3),
                              ),
                            ),
                            child: Row(
                              children: [
                                Icon(
                                  Icons.filter_list,
                                  size: 16,
                                  color: AppColors.primaryBlue,
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    'Showing ${filteredWatchlist.length} of ${watchlist.length} symbols',
                                    style: TextStyle(
                                      fontSize: 13,
                                      color: AppColors.primaryBlue,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ),
                                TextButton(
                                  onPressed: _clearFilters,
                                  child: const Text('Clear'),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 12),
                        ],

                        // Watchlist Items
                        if (filteredWatchlist.isEmpty && hasActiveFilters)
                          Center(
                            child: Padding(
                              padding: const EdgeInsets.all(32.0),
                              child: Column(
                                children: [
                                  Icon(
                                    Icons.search_off,
                                    size: 64,
                                    color: Colors.white.withValues(alpha: 0.3),
                                  ),
                                  const SizedBox(height: 16),
                                  const Text(
                                    'No matching symbols',
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white,
                                    ),
                                  ),
                                  const SizedBox(height: 8),
                                  Text(
                                    'Try adjusting your filters',
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.white.withValues(alpha: 0.7),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          )
                        else
                          ...filteredWatchlist.map((item) => _buildWatchlistItem(item)),
                      ],
                    );
                  },
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
