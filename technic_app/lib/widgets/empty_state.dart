/// Empty State Widget
/// 
/// Shows helpful empty states with illustrations and action buttons
library;

import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

/// Empty state types
enum EmptyStateType {
  watchlist,
  scanResults,
  noInternet,
  searchResults,
  history,
  general,
}

/// Empty state widget with icon, message, and action button
class EmptyState extends StatelessWidget {
  final EmptyStateType type;
  final String? title;
  final String? message;
  final String? actionLabel;
  final VoidCallback? onAction;
  final IconData? customIcon;

  const EmptyState({
    super.key,
    required this.type,
    this.title,
    this.message,
    this.actionLabel,
    this.onAction,
    this.customIcon,
  });

  /// Empty watchlist state
  factory EmptyState.watchlist({VoidCallback? onAddSymbol}) {
    return EmptyState(
      type: EmptyStateType.watchlist,
      title: 'Your Watchlist is Empty',
      message: 'Add symbols to track your favorite stocks and get notified of opportunities.',
      actionLabel: 'Add Symbol',
      onAction: onAddSymbol,
    );
  }

  /// No scan results state
  factory EmptyState.scanResults({VoidCallback? onAdjustFilters}) {
    return EmptyState(
      type: EmptyStateType.scanResults,
      title: 'No Results Found',
      message: 'Try adjusting your filters or scan parameters to find more opportunities.',
      actionLabel: 'Adjust Filters',
      onAction: onAdjustFilters,
    );
  }

  /// No internet connection state
  factory EmptyState.noInternet({VoidCallback? onRetry}) {
    return EmptyState(
      type: EmptyStateType.noInternet,
      title: 'No Internet Connection',
      message: 'Please check your connection and try again.',
      actionLabel: 'Retry',
      onAction: onRetry,
    );
  }

  /// No search results state
  factory EmptyState.searchResults({required String query, VoidCallback? onClear}) {
    return EmptyState(
      type: EmptyStateType.searchResults,
      title: 'No Results for "$query"',
      message: 'Try a different search term or check your spelling.',
      actionLabel: 'Clear Search',
      onAction: onClear,
    );
  }

  /// No history state
  factory EmptyState.history({VoidCallback? onStartScan}) {
    return EmptyState(
      type: EmptyStateType.history,
      title: 'No Scan History',
      message: 'Your scan history will appear here once you run your first scan.',
      actionLabel: 'Run Scan',
      onAction: onStartScan,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Illustration/Icon
            Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                color: _getIconBackgroundColor().withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                customIcon ?? _getIcon(),
                size: 64,
                color: _getIconColor(),
              ),
            ),
            const SizedBox(height: 32),

            // Title
            Text(
              title ?? _getDefaultTitle(),
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),

            // Message
            Text(
              message ?? _getDefaultMessage(),
              style: const TextStyle(
                fontSize: 16,
                color: Colors.white70,
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),

            // Action button
            if (actionLabel != null && onAction != null) ...[
              const SizedBox(height: 32),
              ElevatedButton(
                onPressed: onAction,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 16,
                  ),
                  backgroundColor: AppColors.primaryBlue,
                ),
                child: Text(actionLabel!),
              ),
            ],
          ],
        ),
      ),
    );
  }

  IconData _getIcon() {
    switch (type) {
      case EmptyStateType.watchlist:
        return Icons.bookmark_outline;
      case EmptyStateType.scanResults:
        return Icons.search_off;
      case EmptyStateType.noInternet:
        return Icons.wifi_off;
      case EmptyStateType.searchResults:
        return Icons.search_off;
      case EmptyStateType.history:
        return Icons.history;
      case EmptyStateType.general:
        return Icons.inbox_outlined;
    }
  }

  Color _getIconColor() {
    switch (type) {
      case EmptyStateType.watchlist:
        return AppColors.primaryBlue;
      case EmptyStateType.scanResults:
        return Colors.grey;
      case EmptyStateType.noInternet:
        return Colors.orange;
      case EmptyStateType.searchResults:
        return Colors.grey;
      case EmptyStateType.history:
        return AppColors.primaryBlue;
      case EmptyStateType.general:
        return Colors.grey;
    }
  }

  Color _getIconBackgroundColor() {
    return _getIconColor();
  }

  String _getDefaultTitle() {
    switch (type) {
      case EmptyStateType.watchlist:
        return 'Your Watchlist is Empty';
      case EmptyStateType.scanResults:
        return 'No Results Found';
      case EmptyStateType.noInternet:
        return 'No Internet Connection';
      case EmptyStateType.searchResults:
        return 'No Results';
      case EmptyStateType.history:
        return 'No History';
      case EmptyStateType.general:
        return 'Nothing Here';
    }
  }

  String _getDefaultMessage() {
    switch (type) {
      case EmptyStateType.watchlist:
        return 'Add symbols to track your favorite stocks.';
      case EmptyStateType.scanResults:
        return 'Try adjusting your filters.';
      case EmptyStateType.noInternet:
        return 'Please check your connection.';
      case EmptyStateType.searchResults:
        return 'Try a different search term.';
      case EmptyStateType.history:
        return 'Your history will appear here.';
      case EmptyStateType.general:
        return 'Nothing to show yet.';
    }
  }
}

/// Compact empty state for smaller spaces
class CompactEmptyState extends StatelessWidget {
  final IconData icon;
  final String message;
  final Color? iconColor;

  const CompactEmptyState({
    super.key,
    required this.icon,
    required this.message,
    this.iconColor,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 48,
              color: iconColor ?? Colors.white38,
            ),
            const SizedBox(height: 16),
            Text(
              message,
              style: const TextStyle(
                fontSize: 14,
                color: Colors.white54,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
