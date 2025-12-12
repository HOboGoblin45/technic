/// My Ideas Page
/// 
/// Displays the user's saved/starred symbols from the watchlist.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/app_providers.dart';

class MyIdeasPage extends ConsumerWidget {
  const MyIdeasPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final items = ref.watch(watchlistProvider);

    if (items.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'No saved ideas yet.\nStar symbols in the scanner to add them here.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 16,
              color: Colors.white70,
            ),
          ),
        ),
      );
    }

    return ListView.builder(
      itemCount: items.length,
      itemBuilder: (context, index) {
        final item = items[index];
        return Card(
          margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 0),
          child: ListTile(
            leading: const Icon(Icons.star, color: Colors.amber),
            title: Text(
              item.ticker,
              style: const TextStyle(
                fontWeight: FontWeight.w700,
                fontSize: 16,
              ),
            ),
            subtitle: item.note != null
                ? Text(item.note!)
                : const Text('Saved from scanner'),
            trailing: IconButton(
              icon: const Icon(Icons.delete_outline),
              onPressed: () {
                ref.read(watchlistProvider.notifier).remove(item.ticker);
              },
              tooltip: 'Remove from My Ideas',
            ),
            onTap: () {
              // TODO: Navigate to symbol detail page when implemented
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('Symbol detail for ${item.ticker} coming soon'),
                  duration: const Duration(seconds: 1),
                ),
              );
            },
          ),
        );
      },
    );
  }
}
