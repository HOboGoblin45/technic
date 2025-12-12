/// Preset Manager Widget
/// 
/// Manages saved scan presets (screens).
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../models/saved_screen.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class PresetManager extends ConsumerWidget {
  final List<SavedScreen> presets;
  final ValueChanged<SavedScreen>? onLoad;
  final ValueChanged<String>? onDelete;
  final VoidCallback? onSaveNew;
  
  const PresetManager({
    super.key,
    required this.presets,
    this.onLoad,
    this.onDelete,
    this.onSaveNew,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: tone(AppColors.darkDeep, 0.98),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Saved Presets',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                ),
              ),
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.close),
              ),
            ],
          ),
          const SizedBox(height: 20),
          
          // Save New Button
          if (onSaveNew != null)
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () {
                  Navigator.pop(context);
                  onSaveNew?.call();
                },
                icon: const Icon(Icons.add, size: 20),
                label: const Text('Save Current as Preset'),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  side: BorderSide(color: AppColors.skyBlue),
                ),
              ),
            ),
          
          if (presets.isEmpty) ...[
            const SizedBox(height: 40),
            Center(
              child: Column(
                children: [
                  Icon(
                    Icons.bookmark_border,
                    size: 48,
                    color: Colors.white38,
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'No saved presets yet',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.white60,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Save your favorite filter combinations',
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.white38,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 40),
          ] else ...[
            const SizedBox(height: 20),
            const Text(
              'Your Presets',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 12),
            ...presets.map((preset) => _presetCard(context, preset)),
          ],
        ],
      ),
    );
  }
  
  Widget _presetCard(BuildContext context, SavedScreen preset) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.03),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: tone(Colors.white, 0.08)),
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: tone(AppColors.skyBlue, 0.2),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(
            Icons.bookmark,
            size: 20,
            color: AppColors.skyBlue,
          ),
        ),
        title: Text(
          preset.name,
          style: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        subtitle: Text(
          _buildSubtitle(preset),
          style: const TextStyle(
            fontSize: 12,
            color: Colors.white60,
          ),
        ),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            IconButton(
              onPressed: () {
                Navigator.pop(context);
                onLoad?.call(preset);
              },
              icon: const Icon(Icons.play_arrow, size: 20),
              tooltip: 'Load',
            ),
            IconButton(
              onPressed: () {
                _confirmDelete(context, preset);
              },
              icon: const Icon(Icons.delete_outline, size: 20),
              tooltip: 'Delete',
            ),
          ],
        ),
      ),
    );
  }
  
  String _buildSubtitle(SavedScreen preset) {
    final parts = <String>[];
    
    if (preset.params?['trade_style']?.isNotEmpty ?? false) {
      parts.add(preset.params!['trade_style']!);
    }
    if (preset.params?['sector']?.isNotEmpty ?? false) {
      parts.add(preset.params!['sector']!);
    }
    if (preset.params?['lookback_days']?.isNotEmpty ?? false) {
      parts.add('${preset.params!['lookback_days']}d');
    }
    
    return parts.isEmpty ? preset.description : parts.join(' • ');
  }
  
  void _confirmDelete(BuildContext context, SavedScreen preset) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Preset?'),
        content: Text('Are you sure you want to delete "${preset.name}"?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context);
              onDelete?.call(preset.name);
            },
            child: const Text(
              'Delete',
              style: TextStyle(color: Colors.red),
            ),
          ),
        ],
      ),
    );
  }
}
