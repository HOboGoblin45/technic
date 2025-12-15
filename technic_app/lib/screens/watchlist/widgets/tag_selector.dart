import 'package:flutter/material.dart';

/// Widget for selecting and managing tags on watchlist items
class TagSelector extends StatefulWidget {
  final List<String> selectedTags;
  final Function(List<String>) onTagsChanged;

  const TagSelector({
    super.key,
    required this.selectedTags,
    required this.onTagsChanged,
  });

  @override
  State<TagSelector> createState() => _TagSelectorState();
}

class _TagSelectorState extends State<TagSelector> {
  late List<String> _selectedTags;
  final TextEditingController _customTagController = TextEditingController();

  // Predefined tags
  static const List<String> predefinedTags = [
    'earnings-play',
    'breakout',
    'dividend',
    'growth',
    'value',
    'momentum',
    'tech',
    'healthcare',
    'finance',
    'energy',
    'swing-trade',
    'day-trade',
    'long-term',
    'high-risk',
    'low-risk',
    'watchlist',
  ];

  @override
  void initState() {
    super.initState();
    _selectedTags = List.from(widget.selectedTags);
  }

  @override
  void dispose() {
    _customTagController.dispose();
    super.dispose();
  }

  void _toggleTag(String tag) {
    setState(() {
      if (_selectedTags.contains(tag)) {
        _selectedTags.remove(tag);
      } else {
        _selectedTags.add(tag);
      }
    });
    widget.onTagsChanged(_selectedTags);
  }

  void _addCustomTag() {
    final tag = _customTagController.text.trim().toLowerCase();
    if (tag.isNotEmpty && !_selectedTags.contains(tag)) {
      setState(() {
        _selectedTags.add(tag);
        _customTagController.clear();
      });
      widget.onTagsChanged(_selectedTags);
    }
  }

  void _removeTag(String tag) {
    setState(() {
      _selectedTags.remove(tag);
    });
    widget.onTagsChanged(_selectedTags);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    // Separate selected tags into predefined and custom
    final selectedPredefined = _selectedTags
        .where((tag) => predefinedTags.contains(tag))
        .toList();
    final selectedCustom = _selectedTags
        .where((tag) => !predefinedTags.contains(tag))
        .toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Selected tags section
        if (_selectedTags.isNotEmpty) ...[
          Text(
            'Selected Tags (${_selectedTags.length})',
            style: theme.textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _selectedTags.map((tag) {
              final isCustom = !predefinedTags.contains(tag);
              return Chip(
                label: Text(tag),
                deleteIcon: const Icon(Icons.close, size: 18),
                onDeleted: () => _removeTag(tag),
                backgroundColor: isCustom
                    ? theme.colorScheme.secondaryContainer
                    : theme.colorScheme.primaryContainer,
                labelStyle: TextStyle(
                  color: isCustom
                      ? theme.colorScheme.onSecondaryContainer
                      : theme.colorScheme.onPrimaryContainer,
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 16),
          const Divider(),
          const SizedBox(height: 16),
        ],

        // Predefined tags section
        Text(
          'Quick Tags',
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: predefinedTags.map((tag) {
            final isSelected = _selectedTags.contains(tag);
            return FilterChip(
              label: Text(tag),
              selected: isSelected,
              onSelected: (_) => _toggleTag(tag),
              selectedColor: theme.colorScheme.primaryContainer,
              checkmarkColor: theme.colorScheme.onPrimaryContainer,
            );
          }).toList(),
        ),
        const SizedBox(height: 16),

        // Custom tag input
        Text(
          'Add Custom Tag',
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _customTagController,
                decoration: InputDecoration(
                  hintText: 'Enter custom tag...',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: theme.colorScheme.surface,
                  prefixIcon: const Icon(Icons.label_outline),
                ),
                onSubmitted: (_) => _addCustomTag(),
              ),
            ),
            const SizedBox(width: 8),
            FilledButton.icon(
              onPressed: _addCustomTag,
              icon: const Icon(Icons.add, size: 18),
              label: const Text('Add'),
            ),
          ],
        ),
      ],
    );
  }
}

/// Dialog for managing tags
class TagSelectorDialog extends StatefulWidget {
  final String ticker;
  final List<String> initialTags;
  final Function(List<String>) onSave;

  const TagSelectorDialog({
    super.key,
    required this.ticker,
    required this.initialTags,
    required this.onSave,
  });

  @override
  State<TagSelectorDialog> createState() => _TagSelectorDialogState();
}

class _TagSelectorDialogState extends State<TagSelectorDialog> {
  late List<String> _currentTags;

  @override
  void initState() {
    super.initState();
    _currentTags = List.from(widget.initialTags);
  }

  void _handleSave() {
    widget.onSave(_currentTags);
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return AlertDialog(
      title: Row(
        children: [
          Icon(Icons.label, color: theme.colorScheme.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Manage Tags',
                  style: theme.textTheme.titleLarge,
                ),
                Text(
                  widget.ticker,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: theme.colorScheme.primary,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      content: SizedBox(
        width: double.maxFinite,
        child: SingleChildScrollView(
          child: TagSelector(
            selectedTags: _currentTags,
            onTagsChanged: (tags) {
              setState(() {
                _currentTags = tags;
              });
            },
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton.icon(
          onPressed: _handleSave,
          icon: const Icon(Icons.save, size: 18),
          label: const Text('Save'),
        ),
      ],
    );
  }
}

/// Helper function to show the tag selector dialog
Future<void> showTagSelectorDialog({
  required BuildContext context,
  required String ticker,
  required List<String> initialTags,
  required Function(List<String>) onSave,
}) {
  return showDialog(
    context: context,
    builder: (context) => TagSelectorDialog(
      ticker: ticker,
      initialTags: initialTags,
      onSave: onSave,
    ),
  );
}
