import 'package:flutter/material.dart';

/// Dialog for adding or editing notes on watchlist items
class AddNoteDialog extends StatefulWidget {
  final String ticker;
  final String? initialNote;
  final Function(String?) onSave;

  const AddNoteDialog({
    super.key,
    required this.ticker,
    this.initialNote,
    required this.onSave,
  });

  @override
  State<AddNoteDialog> createState() => _AddNoteDialogState();
}

class _AddNoteDialogState extends State<AddNoteDialog> {
  late final TextEditingController _controller;
  static const int maxCharacters = 500;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialNote ?? '');
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _handleSave() {
    final note = _controller.text.trim();
    widget.onSave(note.isEmpty ? null : note);
    Navigator.of(context).pop();
  }

  void _handleClear() {
    setState(() {
      _controller.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final characterCount = _controller.text.length;
    final isOverLimit = characterCount > maxCharacters;

    return AlertDialog(
      title: Row(
        children: [
          Icon(Icons.note_add, color: theme.colorScheme.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.initialNote == null ? 'Add Note' : 'Edit Note',
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
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              controller: _controller,
              maxLines: 6,
              maxLength: maxCharacters,
              decoration: InputDecoration(
                hintText: 'Add your trading notes here...\n\nExamples:\n• Watching for earnings beat\n• Breakout above \$150\n• Strong momentum',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                filled: true,
                fillColor: theme.colorScheme.surface,
                counterText: '',
              ),
              onChanged: (_) => setState(() {}),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '$characterCount / $maxCharacters characters',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: isOverLimit
                        ? theme.colorScheme.error
                        : theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                if (_controller.text.isNotEmpty)
                  TextButton.icon(
                    onPressed: _handleClear,
                    icon: const Icon(Icons.clear, size: 16),
                    label: const Text('Clear'),
                    style: TextButton.styleFrom(
                      foregroundColor: theme.colorScheme.error,
                    ),
                  ),
              ],
            ),
            if (isOverLimit)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  'Note is too long. Please shorten it.',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.error,
                  ),
                ),
              ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton.icon(
          onPressed: isOverLimit ? null : _handleSave,
          icon: const Icon(Icons.save, size: 18),
          label: const Text('Save'),
        ),
      ],
    );
  }
}

/// Helper function to show the add note dialog
Future<void> showAddNoteDialog({
  required BuildContext context,
  required String ticker,
  String? initialNote,
  required Function(String?) onSave,
}) {
  return showDialog(
    context: context,
    builder: (context) => AddNoteDialog(
      ticker: ticker,
      initialNote: initialNote,
      onSave: onSave,
    ),
  );
}
