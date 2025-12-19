/// Add Alert Dialog
/// 
/// Dialog for creating price alerts on watchlist symbols.
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../../models/price_alert.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

/// Show add alert dialog
Future<void> showAddAlertDialog({
  required BuildContext context,
  required String ticker,
  required Function(PriceAlert) onSave,
}) async {
  await showDialog(
    context: context,
    builder: (context) => _AddAlertDialog(
      ticker: ticker,
      onSave: onSave,
    ),
  );
}

class _AddAlertDialog extends StatefulWidget {
  final String ticker;
  final Function(PriceAlert) onSave;

  const _AddAlertDialog({
    required this.ticker,
    required this.onSave,
  });

  @override
  State<_AddAlertDialog> createState() => _AddAlertDialogState();
}

class _AddAlertDialogState extends State<_AddAlertDialog> {
  final _targetController = TextEditingController();
  final _noteController = TextEditingController();
  AlertType _selectedType = AlertType.priceAbove;
  String? _errorMessage;

  @override
  void dispose() {
    _targetController.dispose();
    _noteController.dispose();
    super.dispose();
  }

  void _handleSave() {
    // Validate input
    final targetText = _targetController.text.trim();
    if (targetText.isEmpty) {
      setState(() {
        _errorMessage = 'Please enter a target value';
      });
      return;
    }

    final targetValue = double.tryParse(targetText);
    if (targetValue == null || targetValue <= 0) {
      setState(() {
        _errorMessage = 'Please enter a valid number greater than 0';
      });
      return;
    }

    // Create alert
    final alert = PriceAlert.create(
      ticker: widget.ticker,
      type: _selectedType,
      targetValue: targetValue,
      note: _noteController.text.trim().isEmpty ? null : _noteController.text.trim(),
    );

    // Call onSave callback
    widget.onSave(alert);

    // Close dialog
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: AppColors.darkCard,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
      ),
      title: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: tone(AppColors.warningOrange, 0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              Icons.notifications_active,
              color: AppColors.warningOrange,
              size: 24,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Set Price Alert',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  widget.ticker,
                  style: TextStyle(
                    fontSize: 14,
                    color: AppColors.primaryBlue,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Alert Type Selector
            const Text(
              'Alert Type',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 8),
            ...AlertType.values.map((type) {
              return RadioListTile<AlertType>(
                title: Text(type.displayName),
                subtitle: Text(
                  _getAlertTypeDescription(type),
                  style: const TextStyle(
                    fontSize: 12,
                    color: Colors.white60,
                  ),
                ),
                value: type,
                // ignore: deprecated_member_use
                groupValue: _selectedType,
                // ignore: deprecated_member_use
                onChanged: (AlertType? value) {
                  if (value != null) {
                    setState(() {
                      _selectedType = value;
                      _errorMessage = null;
                    });
                  }
                },
                activeColor: AppColors.primaryBlue,
                contentPadding: EdgeInsets.zero,
              );
            }),

            const SizedBox(height: 16),

            // Target Value Input
            Text(
              _getTargetLabel(),
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _targetController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              inputFormatters: [
                FilteringTextInputFormatter.allow(RegExp(r'^\d+\.?\d{0,2}')),
              ],
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                hintText: _getTargetHint(),
                hintStyle: const TextStyle(color: Colors.white38),
                prefixIcon: Icon(
                  _selectedType == AlertType.percentChange
                      ? Icons.percent
                      : Icons.attach_money,
                  color: Colors.white70,
                ),
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
                errorBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Colors.red, width: 2),
                ),
                errorText: _errorMessage,
              ),
              onChanged: (value) {
                if (_errorMessage != null) {
                  setState(() {
                    _errorMessage = null;
                  });
                }
              },
            ),

            const SizedBox(height: 16),

            // Optional Note
            const Text(
              'Note (Optional)',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _noteController,
              maxLines: 2,
              maxLength: 100,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                hintText: 'Add a note about this alert...',
                hintStyle: const TextStyle(color: Colors.white38),
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
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        ElevatedButton.icon(
          onPressed: _handleSave,
          icon: const Icon(Icons.notifications_active),
          label: const Text('Set Alert'),
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.warningOrange,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ],
    );
  }

  String _getAlertTypeDescription(AlertType type) {
    switch (type) {
      case AlertType.priceAbove:
        return 'Alert when price rises above target';
      case AlertType.priceBelow:
        return 'Alert when price falls below target';
      case AlertType.percentChange:
        return 'Alert when price changes by target %';
    }
  }

  String _getTargetLabel() {
    switch (_selectedType) {
      case AlertType.priceAbove:
        return 'Target Price (Above)';
      case AlertType.priceBelow:
        return 'Target Price (Below)';
      case AlertType.percentChange:
        return 'Percent Change';
    }
  }

  String _getTargetHint() {
    switch (_selectedType) {
      case AlertType.priceAbove:
      case AlertType.priceBelow:
        return 'e.g., 150.00';
      case AlertType.percentChange:
        return 'e.g., 5.0';
    }
  }
}
