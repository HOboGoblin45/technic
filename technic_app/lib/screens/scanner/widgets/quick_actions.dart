/// Quick Actions Widget
/// 
/// Provides quick profile selection and randomize functionality.
library;

import 'package:flutter/material.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class QuickActions extends StatelessWidget {
  final VoidCallback onRunScan;
  final VoidCallback? onConservative;
  final VoidCallback? onModerate;
  final VoidCallback? onAggressive;
  final VoidCallback? onRandomize;
  final bool advancedMode;
  final ValueChanged<bool>? onAdvancedModeChanged;
  
  const QuickActions({
    required this.onRunScan,
    super.key,
    this.onConservative,
    this.onModerate,
    this.onAggressive,
    this.onRandomize,
    this.advancedMode = false,
    this.onAdvancedModeChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.03),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: tone(Colors.white, 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Quick Profiles',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: Colors.white70,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _profileButton(
                  'Conservative',
                  Icons.shield,
                  Colors.green,
                  onConservative,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _profileButton(
                  'Moderate',
                  Icons.balance,
                  Colors.blue,
                  onModerate,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _profileButton(
                  'Aggressive',
                  Icons.rocket_launch,
                  Colors.orange,
                  onAggressive,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: onRandomize,
                  icon: const Icon(Icons.shuffle, size: 16),
                  label: const Text('Randomize'),
                ),
              ),
              const SizedBox(width: 12),
              if (onAdvancedModeChanged != null)
                Row(
                  children: [
                    const Text(
                      'Advanced',
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.white70,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Switch(
                      value: advancedMode,
                      onChanged: onAdvancedModeChanged,
                      activeTrackColor: AppColors.primaryBlue,
                    ),
                  ],
                ),
            ],
          ),
          const SizedBox(height: 12),
          // Run Scan Button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: onRunScan,
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primaryBlue,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              icon: const Icon(Icons.play_arrow, size: 24),
              label: const Text(
                'Run Scan',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _profileButton(
    String label,
    IconData icon,
    Color color,
    VoidCallback? onPressed,
  ) {
    // Tooltip messages for each profile
    String tooltip = '';
    if (label == 'Conservative') {
      tooltip = 'Position trading: 7.0+ rating, 180 days lookback';
    } else if (label == 'Moderate') {
      tooltip = 'Swing trading: 5.0+ rating, 90 days lookback';
    } else if (label == 'Aggressive') {
      tooltip = 'Day trading: 3.0+ rating, 30 days lookback';
    }
    
    return Tooltip(
      message: tooltip,
      child: OutlinedButton(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 12),
          side: BorderSide(color: tone(color, 0.5)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 20, color: color),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
