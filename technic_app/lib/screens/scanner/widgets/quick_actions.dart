/// Quick Actions Widget
/// 
/// Provides quick profile selection and randomize functionality.
library;

import 'package:flutter/material.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class QuickActions extends StatelessWidget {
  final VoidCallback? onConservative;
  final VoidCallback? onModerate;
  final VoidCallback? onAggressive;
  final VoidCallback? onRandomize;
  final bool advancedMode;
  final ValueChanged<bool>? onAdvancedModeChanged;
  
  const QuickActions({
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
                      activeTrackColor: AppColors.skyBlue,
                    ),
                  ],
                ),
            ],
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
    return OutlinedButton(
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
    );
  }
}
