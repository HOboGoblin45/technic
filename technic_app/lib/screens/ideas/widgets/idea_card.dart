/// Idea Card Widget
/// 
/// Displays a trade idea with sparkline and action buttons.
library;

import 'package:flutter/material.dart';
import '../../../models/idea.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';
import '../../../widgets/sparkline.dart';

class IdeaCard extends StatelessWidget {
  final Idea idea;
  final VoidCallback? onAskCopilot;
  final VoidCallback? onSave;
  
  const IdeaCard({
    super.key,
    required this.idea,
    this.onAskCopilot,
    this.onSave,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      idea.ticker,
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      idea.title,
                      style: const TextStyle(
                        fontSize: 14,
                        color: Colors.white70,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
                if (idea.sparkline.isNotEmpty)
                  SizedBox(
                    width: 80,
                    height: 40,
                    child: Sparkline(
                      data: idea.sparkline,
                      positive: idea.sparkline.last > idea.sparkline.first,
                      color: AppColors.skyBlue,
                    ),
                  ),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Why section
            Text(
              idea.meta,
              style: const TextStyle(
                fontSize: 13,
                color: Colors.white,
                height: 1.4,
              ),
            ),
            
            const SizedBox(height: 12),
            
            // Plan section
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: tone(Colors.white, 0.03),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: tone(Colors.white, 0.08)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.auto_graph,
                        size: 14,
                        color: tone(AppColors.skyBlue, 0.9),
                      ),
                      const SizedBox(width: 6),
                      const Text(
                        'Trade Plan',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: Colors.white70,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(
                    idea.plan,
                    style: const TextStyle(
                      fontSize: 12,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 12),
            
            // Actions
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: onAskCopilot,
                    icon: const Icon(Icons.chat_bubble_outline, size: 16),
                    label: const Text('Ask Copilot'),
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton.icon(
                  onPressed: onSave,
                  icon: const Icon(Icons.star_outline, size: 16),
                  label: const Text('Save'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
