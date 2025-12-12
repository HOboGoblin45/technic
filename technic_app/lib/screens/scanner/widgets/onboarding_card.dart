/// Onboarding Card Widget
/// 
/// Welcome message and feature highlights for new users.
library;

import 'package:flutter/material.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class OnboardingCard extends StatelessWidget {
  final VoidCallback? onDismiss;
  
  const OnboardingCard({
    super.key,
    this.onDismiss,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            tone(AppColors.skyBlue, 0.15),
            tone(AppColors.darkDeep, 0.9),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: tone(AppColors.skyBlue, 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.waving_hand,
                color: AppColors.skyBlue,
                size: 24,
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Text(
                  'Welcome to Technic!',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
              if (onDismiss != null)
                IconButton(
                  onPressed: onDismiss,
                  icon: const Icon(Icons.close, size: 20),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                ),
            ],
          ),
          const SizedBox(height: 16),
          _featureRow(
            Icons.auto_graph,
            'Quantitative Scanner',
            'AI-powered stock analysis with technical and fundamental metrics',
          ),
          const SizedBox(height: 12),
          _featureRow(
            Icons.chat_bubble_outline,
            'AI Copilot',
            'Ask questions about any stock and get instant analysis',
          ),
          const SizedBox(height: 12),
          _featureRow(
            Icons.tune,
            'Custom Profiles',
            'Adjust risk tolerance and trading style to match your strategy',
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: tone(Colors.white, 0.05),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                Icon(
                  Icons.lightbulb_outline,
                  size: 16,
                  color: AppColors.skyBlue,
                ),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'Tap any result to see detailed analysis and trade plans',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white70,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _featureRow(IconData icon, String title, String description) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: tone(AppColors.skyBlue, 0.2),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, size: 16, color: AppColors.skyBlue),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                description,
                style: const TextStyle(
                  fontSize: 12,
                  color: Colors.white60,
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
