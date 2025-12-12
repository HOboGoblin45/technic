/// Message Bubble Widget
/// 
/// Displays a chat message bubble for Copilot conversations.
library;

import 'package:flutter/material.dart';
import '../../../models/copilot_message.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class MessageBubble extends StatelessWidget {
  final CopilotMessage message;
  
  const MessageBubble({
    super.key,
    required this.message,
  });

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == 'user';
    
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment:
            isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: tone(AppColors.primaryBlue, 0.15),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Icon(
                Icons.auto_awesome,
                size: 16,
                color: Colors.white,
              ),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: isUser
                    ? tone(AppColors.primaryBlue, 0.15)
                    : tone(Colors.white, 0.04),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: isUser
                      ? tone(AppColors.primaryBlue, 0.3)
                      : tone(Colors.white, 0.08),
                ),
              ),
              child: Text(
                message.body,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  height: 1.4,
                ),
              ),
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 8),
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: tone(Colors.white, 0.08),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Icon(
                Icons.person_outline,
                size: 16,
                color: Colors.white70,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
