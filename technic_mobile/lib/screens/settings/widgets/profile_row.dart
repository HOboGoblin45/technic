/// Profile Row Widget
/// 
/// Displays a key-value pair in settings.
library;

import 'package:flutter/material.dart';

class ProfileRow extends StatelessWidget {
  final String label;
  final String value;
  
  const ProfileRow({
    super.key,
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Colors.white70,
              fontWeight: FontWeight.w600,
            ),
          ),
          const Spacer(),
          Text(value, style: const TextStyle(color: Colors.white)),
        ],
      ),
    );
  }
}
