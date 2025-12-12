/// Settings Page
/// 
/// User preferences, profile management, theme settings, and app information.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/app_providers.dart';
import '../../services/local_store.dart';
import '../../theme/app_colors.dart';
import '../../utils/helpers.dart';
import '../../widgets/section_header.dart';
import '../../widgets/info_card.dart';
import '../../widgets/pulse_badge.dart';
import 'widgets/profile_row.dart';

class SettingsPage extends ConsumerWidget {
  const SettingsPage({super.key});

  /// Build hero banner widget
  Widget _buildHeroBanner(
    BuildContext context, {
    required String title,
    required String subtitle,
    required String badge,
    required Widget trailing,
    required Widget child,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: tone(AppColors.darkCard, 0.5),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: tone(Colors.white, 0.08)),
        boxShadow: [
          BoxShadow(
            color: tone(Colors.black, 0.15),
            blurRadius: 6,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.08),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: tone(Colors.white, 0.1)),
                ),
                child: Text(
                  badge,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              const Spacer(),
              trailing,
            ],
          ),
          const SizedBox(height: 12),
          Text(
            title,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userId = ref.watch(userIdProvider);
    final copilotStatus = ref.watch(copilotStatusProvider);
    final isDarkMode = ref.watch(themeModeProvider);

    return ListView(
      children: [
        // Sign In Section
        InfoCard(
          title: userId == null ? 'Sign in to sync' : 'Signed in as $userId',
          subtitle: 'Sync presets, streaks, and preferences across devices.',
          child: Row(
            children: [
              ElevatedButton.icon(
                onPressed: () async {
                  final messenger = ScaffoldMessenger.of(context);
                  await ref.read(userIdProvider.notifier).signIn('google_user');
                  messenger.showSnackBar(
                    const SnackBar(
                      content: Text('Signed in with Google (stub)'),
                    ),
                  );
                },
                icon: const Icon(Icons.login),
                label: const Text('Google'),
              ),
              const SizedBox(width: 8),
              OutlinedButton.icon(
                onPressed: () async {
                  final messenger = ScaffoldMessenger.of(context);
                  await ref.read(userIdProvider.notifier).signIn('apple_user');
                  messenger.showSnackBar(
                    const SnackBar(
                      content: Text('Signed in with Apple (stub)'),
                    ),
                  );
                },
                icon: const Icon(Icons.apple),
                label: const Text('Apple'),
              ),
              const SizedBox(width: 8),
              if (userId != null)
                TextButton(
                  onPressed: () async {
                    final messenger = ScaffoldMessenger.of(context);
                    await ref.read(userIdProvider.notifier).signOut();
                    messenger.showSnackBar(
                      const SnackBar(content: Text('Signed out')),
                    );
                  },
                  child: const Text('Sign out'),
                ),
            ],
          ),
        ),

        // Copilot Status Section
        if (copilotStatus != null)
          InfoCard(
            title: 'Copilot offline',
            subtitle: 'Cached guidance will display until the service recovers.',
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    copilotStatus,
                    style: const TextStyle(color: Colors.white70),
                  ),
                ),
                const SizedBox(width: 12),
                OutlinedButton(
                  onPressed: () {
                    ref.read(copilotPrefillProvider.notifier).state =
                        'Retry Copilot with the last question.';
                    ref.read(currentTabProvider.notifier).setTab(2);
                  },
                  child: const Text('Open Copilot'),
                ),
              ],
            ),
          ),

        // Profile Hero Banner
        _buildHeroBanner(
          context,
          title: 'Profile and preferences',
          subtitle: 'Preserve every setting across devices.',
          badge: 'Synced',
          trailing: TextButton.icon(
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Profile editing coming soon'),
                ),
              );
            },
            icon: const Icon(Icons.edit_outlined),
            label: const Text('Edit profile'),
          ),
          child: Row(
            children: const [
              PulseBadge(
                text: 'Advanced view on',
                color: AppColors.successGreen,
              ),
              SizedBox(width: 8),
              PulseBadge(
                text: 'Alerts enabled',
                color: AppColors.successGreen,
              ),
              SizedBox(width: 8),
              PulseBadge(
                text: 'Sessions persist',
                color: AppColors.successGreen,
              ),
            ],
          ),
        ),

        const SizedBox(height: 16),

        // Profile Section
        const SectionHeader('Profile', caption: 'Mode, risk, and universe'),
        InfoCard(
          title: 'Account',
          subtitle: 'Connect your profile and preferences',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 42,
                    height: 42,
                    decoration: BoxDecoration(
                      color: tone(AppColors.primaryBlue, 0.12),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(
                      Icons.person_outline,
                      color: Colors.white70,
                    ),
                  ),
                  const SizedBox(width: 10),
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Primary workspace',
                        style: TextStyle(fontWeight: FontWeight.w700),
                      ),
                      Text(
                        'Synced across devices',
                        style: TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const ProfileRow(label: 'Mode', value: 'Swing / Long-term'),
              const ProfileRow(label: 'Risk per trade', value: '1.0%'),
              const ProfileRow(label: 'Universe', value: 'US Equities'),
              const ProfileRow(label: 'Experience', value: 'Advanced'),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 6,
                children: const [
                  PulseBadge(
                    text: 'Dark mode',
                    color: AppColors.successGreen,
                  ),
                  PulseBadge(
                    text: 'Advanced view',
                    color: AppColors.successGreen,
                  ),
                  PulseBadge(
                    text: 'Session memory',
                    color: AppColors.successGreen,
                  ),
                ],
              ),
              const SizedBox(height: 8),
              const Text(
                'Data sources: Polygon/rest API; Copilot: OpenAI.',
                style: TextStyle(color: Colors.white70, fontSize: 12),
              ),
              const SizedBox(height: 4),
              const Text(
                'Your API keys are stored locally (not uploaded).',
                style: TextStyle(color: Colors.white70, fontSize: 12),
              ),
            ],
          ),
        ),

        // Appearance Section
        const SectionHeader(
          'Appearance',
          caption: 'Dark mode with Technic accent',
        ),
        InfoCard(
          title: 'Theme',
          subtitle: 'Institutional minimal with optional high contrast',
          child: Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: AppColors.primaryBlue,
                ),
              ),
              const SizedBox(width: 12),
              const Text(
                'Technic Dark',
                style: TextStyle(color: Colors.white),
              ),
              const Spacer(),
              Row(
                children: [
                  const Text(
                    'Dark mode',
                    style: TextStyle(color: Colors.white70),
                  ),
                  const SizedBox(width: 6),
                  Switch(
                    value: isDarkMode,
                    onChanged: (v) {
                      ref.read(themeModeProvider.notifier).setDarkMode(v);
                    },
                    thumbColor: const WidgetStatePropertyAll(AppColors.primaryBlue),
                    trackColor: WidgetStatePropertyAll(
                      tone(AppColors.primaryBlue, 0.3),
                    ),
                  ),
                ],
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.06),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Text(
                  'Sync across devices',
                  style: TextStyle(color: Colors.white70),
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 8),

        // Display Options
        InfoCard(
          title: 'Display options',
          subtitle: 'Toggle modes and accessibility presets',
          child: Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              Chip(
                label: const Text('Dark mode'),
                avatar: const Icon(
                  Icons.dark_mode,
                  size: 16,
                  color: Colors.white70,
                ),
                backgroundColor: tone(Colors.white, 0.05),
              ),
              Chip(
                label: const Text('Light mode'),
                avatar: const Icon(
                  Icons.light_mode_outlined,
                  size: 16,
                  color: Colors.white70,
                ),
                backgroundColor: tone(
                  AppColors.primaryBlue,
                  Theme.of(context).brightness == Brightness.dark ? 0.05 : 0.12,
                ),
              ),
              Chip(
                label: const Text('High contrast'),
                avatar: const Icon(
                  Icons.contrast,
                  size: 16,
                  color: Colors.white70,
                ),
                backgroundColor: tone(Colors.white, 0.05),
              ),
            ],
          ),
        ),

        // Data & Alerts Section
        const SectionHeader('Data & alerts', caption: 'Control intensity'),
        InfoCard(
          title: 'Notifications',
          subtitle: 'Goal progress, alerts, and data refresh cadence',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const ProfileRow(label: 'Goal tracking', value: 'On'),
              const ProfileRow(label: 'Scanner refresh', value: 'Every 60s'),
              const ProfileRow(label: 'Haptics', value: 'Subtle'),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 6,
                children: [
                  ActionChip(
                    label: const Text('Mute alerts'),
                    onPressed: () {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Mute alerts feature coming soon'),
                        ),
                      );
                    },
                    backgroundColor: tone(Colors.white, 0.05),
                  ),
                  ActionChip(
                    label: const Text('Set refresh to 30s'),
                    onPressed: () {
                      showDialog(
                        context: context,
                        builder: (ctx) => AlertDialog(
                          title: const Text('Refresh Rate'),
                          content: const Text(
                            'Choose refresh rate:\n\n• 30 seconds\n• 1 minute\n• 5 minutes',
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.pop(ctx),
                              child: const Text('Close'),
                            ),
                          ],
                        ),
                      );
                    },
                    backgroundColor: tone(Colors.white, 0.05),
                  ),
                ],
              ),
            ],
          ),
        ),

        const SizedBox(height: 12),

        // Data & Trust Section
        const InfoCard(
          title: 'Data & trust',
          subtitle: 'How scores and keys are handled',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Scores: trend, momentum, volatility, and risk signals combined.',
              ),
              SizedBox(height: 4),
              Text('Data sources: Polygon/rest API; Copilot: OpenAI.'),
              SizedBox(height: 4),
              Text('Your API keys are stored locally (not uploaded).'),
            ],
          ),
        ),

        const SizedBox(height: 12),

        // Activity Stats Section
        FutureBuilder<Map<String, dynamic>?>(
          future: LocalStore.loadScannerState(),
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const LinearProgressIndicator(minHeight: 2);
            }

            final data = snapshot.data;
            if (data == null) return const SizedBox.shrink();

            final scanCount = data['scanCount'] as int? ?? 0;
            final streak = data['streakDays'] as int? ?? 0;
            final savedPresets =
                (data['saved_screens'] as List?)?.length ?? 0;
            final filters = Map<String, String>.from(data['filters'] as Map);
            final sectors = (filters['sectors'] ?? '')
                .split(',')
                .where((e) => e.trim().isNotEmpty)
                .toList();
            final lastScanStr = data['lastScan'] as String?;
            final lastScan =
                lastScanStr != null ? DateTime.tryParse(lastScanStr) : null;

            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                InfoCard(
                  title: 'Your month in technic',
                  subtitle: 'Activity recap',
                  child: Row(
                    children: [
                      Chip(
                        label: Text('Scans: $scanCount'),
                        backgroundColor: tone(Colors.white, 0.05),
                      ),
                      const SizedBox(width: 8),
                      Chip(
                        label: Text('Streak: $streak d'),
                        backgroundColor: tone(AppColors.primaryBlue, 0.15),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          sectors.isEmpty
                              ? 'Top sector: All'
                              : 'Top sectors: ${sectors.join(', ')}',
                          style: const TextStyle(color: Colors.white70),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Chip(
                        label: Text('Presets: $savedPresets'),
                        backgroundColor: tone(Colors.white, 0.05),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                if (lastScan != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    child: Text(
                      'Last scan: ${lastScan.toLocal().toString().split('.').first} • ${DateTime.now().difference(lastScan).inDays}d ago',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                    ),
                  ),
                if (lastScan != null) const SizedBox(height: 8),
                InfoCard(
                  title: 'Achievements',
                  subtitle: 'Celebrate streaks and progress',
                  child: Wrap(
                    spacing: 8,
                    runSpacing: 6,
                    children: [
                      Chip(
                        label: Text(
                          scanCount >= 5
                              ? 'Starter: 5 scans'
                              : 'Next: 5 scans',
                        ),
                        backgroundColor: scanCount >= 5
                            ? tone(AppColors.primaryBlue, 0.2)
                            : tone(Colors.white, 0.05),
                      ),
                      Chip(
                        label: Text(
                          scanCount >= 10
                              ? 'Builder: 10 scans'
                              : 'Next: 10 scans',
                        ),
                        backgroundColor: scanCount >= 10
                            ? tone(AppColors.primaryBlue, 0.2)
                            : tone(Colors.white, 0.05),
                      ),
                      Chip(
                        label: Text(
                          streak >= 3
                              ? 'Streak 3 days'
                              : 'Keep a 3-day streak',
                        ),
                        backgroundColor: streak >= 3
                            ? tone(AppColors.primaryBlue, 0.2)
                            : tone(Colors.white, 0.05),
                      ),
                      Chip(
                        label: Text(
                          streak >= 7
                              ? 'Streak 7 days'
                              : 'Keep a 7-day streak',
                        ),
                        backgroundColor: streak >= 7
                            ? tone(AppColors.primaryBlue, 0.2)
                            : tone(Colors.white, 0.05),
                      ),
                      Chip(
                        label: Text(
                          savedPresets >= 3
                              ? 'Preset pro: 3 saved'
                              : 'Next: save 3 presets',
                        ),
                        backgroundColor: savedPresets >= 3
                            ? tone(AppColors.primaryBlue, 0.2)
                            : tone(Colors.white, 0.05),
                      ),
                    ],
                  ),
                ),
              ],
            );
          },
        ),

        const SizedBox(height: 12),

        // Copilot Status Card
        InfoCard(
          title: 'Copilot status',
          subtitle: copilotStatus == null
              ? 'Online. Answers will return live.'
              : 'Offline. Showing cached guidance until service recovers.',
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Flexible(
                child: Text(
                  copilotStatus ?? 'All systems go.',
                  style: const TextStyle(color: Colors.white70),
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton.icon(
                onPressed: () {
                  ref.read(copilotPrefillProvider.notifier).state =
                      'Check Copilot status';
                  ref.read(currentTabProvider.notifier).setTab(2);
                },
                icon: const Icon(Icons.chat_bubble_outline),
                label: const Text('Open Copilot'),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
