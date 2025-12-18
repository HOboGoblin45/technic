/// Settings Page
/// 
/// User preferences, profile management, theme settings, and app information.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/app_providers.dart';
import '../../theme/app_colors.dart';
import '../../utils/helpers.dart';
import '../../widgets/section_header.dart';
import '../../widgets/info_card.dart';
import '../../widgets/pulse_badge.dart';
import 'widgets/profile_row.dart';
import '../auth/login_page.dart';

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
    final authState = ref.watch(authProvider);
    final copilotStatus = ref.watch(copilotStatusProvider);
    final user = authState.user;

    return ListView(
      children: [
        // Authentication Section
        if (!authState.isAuthenticated)
          InfoCard(
            title: 'Sign in to unlock all features',
            subtitle: 'Access your watchlist, saved scans, and sync preferences across devices.',
            child: Row(
              children: [
                ElevatedButton.icon(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (context) => const LoginPage(),
                      ),
                    );
                  },
                  icon: const Icon(Icons.login),
                  label: const Text('Sign In'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primaryBlue,
                    foregroundColor: Colors.white,
                  ),
                ),
              ],
            ),
          )
        else
          InfoCard(
            title: 'Account',
            subtitle: 'Signed in as ${user?.email ?? "User"}',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 48,
                      height: 48,
                      decoration: BoxDecoration(
                        color: tone(AppColors.primaryBlue, 0.2),
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(
                          color: AppColors.primaryBlue.withValues(alpha: 0.3),
                          width: 2,
                        ),
                      ),
                      child: Center(
                        child: Text(
                          (user?.name ?? 'U').substring(0, 1).toUpperCase(),
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            user?.name ?? 'User',
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          Text(
                            user?.email ?? '',
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () {
                          // Note: Profile editing feature planned for future release
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Profile editing coming soon'),
                            ),
                          );
                        },
                        icon: const Icon(Icons.edit_outlined),
                        label: const Text('Edit Profile'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.white70,
                          side: BorderSide(color: tone(Colors.white, 0.2)),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () async {
                          // Show confirmation dialog
                          final confirmed = await showDialog<bool>(
                            context: context,
                            builder: (context) => AlertDialog(
                              backgroundColor: AppColors.darkCard,
                              title: const Text('Sign Out'),
                              content: const Text(
                                'Are you sure you want to sign out?',
                              ),
                              actions: [
                                TextButton(
                                  onPressed: () => Navigator.pop(context, false),
                                  child: const Text('Cancel'),
                                ),
                                ElevatedButton(
                                  onPressed: () => Navigator.pop(context, true),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.red,
                                  ),
                                  child: const Text('Sign Out'),
                                ),
                              ],
                            ),
                          );

                          if (confirmed == true) {
                            await ref.read(authProvider.notifier).logout();
                            if (context.mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text('Signed out successfully'),
                                ),
                              );
                            }
                          }
                        },
                        icon: const Icon(Icons.logout),
                        label: const Text('Sign Out'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.red,
                          side: const BorderSide(color: Colors.red),
                        ),
                      ),
                    ),
                  ],
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

        // Appearance Section - REMOVED (Dark mode only)
        // Light mode has visibility issues, so we're keeping dark mode only

        const SizedBox(height: 16),

        // Legal Disclaimer Card
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: tone(AppColors.darkCard, 0.5),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: AppColors.warningOrange.withValues(alpha: 0.3),
              width: 1,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    color: AppColors.warningOrange,
                    size: 24,
                  ),
                  const SizedBox(width: 12),
                  const Text(
                    'Important Disclaimer',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Text(
                'Technic provides educational analysis and quantitative insights for informational purposes only. This app does not provide financial, investment, or trading advice.',
                style: TextStyle(
                  fontSize: 14,
                  height: 1.5,
                  color: tone(Colors.white, 0.8),
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Past performance does not guarantee future results. Trading and investing involve substantial risk of loss. Always consult with a licensed financial advisor before making investment decisions.',
                style: TextStyle(
                  fontSize: 14,
                  height: 1.5,
                  color: tone(Colors.white, 0.8),
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'By using this app, you acknowledge that you understand these risks and agree to use the information provided at your own discretion.',
                style: TextStyle(
                  fontSize: 13,
                  height: 1.5,
                  color: tone(Colors.white, 0.7),
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 32),

      ],

      );
  }
}
