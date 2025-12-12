/// Copilot Page
/// 
/// AI-powered chat interface for market analysis and trade guidance.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/copilot_message.dart';
import '../../providers/app_providers.dart';
import '../../theme/app_colors.dart';
import '../../utils/helpers.dart';
import '../../utils/mock_data.dart';
import '../../widgets/info_card.dart';
import 'widgets/message_bubble.dart';

class CopilotPage extends ConsumerStatefulWidget {
  const CopilotPage({super.key});

  @override
  ConsumerState<CopilotPage> createState() => _CopilotPageState();
}

class _CopilotPageState extends ConsumerState<CopilotPage>
    with AutomaticKeepAliveClientMixin {
  final TextEditingController _controller = TextEditingController();
  final List<CopilotMessage> _messages = List.of(copilotMessages);
  bool _sending = false;
  bool _copilotError = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  bool get wantKeepAlive => true;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final prefill = ref.read(copilotPrefillProvider);
    if (prefill != null && _controller.text.isEmpty) {
      _controller.text = prefill;
      ref.read(copilotPrefillProvider.notifier).state = null;
    }
  }

  Future<void> _sendPrompt([String? prompt]) async {
    final text = prompt ?? _controller.text.trim();
    if (text.isEmpty || _sending) return;

    setState(() {
      _sending = true;
      _messages.add(CopilotMessage('user', text));
      _controller.clear();
    });

    try {
      final apiService = ref.read(apiServiceProvider);
      final context = ref.read(copilotContextProvider);
      
      final reply = await apiService.sendCopilot(
        text,
        symbol: context?.ticker,
      );
      
      if (!mounted) return;
      
      setState(() {
        _messages.add(reply);
        _sending = false;
        _copilotError = false;
      });
      
      ref.read(copilotStatusProvider.notifier).state = null;
    } catch (e) {
      if (!mounted) return;
      
      setState(() {
        _sending = false;
        _copilotError = true;
      });
      
      ref.read(copilotStatusProvider.notifier).state = e.toString();
      
      _messages.add(
        const CopilotMessage(
          'assistant',
          'Copilot is temporarily offline. Showing cached guidance instead.',
        ),
      );
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Copilot unavailable: $e')),
        );
      }
    }
  }

  Widget _buildHeroBanner({
    required String title,
    required String subtitle,
    required String badge,
    required Widget trailing,
    required Widget child,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            tone(AppColors.skyBlue, 0.12),
            tone(AppColors.darkDeep, 0.9),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: tone(Colors.white, 0.08)),
        boxShadow: [
          BoxShadow(
            color: tone(Colors.black, 0.35),
            blurRadius: 18,
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
  Widget build(BuildContext context) {
    super.build(context);
    
    final copilotStatus = ref.watch(copilotStatusProvider);
    final copilotContext = ref.watch(copilotContextProvider);
    final copilotPrefill = ref.watch(copilotPrefillProvider);

    return ListView(
      children: [
        // Offline Status Banner
        if (copilotStatus != null)
          InfoCard(
            title: 'Copilot offline',
            subtitle: 'Cached guidance will display until service recovers.',
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
                        'Retry the last question';
                    ref.read(currentTabProvider.notifier).setTab(2);
                  },
                  child: const Text('Retry'),
                ),
              ],
            ),
          ),

        // Hero Banner
        _buildHeroBanner(
          title: 'Quant Copilot',
          subtitle: 'Context-aware chat with structured answers.',
          badge: 'Conversational',
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              OutlinedButton.icon(
                onPressed: _sending ? null : () => _sendPrompt('Voice request'),
                icon: const Icon(Icons.mic_none),
                label: const Text('Voice'),
              ),
              const SizedBox(width: 8),
              TextButton.icon(
                onPressed: () {},
                icon: const Icon(Icons.note_alt_outlined),
                label: const Text('Notes'),
              ),
            ],
          ),
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: copilotPrompts
                  .map(
                    (p) => Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ActionChip(
                        backgroundColor: tone(Colors.white, 0.04),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                          side: BorderSide(color: tone(Colors.white, 0.06)),
                        ),
                        label: Text(p, style: const TextStyle(color: Colors.white)),
                        onPressed: () => _sendPrompt(p),
                      ),
                    ),
                  )
                  .toList(),
            ),
          ),
        ),

        const SizedBox(height: 12),

        // Context Card
        if (copilotContext != null)
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 0, vertical: 8),
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            copilotContext.ticker,
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            '${copilotContext.signal} • ${copilotContext.playStyle ?? "Swing"}',
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.white70,
                            ),
                          ),
                          if (copilotContext.institutionalCoreScore != null)
                            Text(
                              'ICS ${copilotContext.institutionalCoreScore!.toStringAsFixed(0)}/100'
                              '${copilotContext.icsTier != null ? " (${copilotContext.icsTier})" : ""}',
                              style: TextStyle(
                                fontSize: 12,
                                color: tone(AppColors.skyBlue, 0.9),
                              ),
                            ),
                        ],
                      ),
                      IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          ref.read(copilotContextProvider.notifier).state = null;
                        },
                        tooltip: 'Clear context',
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Win ~${copilotContext.winProb10d != null ? (copilotContext.winProb10d! * 100).toStringAsFixed(0) : "--"}% • '
                    'Quality ${copilotContext.qualityScore?.toStringAsFixed(1) ?? "--"} • '
                    'ATR ${(copilotContext.atrPct != null ? (copilotContext.atrPct! * 100).toStringAsFixed(1) : "--")}%',
                    style: const TextStyle(fontSize: 11, color: Colors.white70),
                  ),
                ],
              ),
            ),
          ),

        const SizedBox(height: 16),

        // Error State
        if (_copilotError)
          InfoCard(
            title: 'Copilot unavailable',
            subtitle: 'Check your connection or retry shortly.',
            child: TextButton.icon(
              onPressed: _sending ? null : () => _sendPrompt(),
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ),

        // Empty State
        if (_messages.isEmpty && !_copilotError)
          InfoCard(
            title: 'Start a Copilot session',
            subtitle: 'Ask a question to begin. Voice coming soon.',
            child: const Text(
              'Examples: "Summarize today\'s scan", "Explain risk on NVDA setup", "Compare TSLA vs AAPL momentum".',
              style: TextStyle(color: Colors.white70),
            ),
          ),

        // Conversation
        InfoCard(
          title: 'Conversation',
          subtitle: 'Persist responses with tables, bullets, and calls to action.',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Messages
              ..._messages.map((m) => MessageBubble(message: m)),
              
              const SizedBox(height: 12),

              // Input Area
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: tone(Colors.white, 0.02),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: tone(Colors.white, 0.05)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Suggestion
                    if (copilotPrefill != null && copilotPrefill.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 8),
                        child: Row(
                          children: [
                            Expanded(
                              child: Text(
                                copilotPrefill,
                                style: const TextStyle(
                                  color: Colors.white70,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                            TextButton(
                              onPressed: () {
                                _controller.text = copilotPrefill;
                                ref.read(copilotPrefillProvider.notifier).state = null;
                              },
                              child: const Text('Use suggestion'),
                            ),
                          ],
                        ),
                      ),

                    // Text Field
                    TextField(
                      controller: _controller,
                      maxLines: 4,
                      minLines: 2,
                      decoration: InputDecoration(
                        hintText: 'Type your question...',
                        border: InputBorder.none,
                        suffixIcon: _sending
                            ? const Padding(
                                padding: EdgeInsets.all(10),
                                child: SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                ),
                              )
                            : IconButton(
                                icon: const Icon(Icons.send),
                                onPressed: _sending ? null : () => _sendPrompt(),
                              ),
                      ),
                      onSubmitted: (_) => _sendPrompt(),
                    ),

                    const SizedBox(height: 8),

                    // Disclaimer
                    Text(
                      'Copilot explains what Technic sees in this setup and outlines an example trade. '
                      'Responses are educational, not financial advice.',
                      style: TextStyle(
                        color: tone(Colors.white, 0.6),
                        fontSize: 12,
                      ),
                    ),

                    const SizedBox(height: 8),

                    // Actions
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.memory,
                              size: 14,
                              color: tone(Colors.white, 0.6),
                            ),
                            const SizedBox(width: 6),
                            Text(
                              'Session memory on',
                              style: TextStyle(color: tone(Colors.white, 0.7)),
                            ),
                          ],
                        ),
                        ElevatedButton.icon(
                          onPressed: _sending ? null : _sendPrompt,
                          icon: _sending
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Icon(Icons.send),
                          label: Text(_sending ? 'Sending...' : 'Send to Copilot'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
