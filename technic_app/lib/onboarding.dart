import 'package:flutter/material.dart';

import 'user_profile.dart';

class OnboardingFlow extends StatefulWidget {
  final Widget child;
  final UserProfileStore store;
  final ValueNotifier<bool> themeNotifier;
  final ValueNotifier<String> optionsNotifier;

  const OnboardingFlow({
    super.key,
    required this.child,
    required this.store,
    required this.themeNotifier,
    required this.optionsNotifier,
  });

  @override
  State<OnboardingFlow> createState() => _OnboardingFlowState();
}

class _OnboardingFlowState extends State<OnboardingFlow> {
  bool _completed = false;
  late String _riskProfile;
  late String _optionsMode;
  late String _timeHorizon;

  @override
  void initState() {
    super.initState();
    final p = widget.store.current.value;
    _riskProfile = p.riskProfile;
    _optionsMode = p.optionsMode;
    _timeHorizon = p.timeHorizon;
    _checkCompleted();
  }

  Future<void> _checkCompleted() async {
    final p = widget.store.current.value;
    if (p != UserProfile.defaults) {
      setState(() => _completed = true);
    }
  }

  Future<void> _finish() async {
    final profile = UserProfile(
      riskProfile: _riskProfile,
      optionsMode: _optionsMode,
      timeHorizon: _timeHorizon,
      themeMode: widget.store.current.value.themeMode,
    );
    await widget.store.save(profile);
    widget.themeNotifier.value = profile.themeMode == 'dark';
    widget.optionsNotifier.value = profile.optionsMode;
    setState(() => _completed = true);
  }

  @override
  Widget build(BuildContext context) {
    if (_completed) return widget.child;

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Welcome to Technic',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Let’s set up your investing style so the scanner and Copilot speak your language.',
              ),
              const SizedBox(height: 24),
              _sectionTitle('1. Risk profile'),
              Wrap(
                spacing: 8,
                children: [
                  _choiceChip('Conservative', 'conservative', _riskProfile,
                      (v) => setState(() => _riskProfile = v)),
                  _choiceChip('Balanced', 'balanced', _riskProfile,
                      (v) => setState(() => _riskProfile = v)),
                  _choiceChip('Aggressive', 'aggressive', _riskProfile,
                      (v) => setState(() => _riskProfile = v)),
                ],
              ),
              const SizedBox(height: 24),
              _sectionTitle('2. Time horizon'),
              Wrap(
                spacing: 8,
                children: [
                  _choiceChip('Short-term', 'short_term', _timeHorizon,
                      (v) => setState(() => _timeHorizon = v)),
                  _choiceChip('Swing', 'swing', _timeHorizon,
                      (v) => setState(() => _timeHorizon = v)),
                  _choiceChip('Position', 'position', _timeHorizon,
                      (v) => setState(() => _timeHorizon = v)),
                ],
              ),
              const SizedBox(height: 24),
              _sectionTitle('3. Options'),
              Wrap(
                spacing: 8,
                children: [
                  _choiceChip('Stocks only', 'stock_only', _optionsMode,
                      (v) => setState(() => _optionsMode = v)),
                  _choiceChip('Stocks + options', 'stock_plus_options',
                      _optionsMode, (v) => setState(() => _optionsMode = v)),
                ],
              ),
              const Spacer(),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _finish,
                  child: const Text('Finish setup'),
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'You can change these later in Settings.',
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionTitle(String text) {
    return Text(
      text,
      style: const TextStyle(
        fontSize: 16,
        fontWeight: FontWeight.w700,
      ),
    );
  }

  Widget _choiceChip(
    String label,
    String value,
    String current,
    void Function(String) onChanged,
  ) {
    final selected = value == current;
    return ChoiceChip(
      label: Text(label),
      selected: selected,
      onSelected: (_) => onChanged(value),
    );
  }
}
