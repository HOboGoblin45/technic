/// Scan Progress Overlay Widget
/// 
/// Shows real-time progress during scanning with estimated time remaining.
library;

import 'package:flutter/material.dart';
import 'dart:async';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class ScanProgressOverlay extends StatefulWidget {
  final String? progressMessage;
  final double? progress; // 0.0 to 1.0
  final VoidCallback? onCancel;
  final int? symbolsScanned;
  final int? totalSymbols;
  final DateTime? startTime;

  const ScanProgressOverlay({
    super.key,
    this.progressMessage,
    this.progress,
    this.onCancel,
    this.symbolsScanned,
    this.totalSymbols,
    this.startTime,
  });

  @override
  State<ScanProgressOverlay> createState() => _ScanProgressOverlayState();
}

class _ScanProgressOverlayState extends State<ScanProgressOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  Timer? _etaTimer;
  String _eta = 'Calculating...';

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat();
    _animation = CurvedAnimation(parent: _controller, curve: Curves.easeInOut);
    
    // Update ETA every second
    _etaTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {
          _eta = _calculateETA();
        });
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _etaTimer?.cancel();
    super.dispose();
  }

  String _calculateETA() {
    if (widget.startTime == null ||
        widget.symbolsScanned == null ||
        widget.totalSymbols == null ||
        widget.symbolsScanned == 0) {
      return 'Calculating...';
    }

    final elapsed = DateTime.now().difference(widget.startTime!);
    final symbolsRemaining = widget.totalSymbols! - widget.symbolsScanned!;
    
    if (symbolsRemaining <= 0) return 'Almost done...';

    final avgTimePerSymbol = elapsed.inMilliseconds / widget.symbolsScanned!;
    final estimatedRemainingMs = (avgTimePerSymbol * symbolsRemaining).round();
    
    final remainingSeconds = (estimatedRemainingMs / 1000).round();
    
    if (remainingSeconds < 60) {
      return '$remainingSeconds seconds';
    } else {
      final minutes = (remainingSeconds / 60).floor();
      final seconds = remainingSeconds % 60;
      return '$minutes min ${seconds}s';
    }
  }

  @override
  Widget build(BuildContext context) {
    final progress = widget.progress ?? 0.0;
    final hasProgress = widget.symbolsScanned != null && widget.totalSymbols != null;

    return Container(
      color: Colors.black.withValues(alpha: 0.85),
      child: Center(
        child: Container(
          margin: const EdgeInsets.all(32),
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: tone(Colors.white, 0.05),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: tone(AppColors.primaryBlue, 0.3)),
            boxShadow: [
              BoxShadow(
                color: AppColors.primaryBlue.withValues(alpha: 0.2),
                blurRadius: 20,
                spreadRadius: 2,
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Animated Scanner Icon
              FadeTransition(
                opacity: _animation,
                child: Icon(
                  Icons.radar,
                  size: 64,
                  color: AppColors.primaryBlue,
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Title
              const Text(
                'Scanning Markets',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                ),
              ),
              
              const SizedBox(height: 8),
              
              // Progress Message
              if (widget.progressMessage != null)
                Text(
                  widget.progressMessage!,
                  style: const TextStyle(
                    fontSize: 14,
                    color: Colors.white70,
                  ),
                  textAlign: TextAlign.center,
                ),
              
              const SizedBox(height: 24),
              
              // Progress Bar
              SizedBox(
                width: 280,
                child: Column(
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: LinearProgressIndicator(
                        value: progress > 0 ? progress : null,
                        minHeight: 8,
                        backgroundColor: tone(Colors.white, 0.1),
                        valueColor: AlwaysStoppedAnimation<Color>(
                          AppColors.primaryBlue,
                        ),
                      ),
                    ),
                    
                    const SizedBox(height: 12),
                    
                    // Progress Stats
                    if (hasProgress)
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            '${widget.symbolsScanned} / ${widget.totalSymbols} symbols',
                            style: const TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: Colors.white70,
                            ),
                          ),
                          Text(
                            '${(progress * 100).toStringAsFixed(0)}%',
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: AppColors.primaryBlue,
                            ),
                          ),
                        ],
                      ),
                  ],
                ),
              ),
              
              const SizedBox(height: 16),
              
              // ETA
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: tone(AppColors.primaryBlue, 0.15),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: tone(AppColors.primaryBlue, 0.3)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.schedule,
                      size: 16,
                      color: AppColors.primaryBlue,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'ETA: $_eta',
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Cancel Button
              if (widget.onCancel != null)
                OutlinedButton.icon(
                  onPressed: widget.onCancel,
                  icon: const Icon(Icons.close, size: 18),
                  label: const Text('Cancel Scan'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.red,
                    side: const BorderSide(color: Colors.red),
                  ),
                ),
              
              const SizedBox(height: 8),
              
              // Tips
              const Text(
                'Tip: Scans are cached for faster repeat runs',
                style: TextStyle(
                  fontSize: 11,
                  color: Colors.white38,
                  fontStyle: FontStyle.italic,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
