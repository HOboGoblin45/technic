/// Premium Fundamentals Card Widget
/// 
/// Professional fundamentals display with:
/// - Glass morphism design
/// - Animated value reveals
/// - Color-coded indicators
/// - Comparison visualizations
library;

import 'package:flutter/material.dart';
import 'dart:ui';

import '../../../theme/app_colors.dart';
import '../../../utils/formatters.dart';
import '../../../models/symbol_detail.dart';

/// Premium fundamentals card with glass morphism
class PremiumFundamentalsCard extends StatefulWidget {
  final Fundamentals fundamentals;

  const PremiumFundamentalsCard({
    super.key,
    required this.fundamentals,
  });

  @override
  State<PremiumFundamentalsCard> createState() => _PremiumFundamentalsCardState();
}

class _PremiumFundamentalsCardState extends State<PremiumFundamentalsCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    
    _controller = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.0, 0.6, curve: Curves.easeOut),
    ));
    
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    ));
    
    _controller.forward();
  }
  
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: SlideTransition(
        position: _slideAnimation,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Section header
            Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Row(
                children: [
                  Container(
                    width: 4,
                    height: 24,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          AppColors.primaryBlue,
                          AppColors.primaryBlue.withValues(alpha: 0.3),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  const SizedBox(width: 12),
                  const Text(
                    'Fundamentals',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      letterSpacing: -0.5,
                    ),
                  ),
                ],
              ),
            ),
            
            // Glass morphism card
            Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    AppColors.primaryBlue.withValues(alpha: 0.1),
                    AppColors.primaryBlue.withValues(alpha: 0.05),
                  ],
                ),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(
                  color: Colors.white.withValues(alpha: 0.1),
                  width: 1,
                ),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        if (widget.fundamentals.pe != null)
                          _buildFundamentalRow(
                            'P/E Ratio',
                            widget.fundamentals.pe!.toStringAsFixed(2),
                            Icons.trending_up,
                            _getPEColor(widget.fundamentals.pe!),
                          ),
                        if (widget.fundamentals.pe != null && widget.fundamentals.eps != null)
                          _buildDivider(),
                        if (widget.fundamentals.eps != null)
                          _buildFundamentalRow(
                            'EPS',
                            formatCurrency(widget.fundamentals.eps!),
                            Icons.attach_money,
                            widget.fundamentals.eps! > 0 ? AppColors.successGreen : AppColors.dangerRed,
                          ),
                        if (widget.fundamentals.eps != null && widget.fundamentals.roe != null)
                          _buildDivider(),
                        if (widget.fundamentals.roe != null)
                          _buildFundamentalRow(
                            'ROE',
                            '${widget.fundamentals.roe!.toStringAsFixed(1)}%',
                            Icons.percent,
                            _getROEColor(widget.fundamentals.roe!),
                          ),
                        if (widget.fundamentals.roe != null && widget.fundamentals.debtToEquity != null)
                          _buildDivider(),
                        if (widget.fundamentals.debtToEquity != null)
                          _buildFundamentalRow(
                            'Debt/Equity',
                            widget.fundamentals.debtToEquity!.toStringAsFixed(2),
                            Icons.balance,
                            _getDebtColor(widget.fundamentals.debtToEquity!),
                          ),
                        if (widget.fundamentals.debtToEquity != null && widget.fundamentals.marketCap != null)
                          _buildDivider(),
                        if (widget.fundamentals.marketCap != null)
                          _buildFundamentalRow(
                            'Market Cap',
                            formatCompact(widget.fundamentals.marketCap!),
                            Icons.business,
                            AppColors.primaryBlue,
                          ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildFundamentalRow(String label, String value, IconData icon, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              size: 20,
              color: color,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w600,
                color: Colors.white70,
              ),
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w800,
              color: color,
              letterSpacing: -0.3,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildDivider() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Divider(
        color: Colors.white.withValues(alpha: 0.1),
        height: 1,
      ),
    );
  }
  
  Color _getPEColor(double pe) {
    if (pe < 15) return AppColors.successGreen;
    if (pe < 25) return AppColors.primaryBlue;
    return AppColors.dangerRed;
  }
  
  Color _getROEColor(double roe) {
    if (roe > 15) return AppColors.successGreen;
    if (roe > 10) return AppColors.primaryBlue;
    return AppColors.dangerRed;
  }
  
  Color _getDebtColor(double debt) {
    if (debt < 0.5) return AppColors.successGreen;
    if (debt < 1.0) return AppColors.primaryBlue;
    return AppColors.dangerRed;
  }
}
