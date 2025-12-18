/// Premium Filter Panel Widget
/// 
/// Glass morphism filter panel with animated controls and premium design.
library;

import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';
import '../../../widgets/glass_container.dart';

class PremiumFilterPanel extends ConsumerStatefulWidget {
  final Map<String, String> filters;
  final ValueChanged<Map<String, String>>? onFiltersChanged;
  
  const PremiumFilterPanel({
    super.key,
    required this.filters,
    this.onFiltersChanged,
  });

  @override
  ConsumerState<PremiumFilterPanel> createState() => _PremiumFilterPanelState();
}

class _PremiumFilterPanelState extends ConsumerState<PremiumFilterPanel>
    with SingleTickerProviderStateMixin {
  late Map<String, String> _filters;
  late Set<String> _selectedSectors;
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  
  @override
  void initState() {
    super.initState();
    _filters = Map.from(widget.filters);
    
    // Parse selected sectors from comma-separated string
    final sectorStr = _filters['sector'] ?? '';
    _selectedSectors = sectorStr.isEmpty ? {} : sectorStr.split(',').toSet();
    
    // Setup animations
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );
    
    _fadeAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOut,
    );
    
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.1),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOutCubic,
    ));
    
    _animationController.forward();
  }
  
  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }
  
  void _updateFilter(String key, String value) {
    setState(() {
      _filters[key] = value;
    });
    widget.onFiltersChanged?.call(_filters);
  }
  
  void _toggleSector(String sector) {
    setState(() {
      if (sector.isEmpty) {
        // "All" selected - clear all sectors
        _selectedSectors.clear();
        _filters['sector'] = '';
      } else {
        // Toggle individual sector
        if (_selectedSectors.contains(sector)) {
          _selectedSectors.remove(sector);
        } else {
          _selectedSectors.add(sector);
        }
        // Update filter with comma-separated list
        _filters['sector'] = _selectedSectors.join(',');
      }
    });
    widget.onFiltersChanged?.call(_filters);
  }
  
  void _clearAllFilters() {
    setState(() {
      _filters.clear();
      _selectedSectors.clear();
    });
    widget.onFiltersChanged?.call(_filters);
  }
  
  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: SlideTransition(
        position: _slideAnimation,
        child: Container(
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.85,
          ),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                tone(AppColors.darkBackground, 0.95),
                tone(AppColors.darkBackground, 0.98),
              ],
            ),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(32)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.3),
                blurRadius: 20,
                offset: const Offset(0, -5),
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: const BorderRadius.vertical(top: Radius.circular(32)),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Header with drag handle
                  _buildHeader(),
                  
                  // Scrollable content
                  Flexible(
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Trade Style
                          _buildSection(
                            'Trade Style',
                            Icons.trending_up,
                            _buildTradeStyleChips(),
                          ),
                          
                          const SizedBox(height: 24),
                          
                          // Sectors
                          _buildSection(
                            'Sectors',
                            Icons.business,
                            _buildSectorChips(),
                          ),
                          
                          const SizedBox(height: 24),
                          
                          // Lookback Period
                          _buildSection(
                            'Lookback Period',
                            Icons.calendar_today,
                            _buildLookbackSlider(),
                          ),
                          
                          const SizedBox(height: 24),
                          
                          // Tech Rating
                          _buildSection(
                            'Minimum Tech Rating',
                            Icons.star,
                            _buildTechRatingSlider(),
                          ),
                          
                          const SizedBox(height: 24),
                          
                          // Options
                          _buildSection(
                            'Options Trading',
                            Icons.show_chart,
                            _buildOptionsChips(),
                          ),
                          
                          const SizedBox(height: 32),
                          
                          // Action Buttons
                          _buildActionButtons(),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
  
  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 16),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withValues(alpha: 0.1),
            width: 1,
          ),
        ),
      ),
      child: Column(
        children: [
          // Drag handle
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.3),
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 16),
          
          // Title and close button
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          AppColors.primaryBlue,
                          AppColors.primaryBlue.withValues(alpha: 0.6),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(
                      Icons.tune,
                      size: 20,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(width: 12),
                  const Text(
                    'Filters',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(
                    Icons.close,
                    size: 20,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
  
  Widget _buildSection(String title, IconData icon, Widget content) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(
              icon,
              size: 18,
              color: AppColors.primaryBlue,
            ),
            const SizedBox(width: 8),
            Text(
              title,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        content,
      ],
    );
  }
  
  Widget _buildTradeStyleChips() {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        _buildPremiumChip('Day', 'trade_style', 'Day', Icons.flash_on),
        _buildPremiumChip('Swing', 'trade_style', 'Swing', Icons.trending_up),
        _buildPremiumChip('Position', 'trade_style', 'Position', Icons.show_chart),
      ],
    );
  }
  
  Widget _buildSectorChips() {
    final sectors = [
      {'label': 'All', 'value': '', 'icon': Icons.grid_view},
      {'label': 'Communication', 'value': 'Communication Services', 'icon': Icons.wifi},
      {'label': 'Consumer Disc.', 'value': 'Consumer Discretionary', 'icon': Icons.shopping_bag},
      {'label': 'Consumer Staples', 'value': 'Consumer Staples', 'icon': Icons.shopping_cart},
      {'label': 'Energy', 'value': 'Energy', 'icon': Icons.bolt},
      {'label': 'Financials', 'value': 'Financials', 'icon': Icons.account_balance},
      {'label': 'Health Care', 'value': 'Health Care', 'icon': Icons.medical_services},
      {'label': 'Industrials', 'value': 'Industrials', 'icon': Icons.factory},
      {'label': 'Technology', 'value': 'Information Technology', 'icon': Icons.computer},
      {'label': 'Materials', 'value': 'Materials', 'icon': Icons.construction},
      {'label': 'Real Estate', 'value': 'Real Estate', 'icon': Icons.home},
      {'label': 'Utilities', 'value': 'Utilities', 'icon': Icons.power},
    ];
    
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: sectors.map((sector) {
        final label = sector['label'] as String;
        final value = sector['value'] as String;
        final icon = sector['icon'] as IconData;
        
        final isSelected = value.isEmpty 
            ? _selectedSectors.isEmpty 
            : _selectedSectors.contains(value);
        
        return _buildSectorChip(label, value, icon, isSelected);
      }).toList(),
    );
  }
  
  Widget _buildSectorChip(String label, String value, IconData icon, bool isSelected) {
    return GestureDetector(
      onTap: () => _toggleSector(value),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          gradient: isSelected
              ? LinearGradient(
                  colors: [
                    AppColors.primaryBlue,
                    AppColors.primaryBlue.withValues(alpha: 0.7),
                  ],
                )
              : null,
          color: isSelected ? null : Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected
                ? AppColors.primaryBlue
                : Colors.white.withValues(alpha: 0.1),
            width: 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 16,
              color: isSelected ? Colors.white : Colors.white70,
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                fontSize: 13,
                fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
                color: isSelected ? Colors.white : Colors.white70,
              ),
            ),
            if (isSelected) ...[
              const SizedBox(width: 4),
              const Icon(
                Icons.check_circle,
                size: 14,
                color: Colors.white,
              ),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildPremiumChip(String label, String filterKey, String value, IconData icon) {
    final isSelected = _filters[filterKey] == value;
    
    return GestureDetector(
      onTap: () => _updateFilter(filterKey, value),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          gradient: isSelected
              ? LinearGradient(
                  colors: [
                    AppColors.primaryBlue,
                    AppColors.primaryBlue.withValues(alpha: 0.7),
                  ],
                )
              : null,
          color: isSelected ? null : Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected
                ? AppColors.primaryBlue
                : Colors.white.withValues(alpha: 0.1),
            width: 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 18,
              color: isSelected ? Colors.white : Colors.white70,
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
                color: isSelected ? Colors.white : Colors.white70,
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildLookbackSlider() {
    final value = double.tryParse(_filters['lookback_days'] ?? '90') ?? 90;
    
    return GlassContainer(
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Days',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      AppColors.primaryBlue,
                      AppColors.primaryBlue.withValues(alpha: 0.7),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '${value.round()}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: AppColors.primaryBlue,
              inactiveTrackColor: Colors.white.withValues(alpha: 0.1),
              thumbColor: Colors.white,
              overlayColor: AppColors.primaryBlue.withValues(alpha: 0.2),
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 8),
              overlayShape: const RoundSliderOverlayShape(overlayRadius: 16),
            ),
            child: Slider(
              value: value,
              min: 30,
              max: 365,
              divisions: 11,
              onChanged: (newValue) {
                _updateFilter('lookback_days', newValue.round().toString());
              },
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '30d',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.white.withValues(alpha: 0.5),
                ),
              ),
              Text(
                '365d',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.white.withValues(alpha: 0.5),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
  
  Widget _buildTechRatingSlider() {
    final value = double.tryParse(_filters['min_tech_rating'] ?? '0') ?? 0;
    
    return GlassContainer(
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Rating',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      AppColors.primaryBlue,
                      AppColors.primaryBlue.withValues(alpha: 0.7),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.star, size: 14, color: Colors.white),
                    const SizedBox(width: 4),
                    Text(
                      value.toStringAsFixed(1),
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: AppColors.primaryBlue,
              inactiveTrackColor: Colors.white.withValues(alpha: 0.1),
              thumbColor: Colors.white,
              overlayColor: AppColors.primaryBlue.withValues(alpha: 0.2),
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 8),
              overlayShape: const RoundSliderOverlayShape(overlayRadius: 16),
            ),
            child: Slider(
              value: value,
              min: 0,
              max: 10,
              divisions: 20,
              onChanged: (newValue) {
                _updateFilter('min_tech_rating', newValue.toStringAsFixed(1));
              },
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '0.0',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.white.withValues(alpha: 0.5),
                ),
              ),
              Text(
                '10.0',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.white.withValues(alpha: 0.5),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
  
  Widget _buildOptionsChips() {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        _buildPremiumChip('Stock Only', 'options_mode', 'stock_only', Icons.trending_up),
        _buildPremiumChip('Stock + Options', 'options_mode', 'stock_plus_options', Icons.show_chart),
      ],
    );
  }
  
  Widget _buildActionButtons() {
    return Row(
      children: [
        // Clear button
        Expanded(
          child: GestureDetector(
            onTap: _clearAllFilters,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: Colors.white.withValues(alpha: 0.1),
                  width: 1,
                ),
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.clear_all, size: 20, color: Colors.white70),
                  SizedBox(width: 8),
                  Text(
                    'Clear All',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: Colors.white70,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        
        const SizedBox(width: 12),
        
        // Apply button
        Expanded(
          flex: 2,
          child: GestureDetector(
            onTap: () => Navigator.pop(context, _filters),
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    AppColors.primaryBlue,
                    AppColors.primaryBlue.withValues(alpha: 0.8),
                  ],
                ),
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: AppColors.primaryBlue.withValues(alpha: 0.3),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.check_circle, size: 20, color: Colors.white),
                  SizedBox(width: 8),
                  Text(
                    'Apply Filters',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}
