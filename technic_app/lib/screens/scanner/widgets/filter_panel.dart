/// Filter Panel Widget
/// 
/// Comprehensive filtering controls for scanner configuration.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

class FilterPanel extends ConsumerStatefulWidget {
  final Map<String, String> filters;
  final ValueChanged<Map<String, String>>? onFiltersChanged;
  
  const FilterPanel({
    super.key,
    required this.filters,
    this.onFiltersChanged,
  });

  @override
  ConsumerState<FilterPanel> createState() => _FilterPanelState();
}

class _FilterPanelState extends ConsumerState<FilterPanel> {
  late Map<String, String> _filters;
  
  @override
  void initState() {
    super.initState();
    _filters = Map.from(widget.filters);
  }
  
  void _updateFilter(String key, String value) {
    setState(() {
      _filters[key] = value;
    });
    widget.onFiltersChanged?.call(_filters);
  }
  
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: tone(AppColors.darkDeep, 0.98),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Filters',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                ),
              ),
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.close),
              ),
            ],
          ),
          const SizedBox(height: 20),
          
          // Trade Style
          _sectionHeader('Trade Style'),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _filterChip('Day', 'trade_style', 'Day'),
              _filterChip('Swing', 'trade_style', 'Swing'),
              _filterChip('Position', 'trade_style', 'Position'),
            ],
          ),
          
          const SizedBox(height: 20),
          
          // Sector
          _sectionHeader('Sector'),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _filterChip('All', 'sector', ''),
              _filterChip('Technology', 'sector', 'Technology'),
              _filterChip('Healthcare', 'sector', 'Healthcare'),
              _filterChip('Financial', 'sector', 'Financial Services'),
              _filterChip('Energy', 'sector', 'Energy'),
              _filterChip('Consumer', 'sector', 'Consumer Cyclical'),
            ],
          ),
          
          const SizedBox(height: 20),
          
          // Lookback Days
          _sectionHeader('Lookback Period'),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: Slider(
                  value: double.tryParse(_filters['lookback_days'] ?? '90') ?? 90,
                  min: 30,
                  max: 365,
                  divisions: 11,
                  label: '${_filters['lookback_days'] ?? '90'} days',
                  activeColor: AppColors.skyBlue,
                  onChanged: (value) {
                    _updateFilter('lookback_days', value.round().toString());
                  },
                ),
              ),
              Text(
                '${_filters['lookback_days'] ?? '90'}d',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 20),
          
          // Min Rating
          _sectionHeader('Minimum Tech Rating'),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: Slider(
                  value: double.tryParse(_filters['min_tech_rating'] ?? '0') ?? 0,
                  min: 0,
                  max: 10,
                  divisions: 20,
                  label: _filters['min_tech_rating'] ?? '0',
                  activeColor: AppColors.skyBlue,
                  onChanged: (value) {
                    _updateFilter('min_tech_rating', value.toStringAsFixed(1));
                  },
                ),
              ),
              Text(
                _filters['min_tech_rating'] ?? '0',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 20),
          
          // Options Preference
          _sectionHeader('Options'),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            children: [
              _filterChip('Stock Only', 'options_mode', 'stock_only'),
              _filterChip('Stock + Options', 'options_mode', 'stock_plus_options'),
            ],
          ),
          
          const SizedBox(height: 24),
          
          // Apply Button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () => Navigator.pop(context, _filters),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.skyBlue,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: const Text(
                'Apply Filters',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _sectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 14,
        fontWeight: FontWeight.w700,
        color: Colors.white70,
      ),
    );
  }
  
  Widget _filterChip(String label, String filterKey, String value) {
    final isSelected = _filters[filterKey] == value;
    
    return FilterChip(
      label: Text(label),
      selected: isSelected,
      onSelected: (selected) {
        if (selected) {
          _updateFilter(filterKey, value);
        }
      },
      selectedColor: tone(AppColors.skyBlue, 0.3),
      checkmarkColor: Colors.white,
      labelStyle: TextStyle(
        color: isSelected ? Colors.white : Colors.white70,
        fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
      ),
    );
  }
}
